from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import heapq
import torch
import faiss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,DataCollatorWithPadding,AutoModel

# Faiss IVFFlat索引封装类
class FaissIndexIVFFlat:
    def __init__(self, data, nprobe=10):
        self.build(data, nprobe)

    def build(self, data, nprobe):
        nlist = int(np.sqrt(data.shape[0])) // 2
        quantizer = faiss.IndexFlatL2(data.shape[-1])
        self.index = faiss.IndexIVFFlat(quantizer, data.shape[-1], nlist)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = nprobe

    def search(self, query, K):
        return self.index.search(query, K)

@register_selector('tsds')
class TsdsSelector(Selector):
    # TSDS选择器
    def __init__(self, 
                 dataset, 
                 eval_dataset,
                 accelerator, 
                 data_collator,
                 cache_dir,
                 seed: int = 42,
                 max_K: int = 128,
                 kde_K: int = 64,
                 sigma: float = 1.0,
                 alpha: float = 0.5,
                 C: float = 10.0,
                 sample_size: int = 1000,
                 ###需要另外加载模型
                 model_name: str = "/home/lianghao/yry/TSDS/bert_chinese"):
        
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.eval_dataset = eval_dataset
        # tsds可调超参
        self.max_K = max_K
        self.kde_K = kde_K
        self.sigma = sigma
        self.alpha = alpha
        self.C = C

        self.sample_size = sample_size
        self.seed = seed
        self.device = self.accelerator.device
        self.dtype = torch.float16

        # 数据编码模型
        self.model_name = model_name

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"TsdsSelector initialized. Projected gradients will be saved in {self.cache_dir}")

        # ==== 编码候选文本 ====
    def candidate_sentence_embedding(self, batch_size=32):
        embeddings_list = []
        self.dataset = self.dataset.remove_columns(["labels", "images", "videos", "audios"])
        model = AutoModel.from_pretrained(self.model_name).to("cuda")
        model.eval()
        # 创建collator（自动pad）
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # token长度限制在512以内 
            input_ids = input_ids.long()
            max_len = 512
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            # 替换小于0或大于等于vocab_size的token为pad_token_id
            input_ids[input_ids < 0] = tokenizer.pad_token_id
            input_ids[input_ids >= tokenizer.vocab_size] = tokenizer.pad_token_id
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            # 计算句向量 
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # 平均池化得到句向量（忽略 padding token）
                last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                mask = attention_mask.unsqueeze(-1)     # [batch_size, seq_len, 1]
                batch_embeddings = (last_hidden * mask).sum(dim=1)
                batch_embeddings = batch_embeddings / mask.sum(dim=1).clamp(min=1)
            embeddings_list.append(batch_embeddings.cpu())
        # 合并 batch，转成tsds需要的numpy格式
        embeddings = torch.cat(embeddings_list, dim=0)  # [num_samples, hidden_dim]
        return embeddings.numpy()
    
        # ==== 编码查询文本 ====
    def query_sentence_embedding(self, batch_size=32):
        embeddings_list = []
        self.eval_dataset = self.eval_dataset.remove_columns(["labels", "images", "videos", "audios"])
        model = AutoModel.from_pretrained(self.model_name).to("cuda")
        model.eval()
        # 创建collator（自动pad）
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        dataloader = DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # token长度限制在512以内
            input_ids = input_ids.long()
            max_len = 512
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            # 替换小于0或大于等于vocab_size的token为pad_token_id
            input_ids[input_ids < 0] = tokenizer.pad_token_id
            input_ids[input_ids >= tokenizer.vocab_size] = tokenizer.pad_token_id
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # 计算句向量 
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # 平均池化得到句向量（忽略 padding token）
                last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                mask = attention_mask.unsqueeze(-1)     # [batch_size, seq_len, 1]
                batch_embeddings = (last_hidden * mask).sum(dim=1)
                batch_embeddings = batch_embeddings / mask.sum(dim=1).clamp(min=1)

            embeddings_list.append(batch_embeddings.cpu())

        # 合并 batch，转成tsds需要的numpy格式
        embeddings = torch.cat(embeddings_list, dim=0)  # [num_samples, hidden_dim]
        return embeddings.numpy()

    def select(self, model, step_id: int, num_samples: int, **kwargs):

        SAMPLE_SIZE = self.sample_size
        MAX_K = self.max_K
        KDE_K = self.kde_K
        SIGMA = self.sigma
        ALPHA = self.alpha
        C = self.C
        self.num_samples = num_samples
        
        #==== 1.加载嵌入候选向量====
        xb = self.candidate_sentence_embedding().astype(np.float32)
        assert xb.ndim == 2, f"Embeddings of the candidates should be in the form of a 2D array."
        logger.info(f"number of candidates: {xb.shape[0]}, embedding dimension: {xb.shape[1]}")

        #==== 2.加载嵌入训练向量====
        xq = self.query_sentence_embedding().astype(np.float32)
        assert xq.ndim == 2, f"Embeddings of the query examples should be in the form of a 2D array."
        logger.info(f"number of query examples: {xq.shape[0]}, embedding dimension: {xq.shape[1]}")
        MAX_K = min(MAX_K, xb.shape[0] // 10)
        KDE_K = min(KDE_K, xb.shape[0] // 10) 

        # ==== 3.候选样本集建立一个快速相似搜索索引 =====
        logger.info(f"Starting building index for the candidate examples.")
        index = FaissIndexIVFFlat(xb)

        #==== 4.对每个目标样本，找出其最近的 MAX_K 个邻居，并计算 KDE =====
        logger.info(f"Start prefetching {MAX_K}-nearest neighbors for each query example.")
        top_dists, top_indices = index.search(xq, MAX_K)
        top_indices = top_indices.astype(int)
        sorted_indices = np.argsort(top_dists, axis=-1)
        static_indices = np.indices(top_dists.shape)[0]
        top_dists = np.sqrt(top_dists[static_indices, sorted_indices])
        # top_indices[i][j] is the index of the jth nearest neighbor
        # (among the candidates) of the ith query example
        top_indices = top_indices[static_indices, sorted_indices]
        # top_kde[i][j] is the KDE of the jth nearest neighbor of the ith query example
        if SIGMA == 0:
            logger.info("Sigma is zero, KDE (kernel density estimation) set to 1 for all the points.")
            top_kdes = np.ones_like(top_indices)
        else:
            logger.info(f"Start computing KDE (kernel density estimation), neighborhood size: {KDE_K}.")
            top_indices_set = list(set([i for i in top_indices.reshape(-1)]))
            top_features = xb[top_indices_set]
            index_for_kde = FaissIndexIVFFlat(top_features)
            D2, I = index_for_kde.search(top_features, KDE_K)
            kernel = 1 - D2 / (SIGMA ** 2)
            logger.info(f'A point has {(kernel > 0).sum(axis=-1).mean() - 1} near-duplicates on average.')
            kernel = kernel * (kernel > 0)
            kde = kernel.sum(axis=-1)
            kde_map = {top_indices_set[i]:kde[i] for i in range(len(top_indices_set))}
            kde_mapfunc = np.vectorize(lambda t: kde_map[t])
            top_kdes = kde_mapfunc(top_indices)
        #==== 5.计算概率分配 =====       
        logger.info("Start computing the probability assignment.")
        M, N = top_indices.shape[0], xb.shape[0]
        lastK = [0] * M
        heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(M)]
        heapq.heapify(heap)
        dist_weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(M)]
        s = 0
        cost = np.zeros(M)
        total_cost = 0
        while len(heap) > 0:
            count, curr_k, curr_j = heapq.heappop(heap)
            s = count
            # if we increase s by any positive amount, the 0, 1, ..., curr_k has to transport probability mass to curr_k + 1
            total_cost -= cost[curr_j]
            cost[curr_j] = top_dists[curr_j][curr_k + 1] * count - dist_weighted_sum[curr_j]
            total_cost += cost[curr_j]
            # If the condition breaks, the current s will be the final s
            if ALPHA / C * total_cost >= (1 - ALPHA) * M:
                break
            lastK[curr_j] = curr_k
            if curr_k < MAX_K - 2:
                count += 1.0 / top_kdes[curr_j][curr_k + 1]
                heapq.heappush(heap, (count, curr_k + 1, curr_j))
                dist_weighted_sum[curr_j] += top_dists[curr_j][curr_k + 1] / top_kdes[curr_j][curr_k + 1]
        global_probs = np.zeros(N)
        for j in range(M):
            prob_sum = 0
            for k in range(lastK[j] + 1):
                global_probs[top_indices[j][k]] += 1 / M / s / top_kdes[j][k]
                prob_sum += 1 / M / s / top_kdes[j][k]
            global_probs[top_indices[j][lastK[j] + 1]] += max(1.0 / M - prob_sum, 0)
            assert 1.0 / M - prob_sum >= -1e-6, f'{1.0 / M - prob_sum}'
            assert (1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s <= 1 + 1e-6 or lastK[j] == MAX_K - 2, f'{(1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s}'
        
        #==== 6.按概率分布进行采样 =====
        logger.info(f"Start sampling. Sample size: {SAMPLE_SIZE}.")
        global_probs = np.maximum(global_probs, 0)  # 去掉负数
        global_probs = global_probs / np.sum(global_probs)  # 重新归一化为概率分布
        sample_times = np.random.multinomial(SAMPLE_SIZE, global_probs)
        sample_indices = []
        for i in range(sample_times.shape[0]):
            sample_indices.extend([i] * sample_times[i])
        
        #==== 7.返回最终筛选出的数据索引====
        logger.info(f"Saving indices of the selected candidates.")
        return sample_indices
