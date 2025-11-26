import os
import json
import numpy as np
import faiss
import heapq
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
from sentence_transformers import SentenceTransformer
from dataflex.utils.logging import logger

# ========== FAISS IVFFlat 索引封装类 ==========
class FaissIndexIVFFlat:
    def __init__(self, data: np.ndarray, nprobe: int = 10):
        self.build(data, nprobe)

    def build(self, data: np.ndarray, nprobe: int):
        data = np.ascontiguousarray(data.astype(np.float32))
        N, D = data.shape
        nlist = max(1, int(np.sqrt(N)) // 2)
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist)
        index.train(data)
        index.add(data)
        index.nprobe = nprobe
        self.index = index

    def search(self, query: np.ndarray, K: int):
        query = np.ascontiguousarray(query.astype(np.float32))
        return self.index.search(query, K)

class offline_tsds_Selector:
    def __init__(self,
                 candidate_path = None,
                 query_path: str = None,
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 batch_size: int = 32,
                 save_probs_path: str = "tsds_probs.npy",
                 max_K: int = 5000,
                 kde_K: int = 1000,
                 sigma: float = 0.75,
                 alpha: float = 0.6,
                 C: float = 5.0):
        
        self.candidate_path = candidate_path
        self.query_path = query_path
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.save_probs_path = save_probs_path
        self.max_K = max_K
        self.kde_K = kde_K
        self.sigma = sigma
        self.alpha = alpha
        self.C = C

    # ---------- 数据加载方法 ----------
    def _load_alpaca_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [
            "\n".join([
                f"Instruction: {item.get('instruction', '')}",
                f"Input: {item.get('input', '')}",
                f"Output: {item.get('output', '')}",
                f"Prediction:{item.get('prediction','')}"
            ])
            for item in data
        ]
        return texts

    # ---------- Embedding 方法 ----------
    def _embed_texts(self, texts):
        if VLLM_AVAILABLE and self.embed_model.startswith("vllm:"):
            model_name = self.embed_model.replace("vllm:", "")
            logger.info(f"[EMBED] vLLM model: {model_name}")
            llm = LLM(model=model_name, trust_remote_code=True, task="embed")
            
            # 使用 vLLM 的 embed 接口
            outputs = llm.embed(texts)  # 返回 [N, D]
            print(f"Embeddings shape: {np.array(outputs).shape}", outputs[0])
            embs = [o.outputs.embedding for o in outputs]
            embs = np.array(embs, dtype=np.float32)
        else:
            logger.info(f"[EMBED] SentenceTransformer: {self.embed_model}")
            model = SentenceTransformer(self.embed_model)
            embs = model.encode(texts,
                                batch_size=self.batch_size,
                                show_progress_bar=True).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)  # [N, 1]
        # 防止除以 0
        norms = np.maximum(norms, 1e-12)
        embs = embs / norms
        # --------------------------------------    
        return np.ascontiguousarray(embs)

    # ---------- TSDS 调用接口 ----------
    def candidate_sentence_embedding(self):
        texts = self._load_alpaca_json(self.candidate_path)
        logger.info(f"Loaded {len(texts)} candidates")
        return self._embed_texts(texts)

    def query_sentence_embedding(self):
        if self.query_path and os.path.exists(self.query_path):
            texts = self._load_alpaca_json(self.query_path)
            logger.info(f"Loaded {len(texts)} queries from query json")
        else:
            logger.info("No query set provided — using first 100 candidates as queries")
            texts = self._load_alpaca_json(self.candidate_path)[:100]
        return self._embed_texts(texts)

    # ---------- tsds 主程序 ----------
    def selector(self):
        
        logger.info("Start loading embeddings for TSDS...")
        xb = self.candidate_sentence_embedding()  
        xq = self.query_sentence_embedding()      

        M, N = xq.shape[0], xb.shape[0]
        MAX_K = self.max_K
        KDE_K = self.kde_K

        logger.info("Building FAISS index...")
        index = FaissIndexIVFFlat(xb)

        logger.info(f"Searching top-{MAX_K} neighbors for each query...")
        top_dists2, top_indices = index.search(xq, MAX_K)
        sorted_indices = np.argsort(top_dists2, axis=-1)
        static_idx = np.indices(top_dists2.shape)[0]
        top_dists = np.sqrt(top_dists2[static_idx, sorted_indices])
        top_indices = top_indices[static_idx, sorted_indices].astype(int)

        if self.sigma == 0:
            logger.info("Sigma=0, using unity KDE")
            top_kdes = np.ones_like(top_indices)
        else:
            logger.info("Computing KDE...")
            uniq_ids = list(set(top_indices.reshape(-1)))
            sub_xb = xb[uniq_ids]
            index_kde = FaissIndexIVFFlat(sub_xb)
            D2, _ = index_kde.search(sub_xb, KDE_K)
            kernel = np.maximum(0.0, 1 - D2 / (self.sigma ** 2))
            kde = kernel.sum(axis=1)
            kde_map = {uniq_ids[i]: kde[i] for i in range(len(uniq_ids))}
            top_kdes = np.vectorize(lambda t: kde_map[t])(top_indices)

        logger.info("Running heap optimization for TSDS...")
        lastK = [0] * M
        heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(M)]
        heapq.heapify(heap)
        cost = np.zeros(M)
        dist_wsum = [top_dists[j][0] / top_kdes[j][0] for j in range(M)]
        total_cost, s = 0.0, 0.0

        while heap:
            count, k, j = heapq.heappop(heap)
            s = count
            total_cost -= cost[j]
            cost[j] = top_dists[j][k + 1] * count - dist_wsum[j]
            total_cost += cost[j]
            if self.alpha / self.C * total_cost >= (1 - self.alpha) * M:
                break
            lastK[j] = k
            if k < MAX_K - 2:
                count += 1.0 / top_kdes[j][k + 1]
                heapq.heappush(heap, (count, k + 1, j))
                dist_wsum[j] += top_dists[j][k + 1] / top_kdes[j][k + 1]

        logger.info("Calculating global sampling probability...")
        global_probs = np.zeros(N, dtype=np.float64)
        inv_M = 1.0 / M
        for j in range(M):
            psum = 0.0
            for k in range(lastK[j] + 1):
                w = inv_M / s / top_kdes[j][k]
                global_probs[top_indices[j][k]] += w
                psum += w
            global_probs[top_indices[j][lastK[j] + 1]] += max(inv_M - psum, 0)

        global_probs = np.maximum(global_probs, 0)
        global_probs /= global_probs.sum()

        # 保存为 npy 文件
        np.save(self.save_probs_path, global_probs)
        logger.info(f"TSDS probs saved to {self.save_probs_path}")

        return global_probs


if __name__ == "__main__":
    tsds = offline_tsds_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w",
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w",

        # If you want to use vllm,please add "vllm:" before model's name
        # Otherwise it automatically use sentence-transfromer
        embed_model="vllm:Qwen/Qwen3-Embedding-0.6B",
        batch_size=32,
        save_probs_path="tsds_probs.npy",
        max_K=5000,
        kde_K=1000,
        sigma=0.75,
        alpha=0.6,
        C=5.0
    )
    tsds.selector()