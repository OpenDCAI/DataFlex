import os
import json
import numpy as np
import faiss
import heapq
# ===== auto optional embedding backends =====
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False
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

class offline_near_Selector:
    def __init__(self,
                 candidate_path = None,
                 query_path: str =  None,
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 embed_method: str ="auto",
                 batch_size: int = 32,
                 save_indices_path: str = "top_indices.npy",
                 max_K: int = 1000):
    
        self.candidate_path = candidate_path
        self.query_path = query_path
        self.embed_model = embed_model
        self.embed_method = embed_method
        self.batch_size = batch_size
        self.save_indices_path = save_indices_path
        self.max_K = max_K

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
        '''
        auto模式自动尝试 embedding 后端：
        1) 优先 vLLM
        2) 否则 sentence-transformers
        3) 都不可用则报错
        '''

        # -------- 1. 优先 vLLM --------
        if (VLLM_AVAILABLE and self.embed_method == "auto") or self.embed_method == "vllm":
            try:
                logger.info(f"[EMBED] Using vLLM model: {self.embed_model}")
                llm = LLM(model=self.embed_model, trust_remote_code=True, task="embed")

                outputs = llm.embed(texts)  # [N, D]
                embs = [o.outputs.embedding for o in outputs]
                embs = np.array(embs, dtype=np.float32)

                # normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms

                return np.ascontiguousarray(embs)

            except Exception as e:
                logger.warning(f"[EMBED] vLLM available but embedding failed {e}")

        # -------- 2. fallback: sentence-transformers --------
        if (ST_AVAILABLE and self.embed_method == "auto") or self.embed_method == "sentence-transformer":
            try:
                logger.info(f"[EMBED] Using SentenceTransformer: {self.embed_model}")
                model = SentenceTransformer(self.embed_model)
                embs = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True
                ).astype(np.float32)

                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms

                return np.ascontiguousarray(embs)

            except Exception as e:
                raise RuntimeError(
                    f"SentenceTransformer available but embedding failed: {e}"
                )

        # -------- 3. 两个都不可用 --------
        raise RuntimeError(
            "No available embedding backend!\n"
            "Please install at least one of the following:\n"
            "  - vLLM: pip install vllm\n"
            "  - sentence-transformers: pip install sentence-transformers"
        )

    # ---------- 调用接口 ----------
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

    # ---------- 主程序 ----------
    def selector(self):
        
        logger.info("Start loading embeddings for nearest_selector...")
        xb = self.candidate_sentence_embedding()  
        xq = self.query_sentence_embedding()      

        M, N = xq.shape[0], xb.shape[0]
        MAX_K = self.max_K
        
        logger.info("Building FAISS index...")
        index = FaissIndexIVFFlat(xb)

        logger.info(f"Searching top-{MAX_K} neighbors for each query...")
        top_dists2, top_indices = index.search(xq, MAX_K)
        sorted_indices = np.argsort(top_dists2, axis=-1)
        static_idx = np.indices(top_dists2.shape)[0]
        # top_dists = np.sqrt(top_dists2[static_idx, sorted_indices])
        top_indices = top_indices[static_idx, sorted_indices].astype(int)

        #保存每个query最近的前MAX_K个索引
        assert isinstance(top_indices, np.ndarray), \
            "top_indices must be a numpy array"
        assert top_indices.shape == (M, MAX_K), \
            f"Expected shape {(M, MAX_K)}, got {top_indices.shape}"
        assert np.issubdtype(top_indices.dtype, np.integer), \
            f"Expected integer dtype, got {top_indices.dtype}"

        np.save(self.save_indices_path, top_indices)
        logger.info(f"Saved to  {self.save_indices_path}")


if __name__ == "__main__":
    near = offline_near_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = train
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = vaildation
        # It automatically try vllm first, then sentence-transformers
        embed_model="Qwen/Qwen3-Embedding-0.6B",
        # support method:
        #auto(It automatically try vllm first, then sentence-transformers),
        #vllm,
        #sentence-transformer
        embed_method= "auto",
        batch_size=32,
        save_indices_path="top_indices.npy",
        max_K=1000,
        
    )
    near.selector()