import os
import faiss
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# ── Absolute path: index always lives next to research_workflow.db ──
_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
_DEFAULT_INDEX = os.path.join(_DB_DIR, "vector_index.faiss")


class VectorDBService:
    def __init__(self, index_path: str = _DEFAULT_INDEX,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = os.path.abspath(index_path)
        self.map_path   = self.index_path + ".map.npy"
        self.model      = SentenceTransformer(model_name)
        self.dimension  = self.model.get_sentence_embedding_dimension()
        self.index      = None
        # id_map: vector_id → {"paper_id": int, "workflow_id": int}
        self.id_map: Dict[int, Dict] = {}
        self.next_id    = 0
        self._load_or_initialize()

    # ── Init ────────────────────────────────────────────────────────
    def _load_or_initialize(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self.index  = faiss.read_index(self.index_path)
            self.id_map = np.load(self.map_path, allow_pickle=True).item()
            self.next_id = (max(self.id_map.keys()) + 1) if self.id_map else 0
            print(f"[VectorDB] Loaded index: {self.index.ntotal} vectors, "
                  f"{len(self.id_map)} mapped")
        else:
            self.index   = faiss.IndexFlatL2(self.dimension)
            self.id_map  = {}
            self.next_id = 0
            print(f"[VectorDB] New index created at {self.index_path}")

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        np.save(self.map_path, self.id_map)

    # ── Embed ───────────────────────────────────────────────────────
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        return embeddings.astype("float32")

    # ── Add ─────────────────────────────────────────────────────────
    def add_papers(self, papers: List[Dict]):
        """Deprecated — use add_knowledge_records(). Kept for index compat."""
        self.add_knowledge_records(papers)

    def add_knowledge_records(self, records: List[Dict]):
        """
        records: [{
            knowledge_id, paper_id, workflow_id, category, value,
            sentence, doi, title, year, text
        }]
        """
        if not records:
            return
        texts      = [r["text"] for r in records]
        embeddings = self.embed_texts(texts)
        self.index.add(embeddings)
        for i, r in enumerate(records):
            self.id_map[self.next_id + i] = {
                "knowledge_id": r.get("knowledge_id"),
                "paper_id":     r["paper_id"],
                "workflow_id":  r.get("workflow_id", 0),
                "category":     r.get("category", ""),
                "value":        r.get("value", ""),
                "sentence":     r.get("sentence", ""),
                "doi":          r.get("doi", ""),
                "title":        r.get("title", ""),
                "year":         r.get("year"),
            }
        self.next_id += len(records)
        self._save()
        print(f"[VectorDB] Added {len(records)} vectors — "
              f"total={self.index.ntotal}")

    # ── Search ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5,
               workflow_id: Optional[int] = None,
               return_scores: bool = False):
        """
        workflow_id=None  → search across all workflows
        workflow_id=N     → filter results to workflow N only
        """
        if self.index.ntotal == 0:
            return []
        query_vec = self.embed_texts([query])
        # Over-fetch if filtering so we still get top_k after filter
        fetch_k   = top_k * 5 if workflow_id is not None else top_k
        fetch_k   = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, fetch_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx not in self.id_map:
                continue
            entry = self.id_map[idx]
            if isinstance(entry, int):
                paper_id = entry
                wflow_id = None
            else:
                paper_id = entry["paper_id"]
                wflow_id = entry.get("workflow_id")
            if workflow_id is not None and wflow_id != workflow_id:
                continue
            similarity = float(1 / (1 + distances[0][i]))
            if return_scores:
                results.append((paper_id, similarity))
            else:
                results.append({"paper_id": paper_id, "workflow_id": wflow_id, "score": similarity})
            if len(results) >= top_k:
                break
        return results

    # ── Stats ───────────────────────────────────────────────────────
    def stats(self) -> Dict:
        workflow_counts: Dict[int, int] = {}
        for entry in self.id_map.values():
            wid = entry.get("workflow_id", 0)
            workflow_counts[wid] = workflow_counts.get(wid, 0) + 1
        return {
            "total_vectors": self.index.ntotal,
            "total_mapped":  len(self.id_map),
            "by_workflow":   workflow_counts,
            "index_path":    self.index_path,
        }

    # ── Reset ───────────────────────────────────────────────────────
    def reset(self):
        self.index   = faiss.IndexFlatL2(self.dimension)
        self.id_map  = {}
        self.next_id = 0
        self._save()
        print("[VectorDB] Index reset.")
