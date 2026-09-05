import threading
from dataclasses import replace
from typing import Protocol

from sentence_transformers import CrossEncoder

from rag_permission.models import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...


_reranker_cache: dict[str, CrossEncoder] = {}
_reranker_lock = threading.Lock()


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        with _reranker_lock:
            if model_name not in _reranker_cache:
                _reranker_cache[model_name] = CrossEncoder(model_name, max_length=512)
            self.model = _reranker_cache[model_name]

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.model.predict([(query, hit.chunk.text) for hit in hits])
        scored = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(scored, key=lambda hit: -(hit.rerank_score or 0.0))
