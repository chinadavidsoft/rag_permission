from dataclasses import replace
from typing import Protocol

from rag_permission.bm25 import BM25Index
from rag_permission.embeddings import EmbeddingBackend
from rag_permission.models import SearchHit, User
from rag_permission.vector_store import VectorStore


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...


def rrf_fuse(
    dense_hits: list[SearchHit], bm25_hits: list[SearchHit], k: int = 60, limit: int = 10
) -> list[SearchHit]:
    dense_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(dense_hits, 1)}
    bm25_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(bm25_hits, 1)}
    fused: dict[str, SearchHit] = {}
    for rank, hit in enumerate(dense_hits, 1):
        existing = fused.setdefault(hit.chunk.chunk_id, replace(hit, score=0.0))
        fused[hit.chunk.chunk_id] = replace(existing, score=existing.score + 1 / (k + rank))
    for rank, hit in enumerate(bm25_hits, 1):
        existing = fused.setdefault(hit.chunk.chunk_id, replace(hit, score=0.0))
        fused[hit.chunk.chunk_id] = replace(existing, score=existing.score + 1 / (k + rank))
    ranked = sorted(fused.values(), key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:limit]
    return [
        replace(
            hit,
            dense_rank=dense_ranks.get(hit.chunk.chunk_id),
            bm25_rank=bm25_ranks.get(hit.chunk.chunk_id),
        )
        for hit in ranked
    ]


class HybridRetriever:
    def __init__(
        self,
        embedding: EmbeddingBackend,
        vector_store: VectorStore,
        bm25: BM25Index,
        dense_top_k: int = 24,
        bm25_top_k: int = 24,
        final_top_k: int = 8,
        reranker: Reranker | None = None,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.bm25 = bm25
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.reranker = reranker

    def search(self, query: str, user: User) -> list[SearchHit]:
        if not user.groups:
            return []
        dense_hits = self.vector_store.search(
            self.embedding.encode([query])[0], self.dense_top_k, user.groups
        )
        dense_hits = [replace(hit, score=score) for hit, score in dense_hits]
        bm25_hits = self.bm25.search(query, self.bm25_top_k, user.groups)
        candidates = rrf_fuse(dense_hits, bm25_hits, k=60, limit=self.final_top_k)
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates)
        return candidates
