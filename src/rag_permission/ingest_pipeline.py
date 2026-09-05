from pathlib import Path

from rag_permission.embeddings import EmbeddingBackend
from rag_permission.ingest import chunk_document, parse_document
from rag_permission.bm25 import BM25Index
from rag_permission.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, embedding: EmbeddingBackend, vector_store: VectorStore, bm25: BM25Index):
        self.embedding = embedding
        self.vector_store = vector_store
        self.bm25 = bm25

    def ingest(
        self,
        path: Path,
        acl_groups: tuple[str, ...],
        strategy: str = "parent_child",
        chunk_size: int = 260,
        overlap: int = 48,
        parent_chunk_size: int = 1000,
    ) -> str:
        parsed = parse_document(path, acl_groups)
        chunks = chunk_document(
            parsed,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            parent_chunk_size=parent_chunk_size,
        )
        if chunks:
            vectors = self.embedding.encode([chunk.text for chunk in chunks])
            self.vector_store.upsert(chunks, vectors)
            self.bm25.add_documents(chunks)
        return parsed.doc_id
