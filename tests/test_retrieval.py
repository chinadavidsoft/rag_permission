import math
from pathlib import Path

from rag_permission.bm25 import BM25Index, tokenize
from rag_permission.ingest import chunk_document, parse_document
from rag_permission.models import User
from rag_permission.retriever import HybridRetriever, rrf_fuse
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


class DeterministicEmbedding:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.replace("，", " ").replace("。", " ").split():
                vector[hash(token) % len(vector)] += 1
            norm = math.sqrt(sum(value * value for value in vector)) or 1
            vectors.append([value / norm for value in vector])
        return vectors


def _chunks(groups="public"):
    parsed = parse_document(Path("fixtures/sample.md"), (groups,))
    return chunk_document(parsed, "section", chunk_size=1000)


def test_tokenizer_preserves_error_code():
    assert "e-1002" in tokenize("E-1002 风扇故障")


def test_bm25_post_filters_permissions():
    index = BM25Index()
    index.add_documents(_chunks())
    assert index.search("E-1002", 5, frozenset({"eng"})) == []
    assert index.search("E-1002", 5, frozenset({"public"}))


def test_qdrant_dense_prefilters_permissions():
    embedding = DeterministicEmbedding()
    chunks = _chunks()
    store = QdrantVectorStore(create_qdrant_client(":memory:"), "acl-test", 32)
    store.upsert(chunks, embedding.encode([chunk.text for chunk in chunks]))
    query = embedding.encode(["E-1002 风扇"])[0]
    assert store.search(query, 5, frozenset({"eng"})) == []
    assert store.search(query, 5, frozenset({"public"}))


def test_point_id_is_deterministic():
    store = QdrantVectorStore(create_qdrant_client(":memory:"), "point-id", 1)
    assert store.point_id("same") == store.point_id("same")


def test_rrf_fuses_and_deduplicates():
    hit = _chunks()[0]
    from rag_permission.models import SearchHit
    dense = [SearchHit(chunk=hit, score=1.0)]
    sparse = [SearchHit(chunk=hit, score=1.0)]
    fused = rrf_fuse(dense, sparse)
    assert len(fused) == 1
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 1


def test_hybrid_retriever_denies_empty_user():
    embedding = DeterministicEmbedding()
    chunks = _chunks()
    store = QdrantVectorStore(create_qdrant_client(":memory:"), "empty-user", 32)
    store.upsert(chunks, embedding.encode([chunk.text for chunk in chunks]))
    index = BM25Index()
    index.add_documents(chunks)
    retriever = HybridRetriever(embedding, store, index)
    assert retriever.search("E-1002", User(id="u")) == []
