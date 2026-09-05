import math
import tempfile
from pathlib import Path

from rag_permission.bm25 import BM25Index
from rag_permission.ingest import chunk_document, parse_document
from rag_permission.models import User
from rag_permission.retriever import HybridRetriever
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


SAMPLE = """# 设备手册
## 故障码
E-1002 是风扇故障，需要检查电源。
E-1001 是传感器故障。
"""


class DemoEmbedding:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for token in text.replace("，", " ").split():
                vector[hash(token) % len(vector)] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.md"
        path.write_text(SAMPLE, encoding="utf-8")
        chunks = chunk_document(parse_document(path, ("public",)), "parent_child")
        embedding = DemoEmbedding()
        store = QdrantVectorStore(create_qdrant_client(":memory:"), "demo", 64)
        store.upsert(chunks, embedding.encode([chunk.text for chunk in chunks]))
        bm25 = BM25Index()
        bm25.add_documents(chunks)
        retriever = HybridRetriever(embedding, store, bm25, final_top_k=2)
        public = User(id="u1", groups=frozenset({"public"}))
        nobody = User(id="u2")
        for user in (public, nobody):
            hits = retriever.search("E-1002 风扇故障", user)
            print(
                f"user={user.id}, hits={len(hits)}, "
                f"ids={[hit.chunk.chunk_id for hit in hits]}, "
                f"bm25_tokens={hits[0].bm25_rank if hits else None}"
            )


if __name__ == "__main__":
    main()
