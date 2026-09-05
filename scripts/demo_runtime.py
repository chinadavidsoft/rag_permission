import tempfile
from pathlib import Path

from rag_permission.bm25 import BM25Index
from rag_permission.config import Settings
from rag_permission.embeddings import BGEEmbedding
from rag_permission.feedback import FeedbackStore
from rag_permission.ingest_pipeline import IngestionPipeline
from rag_permission.llm import LLMResponse, LLMUsage
from rag_permission.models import User
from rag_permission.observability import InMemoryTracer
from rag_permission.reranker import BGEReranker
from rag_permission.retriever import HybridRetriever
from rag_permission.runtime import RAGService
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


class CiteFirstLLM:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse("根据资料回答，并标注 [1]。", LLMUsage(prompt_tokens=10, completion_tokens=5))


def main() -> None:
    settings = Settings()
    embedding = BGEEmbedding(settings.embedding_model)
    reranker = BGEReranker(settings.reranker_model) if settings.enable_rerank else None
    store = QdrantVectorStore(create_qdrant_client(":memory:"), settings.collection_name, 1024)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(embedding, store, bm25)
    for filename, groups in (
        ("sample.md", ("public",)),
        ("sample.docx", ("eng",)),
        ("sample.pdf", ("hr",)),
    ):
        pipeline.ingest(settings.fixture_dir / filename, groups)
    retriever = HybridRetriever(
        embedding,
        store,
        bm25,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        reranker=reranker,
    )
    with tempfile.TemporaryDirectory() as directory:
        service = RAGService(retriever, CiteFirstLLM(), FeedbackStore(Path(directory) / "db.sqlite"), InMemoryTracer())
        public = service.ask("E-1002 是什么故障？", User("public-user", frozenset({"public"})))
        denied = service.ask("劳动法规考试占比多少？", User("eng-user", frozenset({"eng"})))
        print(f"public_hits={len(public.citations)}, trace={public.trace_id}")
        print(f"denied_answer={denied.answer}, leakage={bool(denied.citations)}")
        print(f"latency={service.tracer.latency_percentiles() if hasattr(service.tracer, 'latency_percentiles') else {}}")


if __name__ == "__main__":
    main()
