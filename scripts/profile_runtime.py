import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from rag_permission.bm25 import BM25Index
from rag_permission.config import Settings
from rag_permission.embeddings import BGEEmbedding
from rag_permission.evaluation.runner import load_golden_set
from rag_permission.feedback import FeedbackStore
from rag_permission.ingest_pipeline import IngestionPipeline
from rag_permission.llm import LLMClient, LLMResponse, LLMUsage, OpenAILLMClient
from rag_permission.models import User
from rag_permission.observability import InMemoryTracer
from rag_permission.retriever import HybridRetriever
from rag_permission.runtime import RAGService
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


class StableGroundedLLM(LLMClient):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse("这是剖面模式的固定回答 [1]。", LLMUsage(prompt_tokens=180, completion_tokens=48))

    def stream(self, system: str, user: str, temperature: float = 0.0):
        raise AssertionError("profile_runtime uses non-streaming requests")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile segmented RAG runtime latency and cost")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--golden-set", default="fixtures/golden_set.json")
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    settings = Settings()
    started = time.perf_counter()
    embedding = BGEEmbedding(settings.embedding_model)
    store = QdrantVectorStore(create_qdrant_client(":memory:"), settings.collection_name, 1024)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(embedding, store, bm25)
    for filename, groups in (
        ("sample.md", ("public",)),
        ("sample.docx", ("eng",)),
        ("sample.pdf", ("hr",)),
    ):
        pipeline.ingest(
            settings.fixture_dir / filename,
            groups,
            strategy=settings.ingestion_strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            parent_chunk_size=settings.parent_chunk_size,
        )
    ingest_seconds = time.perf_counter() - started
    retriever = HybridRetriever(
        embedding,
        store,
        bm25,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        reranker=None,
    )
    llm: LLMClient = (
        OpenAILLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
        if args.real_llm
        else StableGroundedLLM()
    )
    tracer = InMemoryTracer()
    feedback_store = FeedbackStore(Path("data/profile.sqlite") if args.real_llm else ":memory:")
    service = RAGService(
        retriever=retriever,
        llm=llm,
        feedback_store=feedback_store,
        tracer=tracer,
        prompt_cost_per_1k=settings.prompt_cost_per_1k,
        completion_cost_per_1k=settings.completion_cost_per_1k,
    )
    cases = load_golden_set(args.golden_set)
    run_started = time.perf_counter()
    for number in range(args.requests):
        case = cases[number % len(cases)]
        service.ask(
            case.query,
            User(f"profile-{number}", frozenset(case.user_groups)),
        )
    run_seconds = time.perf_counter() - run_started
    result = {
        "llm_mode": "real" if args.real_llm else "stable_fixture",
        "requests": args.requests,
        "ingest_seconds": ingest_seconds,
        "run_seconds": run_seconds,
        "latency_percentiles_ms": tracer.latency_percentiles(),
        "usage_summary": asdict(feedback_store.usage_summary()),
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
