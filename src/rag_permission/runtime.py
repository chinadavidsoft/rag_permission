import uuid
from collections.abc import Iterator
import os
from pathlib import Path

from rag_permission.bm25 import BM25Index
from rag_permission.config import Settings
from rag_permission.embeddings import BGEEmbedding, EmbeddingBackend
from rag_permission.feedback import FeedbackStore
from rag_permission.generation import (
    AnswerResult,
    StreamEvent,
    generate_answer,
    stream_answer,
)
from rag_permission.ingest_pipeline import IngestionPipeline
from rag_permission.llm import LLMClient, LLMUsage, OpenAILLMClient
from rag_permission.models import SearchHit, User
from rag_permission.observability import InMemoryTracer, NullTracer, Tracer, token_cost
from rag_permission.reranker import BGEReranker, Reranker
from rag_permission.retriever import HybridRetriever
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


class RAGService:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        feedback_store: FeedbackStore,
        tracer: Tracer | None = None,
        prompt_cost_per_1k: float = 0.0,
        completion_cost_per_1k: float = 0.0,
    ):
        self.retriever = retriever
        self.llm = llm
        self.feedback_store = feedback_store
        self.tracer = tracer or NullTracer()
        self.prompt_cost_per_1k = prompt_cost_per_1k
        self.completion_cost_per_1k = completion_cost_per_1k

    def _costs(self, usage: LLMUsage) -> tuple[float, float]:
        return (
            token_cost(usage.prompt_tokens, 0, self.prompt_cost_per_1k),
            token_cost(0, usage.completion_tokens, 0.0, self.completion_cost_per_1k),
        )

    def _retrieve(self, query: str, user: User, trace_id: str) -> list[SearchHit]:
        with self.tracer.span("retrieve", trace_id, {"user_id": user.id}):
            return self.retriever.search(query, user)

    def ask(self, query: str, user: User) -> AnswerResult:
        trace_id = uuid.uuid4().hex
        hits = self._retrieve(query, user, trace_id)
        with self.tracer.span("generate", trace_id, {"hit_count": len(hits)}):
            result = generate_answer(query, hits, self.llm, trace_id=trace_id, tracer=self.tracer)
        prompt_cost, completion_cost = self._costs(result.usage)
        self.feedback_store.record_trace(
            trace_id=trace_id,
            user=user,
            query=query,
            answer=result.answer,
            hits=hits,
            citations=result.citations,
            usage=result.usage,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
        )
        return result

    def ask_stream(self, query: str, user: User) -> Iterator[StreamEvent]:
        trace_id = uuid.uuid4().hex
        hits = self._retrieve(query, user, trace_id)
        answer = ""
        final_usage = None
        final_result = None
        for event in stream_answer(query, hits, self.llm, trace_id=trace_id, tracer=self.tracer):
            if hasattr(event, "text"):
                answer += event.text
            if hasattr(event, "citations"):
                final_result = event
            yield event
        if final_result is not None:
            final_usage = final_result.usage
            prompt_cost, completion_cost = self._costs(final_usage)
            answer = final_result.answer
            self.feedback_store.record_trace(
                trace_id=trace_id,
                user=user,
                query=query,
                answer=answer,
                hits=hits,
                citations=final_result.citations,
                usage=final_usage,
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
            )


def build_runtime(settings: Settings) -> RAGService:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_XET", str(settings.hf_hub_disable_xet).lower())
    embedding: EmbeddingBackend = BGEEmbedding(settings.embedding_model)
    reranker: Reranker | None = BGEReranker(settings.reranker_model) if settings.enable_rerank else None
    client = create_qdrant_client(settings.qdrant_url)
    # bge-m3 outputs 1024-dimensional dense vectors.
    vector_store = QdrantVectorStore(client, settings.collection_name, 1024)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(
        embedding,
        vector_store,
        bm25,
    )
    for filename, groups in (
        ("sample.md", ("public",)),
        ("sample.docx", ("eng",)),
        ("sample.pdf", ("hr",)),
    ):
        path = settings.fixture_dir / filename
        if path.exists():
            pipeline.ingest(
                path,
                groups,
                strategy=settings.ingestion_strategy,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                parent_chunk_size=settings.parent_chunk_size,
            )
    llm: LLMClient = OpenAILLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    retriever = HybridRetriever(
        embedding,
        vector_store,
        bm25,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        reranker=reranker,
    )
    return RAGService(
        retriever=retriever,
        llm=llm,
        feedback_store=FeedbackStore(settings.sqlite_path),
        tracer=InMemoryTracer(),
        prompt_cost_per_1k=settings.prompt_cost_per_1k,
        completion_cost_per_1k=settings.completion_cost_per_1k,
    )
