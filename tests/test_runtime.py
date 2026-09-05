import math
import tempfile
from pathlib import Path

from rag_permission.feedback import FeedbackStore
from rag_permission.llm import LLMResponse, LLMUsage
from rag_permission.models import DocumentChunk, SearchHit, User
from rag_permission.runtime import RAGService


class FixedRetriever:
    def search(self, query: str, user: User) -> list[SearchHit]:
        chunk = DocumentChunk(
            chunk_id="doc:section:1",
            doc_id="doc",
            text="E-1002 是风扇故障。",
            chunk_type="section",
            section_path=("故障码",),
            source="sample.md",
            title="手册",
            acl_groups=("public",),
            locator="table:1",
        )
        return [SearchHit(chunk=chunk, score=1.0)]


class CitedLLM:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse("风扇故障 [1]。", LLMUsage(prompt_tokens=5, completion_tokens=3))


def test_rag_service_records_usage_costs():
    with tempfile.TemporaryDirectory() as directory:
        store = FeedbackStore(Path(directory) / "feedback.db")
        service = RAGService(
            FixedRetriever(),
            CitedLLM(),
            store,
            prompt_cost_per_1k=0.1,
            completion_cost_per_1k=0.2,
        )
        result = service.ask("E-1002 是什么？", User("u", frozenset({"public"})))
        summary = store.usage_summary()
        assert result.answer == "风扇故障 [1]。"
        assert summary.prompt_tokens == 5
        assert summary.completion_tokens == 3
        assert math.isclose(summary.total_cost, 0.0011)
