import tempfile
from pathlib import Path

from rag_permission.citations import Citation, CitationChecker
from rag_permission.feedback import FeedbackStore
from rag_permission.generation import generate_answer
from rag_permission.llm import LLMResponse, LLMUsage
from rag_permission.models import DocumentChunk, SearchHit, User


class FakeLLM:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse("风扇故障 [1]。", LLMUsage(prompt_tokens=5, completion_tokens=3))


def _hit(groups=("public",)):
    chunk = DocumentChunk(
        chunk_id="doc:section:1",
        doc_id="doc",
        text="E-1002 是风扇故障。",
        chunk_type="section",
        section_path=("故障码",),
        source="sample.md",
        title="手册",
        acl_groups=groups,
        locator="table:1",
    )
    return SearchHit(chunk=chunk, score=1.0)


def test_citation_checker_accepts_valid_citation():
    citation = Citation(1, "doc:section:1", "doc", "手册", "sample.md", ("故障码",))
    report = CitationChecker().check("风扇故障 [1]。", [citation])
    assert not report.suspected_hallucination


def test_citation_checker_flags_fabricated_and_zero_citations():
    citation = Citation(1, "doc:section:1", "doc", "手册", "sample.md", ("故障码",))
    assert CitationChecker().check("风扇故障 [9]。", [citation]).invalid_citations == (9,)
    assert CitationChecker().check("风扇故障。", [citation]).suspected_hallucination


def test_citation_checker_does_not_flag_refusal():
    assert not CitationChecker().check("未找到权限范围内相关资料。", []).suspected_hallucination


def test_generate_answer_degrades_without_hits():
    result = generate_answer("问题", [], FakeLLM())
    assert result.citation_report.is_refusal
    assert result.citations == []


def test_generate_answer_checks_citations():
    result = generate_answer("E-1002 是什么？", [_hit()], FakeLLM())
    assert result.answer.endswith("[1]。")
    assert result.citations[0].number == 1


def test_generate_answer_returns_only_cited_reference():
    hits = [_hit(), _hit()]
    result = generate_answer("E-1002 是什么？", hits, FakeLLM())
    assert [citation.number for citation in result.citations] == [1]


def test_generate_answer_refusal_has_no_citations():
    class RefusingLLM(FakeLLM):
        def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
            return LLMResponse("未找到权限范围内相关资料。", LLMUsage(prompt_tokens=5, completion_tokens=3))

    result = generate_answer("无关问题", [_hit(), _hit()], RefusingLLM())
    assert result.citation_report.is_refusal
    assert result.citations == []


def test_feedback_store_records_trace_and_clusters_failures():
    with tempfile.TemporaryDirectory() as directory:
        store = FeedbackStore(Path(directory) / "feedback.db")
        user = User(id="u", groups=frozenset({"public"}))
        store.record_trace("t1", user, "q", "a", [_hit()], [], LLMUsage())
        assert store.save_feedback("t1", "down")
        assert not store.save_feedback("missing", "down")
        report = store.weekly_failure_report()
        assert report[0].doc_id == "doc"
