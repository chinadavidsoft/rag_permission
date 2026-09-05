import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace

from rag_permission.citations import (
    Citation,
    CitationChecker,
    CitationReport,
    citations_from_hits,
    is_refusal,
)
from rag_permission.llm import LLMClient, LLMUsage, TokenDelta, UsageDelta
from rag_permission.models import SearchHit
from rag_permission.observability import NullTracer, Tracer


SYSTEM_PROMPT = """你是一个企业知识库助手。只使用用户消息中编号提供的资料回答问题。
每个实质性事实都必须紧跟对应资料编号，例如 [1] 或 [2]。不要使用外部知识。
如果资料不足或没有相关资料，只回答：“未找到权限范围内相关资料。”
"""

REFUSAL_ANSWER = "未找到权限范围内相关资料。"


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    citation_report: CitationReport
    usage: LLMUsage
    trace_id: str


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class UsageEvent:
    usage: LLMUsage


@dataclass(frozen=True, slots=True)
class AnswerCompleteEvent:
    answer: str
    citations: list[Citation]
    citation_report: CitationReport
    usage: LLMUsage
    trace_id: str


StreamEvent = TokenEvent | UsageEvent | AnswerCompleteEvent


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    passages = []
    for number, hit in enumerate(hits, 1):
        text = hit.chunk.parent_text or hit.chunk.text
        section = " > ".join(hit.chunk.section_path) or "未分类"
        passages.append(
            f"[{number}] 标题：{hit.chunk.title}\n章节：{section}\n内容：{text}"
        )
    return f"资料：\n" + "\n\n".join(passages) + f"\n\n问题：{query}"


def _empty_refusal(trace_id: str) -> AnswerResult:
    answer = "未找到权限范围内相关资料。"
    report = CitationChecker().check(answer, [])
    return AnswerResult(answer, [], report, LLMUsage(), trace_id)


def display_citations(answer: str, citations: list[Citation]) -> tuple[list[Citation], CitationReport]:
    if is_refusal(answer):
        return [], CitationChecker().check(answer, [])
    report = CitationChecker().check(answer, citations)
    if report.suspected_hallucination:
        refusal = "未找到权限范围内相关资料。"
        refusal_report = CitationChecker().check(refusal, [])
        refusal_report = replace(
            refusal_report,
            has_citations=report.has_citations,
            valid_citations=report.valid_citations,
            invalid_citations=report.invalid_citations,
            suspected_hallucination=True,
            issues=report.issues + ("refusal_enforced",),
        )
        return [], refusal_report
    valid_numbers = set(report.valid_citations)
    shown = [citation for citation in citations if citation.number in valid_numbers]
    return shown, CitationChecker().check(answer, shown)


def generate_answer(
    query: str,
    hits: list[SearchHit],
    llm: LLMClient,
    trace_id: str | None = None,
    tracer: Tracer | None = None,
    temperature: float = 0.1,
) -> AnswerResult:
    trace_id = trace_id or uuid.uuid4().hex
    tracer = tracer or NullTracer()
    if not hits:
        return _empty_refusal(trace_id)
    prompt = build_prompt(query, hits)
    with tracer.span("llm_generate", trace_id, {"stream": False}):
        response = llm.complete(SYSTEM_PROMPT, prompt, temperature)
    citations = citations_from_hits(hits)
    with tracer.span("check_citations", trace_id):
        shown, report = display_citations(response.text, citations)
    answer = REFUSAL_ANSWER if "refusal_enforced" in report.issues else response.text
    return AnswerResult(answer, shown, report, response.usage, trace_id)


def stream_answer(
    query: str,
    hits: list[SearchHit],
    llm: LLMClient,
    trace_id: str | None = None,
    tracer: Tracer | None = None,
    temperature: float = 0.1,
) -> Iterator[StreamEvent]:
    trace_id = trace_id or uuid.uuid4().hex
    tracer = tracer or NullTracer()
    if not hits:
        yield TokenEvent(REFUSAL_ANSWER)
        yield AnswerCompleteEvent(
            REFUSAL_ANSWER,
            [],
            CitationChecker().check(REFUSAL_ANSWER, []),
            LLMUsage(),
            trace_id,
        )
        return

    prompt = build_prompt(query, hits)
    answer = ""
    usage = LLMUsage()
    with tracer.span("llm_generate", trace_id, {"stream": True}):
        for delta in llm.stream(SYSTEM_PROMPT, prompt, temperature):
            if isinstance(delta, TokenDelta):
                answer += delta.text
                yield TokenEvent(delta.text)
            elif isinstance(delta, UsageDelta):
                usage = delta.usage
                yield UsageEvent(delta.usage)
    citations = citations_from_hits(hits)
    with tracer.span("check_citations", trace_id):
        shown, report = display_citations(answer, citations)
    final_answer = answer
    enforced_refusal = "refusal_enforced" in report.issues
    if enforced_refusal:
        final_answer = REFUSAL_ANSWER
    yield AnswerCompleteEvent(final_answer, shown, report, usage, trace_id)
