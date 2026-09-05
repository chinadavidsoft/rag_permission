from collections.abc import Iterator

from rag_permission.citations import citations_from_hits
from rag_permission.generation import AnswerCompleteEvent, TokenEvent, stream_answer
from rag_permission.llm import LLMResponse, LLMUsage, TokenDelta, UsageDelta
from rag_permission.models import DocumentChunk, SearchHit
from rag_permission.observability import InMemoryTracer


class ScriptedLLM:
    def __init__(self, text: str):
        self.text = text

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(self.text, LLMUsage(prompt_tokens=11, completion_tokens=4))

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[UsageDelta | TokenEvent]:
        yield UsageDelta(LLMUsage(prompt_tokens=11, completion_tokens=4))
        for token in self.text:
            yield TokenDelta(token)


def make_hit(text: str) -> SearchHit:
    chunk = DocumentChunk(
        chunk_id=f"doc:{text}",
        doc_id="doc",
        text=text,
        chunk_type="section",
        section_path=("故障码",),
        source="fixtures/sample.md",
        title="设备手册",
        acl_groups=("public",),
        locator="table:2",
    )
    return SearchHit(chunk=chunk, score=1.0)


def main() -> None:
    hit = make_hit("E-1002 是风扇故障。")
    citations = citations_from_hits([hit])
    tracer = InMemoryTracer()
    events = list(stream_answer("E-1002 是什么故障？", [hit], ScriptedLLM("E-1002 是风扇故障 [1]。"), tracer=tracer))
    tokens = "".join(event.text for event in events if isinstance(event, TokenEvent))
    final = next(event for event in events if isinstance(event, AnswerCompleteEvent))
    print(f"streamed={tokens}, citations={final.citation_report.valid_citations}, usage={final.usage.total_tokens}")

    bad = list(stream_answer("E-1002 是什么故障？", [hit], ScriptedLLM("这是风扇问题 [9]。")))
    final_bad = next(event for event in bad if isinstance(event, AnswerCompleteEvent))
    print(f"fabricated={final_bad.citation_report.invalid_citations}, suspected={final_bad.citation_report.suspected_hallucination}")

    denied = list(stream_answer("E-1002 是什么故障？", [], ScriptedLLM("不该被调用")))
    final_denied = next(event for event in denied if isinstance(event, AnswerCompleteEvent))
    print(f"denied={final_denied.citation_report.is_refusal}, suspected={final_denied.citation_report.suspected_hallucination}")
    print(f"latency={tracer.latency_percentiles()}")


if __name__ == "__main__":
    main()
