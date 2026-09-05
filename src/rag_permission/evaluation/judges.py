import json
import re

from rag_permission.citations import CitationReport, is_refusal
from rag_permission.llm import LLMClient
from rag_permission.models import User


FAITHFULNESS_SYSTEM = """You are an adversarial faithfulness judge.
Split the answer into atomic claims. For each claim, decide whether it is fully
supported by the numbered sources. Do not use outside knowledge. Respond with JSON:
{"assertions":[{"text":"...","supported":true}]}
"""

RELEVANCE_SYSTEM = """You are an answer relevance judge.
Compare the answer with the question. Respond with exactly one label:
relevant, partially_relevant, or irrelevant.
"""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM judge did not return JSON")
    return json.loads(match.group(0))


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def faithfulness(
        self,
        query: str,
        answer: str,
        passages: list[str],
        report: CitationReport | None = None,
    ) -> float:
        if report is not None and report.is_refusal and not report.has_citations:
            return 1.0
        source_text = "\n\n".join(
            f"[{number}] {passage}" for number, passage in enumerate(passages, 1)
        )
        response = self.llm.complete(
            FAITHFULNESS_SYSTEM,
            f"Question: {query}\nAnswer: {answer}\nSources:\n{source_text}",
            temperature=0.0,
        )
        assertions = extract_json(response.text).get("assertions", [])
        if not assertions:
            return 0.0
        return sum(bool(item.get("supported")) for item in assertions) / len(assertions)

    def answer_relevance(self, query: str, answer: str) -> float:
        response = self.llm.complete(
            RELEVANCE_SYSTEM, f"Question: {query}\nAnswer: {answer}", temperature=0.0
        )
        label = response.text.strip().lower()
        return {"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0}.get(label, 0.0)


class NaiveLooseJudge:
    def faithfulness(
        self,
        query: str,
        answer: str,
        passages: list[str],
        report: CitationReport | None = None,
    ) -> float:
        if not answer.strip():
            return 0.0
        if report is not None and report.is_refusal and not report.has_citations:
            return 1.0
        return 1.0 if passages and any(word in answer for word in "".join(passages).split()) else 0.5

    def answer_relevance(self, query: str, answer: str) -> float:
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & answer_tokens) / len(query_tokens)
        return 1.0 if overlap >= 0.5 else 0.5 if overlap > 0 else 0.0
