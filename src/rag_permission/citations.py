import re
from dataclasses import dataclass

from rag_permission.models import SearchHit


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
REFUSAL_PATTERNS = ("未找到", "没有足够", "无法回答", "资料不足", "无法确定")


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationReport:
    has_citations: bool
    valid_citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    is_refusal: bool
    suspected_hallucination: bool
    issues: tuple[str, ...]


def citations_from_hits(hits: list[SearchHit]) -> list[Citation]:
    return [
        Citation(
            number=number,
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            title=hit.chunk.title,
            source=hit.chunk.source,
            section_path=hit.chunk.section_path,
        )
        for number, hit in enumerate(hits, 1)
    ]


def is_refusal(answer: str) -> bool:
    return any(pattern in answer for pattern in REFUSAL_PATTERNS)


class CitationChecker:
    def check(self, answer: str, citations: list[Citation]) -> CitationReport:
        cited_numbers = [int(value) for value in CITATION_PATTERN.findall(answer)]
        valid = tuple(sorted({number for number in cited_numbers if 1 <= number <= len(citations)}))
        invalid = tuple(sorted({number for number in cited_numbers if number < 1 or number > len(citations)}))
        refusal = is_refusal(answer)
        issues = []
        if invalid:
            issues.append(f"fabricated_citations:{invalid}")
        if not cited_numbers and not refusal:
            issues.append("substantive_answer_without_citations")
        suspected = bool(invalid) or (not cited_numbers and not refusal)
        return CitationReport(
            has_citations=bool(cited_numbers),
            valid_citations=valid,
            invalid_citations=invalid,
            is_refusal=refusal,
            suspected_hallucination=suspected,
            issues=tuple(issues),
        )
