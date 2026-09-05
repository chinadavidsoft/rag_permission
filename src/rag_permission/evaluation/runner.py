import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_permission.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_permission.models import SearchHit, User


class Retriever(Protocol):
    def search(self, query: str, user: User) -> list[SearchHit]: ...


@dataclass(frozen=True, slots=True)
class GoldenCase:
    query: str
    user_groups: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...] = ()
    forbidden_chunk_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenCase":
        return cls(
            query=data["query"],
            user_groups=tuple(data["user_groups"]),
            relevant_chunk_ids=tuple(data.get("relevant_chunk_ids", [])),
            forbidden_chunk_ids=tuple(data.get("forbidden_chunk_ids", [])),
        )


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    query: str
    user_groups: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    expected_count: int
    recall: float
    precision: float
    mrr: float
    leakage: bool


@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    case_count: int
    nonempty_hit_cases: int
    recall: float
    precision: float
    mrr: float
    leakage_cases: int


def summarize_evaluations(results: list[RetrievalEvaluation]) -> RetrievalSummary:
    if not results:
        return RetrievalSummary(0, 0, 0.0, 0.0, 0.0, 0)
    scored = [result for result in results if result.expected_count > 0]
    return RetrievalSummary(
        case_count=len(results),
        nonempty_hit_cases=sum(bool(result.retrieved_chunk_ids) for result in results),
        recall=sum(result.recall for result in scored) / len(scored),
        precision=sum(result.precision for result in scored) / len(scored),
        mrr=sum(result.mrr for result in scored) / len(scored),
        leakage_cases=sum(result.leakage for result in results),
    )


def evaluate_case(case: GoldenCase, hits: list[SearchHit], k: int = 8) -> RetrievalEvaluation:
    retrieved = [hit.chunk.chunk_id for hit in hits[:k]]
    relevant = list(case.relevant_chunk_ids)
    return RetrievalEvaluation(
        query=case.query,
        user_groups=case.user_groups,
        retrieved_chunk_ids=tuple(retrieved),
        expected_count=len(case.relevant_chunk_ids),
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant, k),
        leakage=any(chunk_id in case.forbidden_chunk_ids for chunk_id in retrieved),
    )


class EvaluationRunner:
    def __init__(self, retriever: Retriever, k: int = 8):
        self.retriever = retriever
        self.k = k

    def run(self, cases: list[GoldenCase]) -> list[RetrievalEvaluation]:
        results = []
        for case in cases:
            user = User(id="evaluation", groups=frozenset(case.user_groups))
            results.append(evaluate_case(case, self.retriever.search(case.query, user), self.k))
        return results


def load_golden_set(path: Path | str) -> list[GoldenCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase.from_dict(item) for item in data["cases"]]


def evaluation_key(result: RetrievalEvaluation) -> tuple[str, tuple[str, ...]]:
    return result.query, result.user_groups
