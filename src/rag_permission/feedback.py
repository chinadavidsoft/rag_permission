import json
import sqlite3
import threading
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rag_permission.citations import Citation
from rag_permission.llm import LLMUsage
from rag_permission.models import SearchHit, User


@dataclass(frozen=True, slots=True)
class FailureCluster:
    week_start: str
    doc_id: str
    title: str
    count: int
    queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UsageSummary:
    trace_count: int
    prompt_tokens: int
    completion_tokens: int
    prompt_cost: float
    completion_cost: float
    total_cost: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connection as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS qa_traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    retrieved_ids TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    trace_id TEXT PRIMARY KEY,
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES qa_traces(trace_id)
                );
                """
            )
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(qa_traces)")}
        if "prompt_cost" not in columns:
            self._connection.execute(
                "ALTER TABLE qa_traces ADD COLUMN prompt_cost REAL NOT NULL DEFAULT 0"
            )
        if "completion_cost" not in columns:
            self._connection.execute(
                "ALTER TABLE qa_traces ADD COLUMN completion_cost REAL NOT NULL DEFAULT 0"
            )
        if "total_cost" not in columns:
            self._connection.execute(
                "ALTER TABLE qa_traces ADD COLUMN total_cost REAL NOT NULL DEFAULT 0"
            )

    def record_trace(
        self,
        trace_id: str,
        user: User,
        query: str,
        answer: str,
        hits: list[SearchHit],
        citations: list[Citation],
        usage: LLMUsage,
        prompt_cost: float = 0.0,
        completion_cost: float = 0.0,
    ) -> None:
        total_cost = prompt_cost + completion_cost
        with self._lock, self._connection as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO qa_traces
                (trace_id, created_at, user_id, query, answer, retrieved_ids,
                 citations, prompt_tokens, completion_tokens, prompt_cost,
                 completion_cost, total_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    utc_now(),
                    user.id,
                    query,
                    answer,
                    json.dumps([hit.chunk.chunk_id for hit in hits], ensure_ascii=False),
                    json.dumps([asdict(citation) for citation in citations], ensure_ascii=False),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    prompt_cost,
                    completion_cost,
                    total_cost,
                ),
            )

    def save_feedback(self, trace_id: str, rating: str, comment: str = "") -> bool:
        if rating not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        with self._lock, self._connection as connection:
            exists = connection.execute(
                "SELECT 1 FROM qa_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                """
                INSERT OR REPLACE INTO feedback (trace_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (trace_id, rating, comment, utc_now()),
            )
        return True

    def weekly_failure_report(
        self, reference_date: date | None = None
    ) -> list[FailureCluster]:
        reference = reference_date or date.today()
        week_start = reference - timedelta(days=reference.weekday())
        with self._lock, self._connection as connection:
            rows = connection.execute(
                """
                SELECT qa_traces.query, qa_traces.answer, qa_traces.retrieved_ids, qa_traces.citations
                FROM feedback
                JOIN qa_traces ON qa_traces.trace_id = feedback.trace_id
                WHERE feedback.rating = 'down' AND feedback.created_at >= ?
                ORDER BY feedback.created_at
                """,
                (week_start.isoformat(),),
            ).fetchall()

        clusters: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            retrieved_ids = json.loads(row["retrieved_ids"])
            citations = json.loads(row["citations"])
            title = citations[0]["title"] if citations else "no-citation"
            doc_id = (
                retrieved_ids[0].split(":", 1)[0]
                if retrieved_ids
                else "no-retrieved-document"
            )
            clusters[(week_start.isoformat(), doc_id, title)].append(row["query"])
        return [
            FailureCluster(
                week_start=week_start.isoformat(),
                doc_id=doc_id,
                title=title,
                count=len(queries),
                queries=tuple(queries),
            )
            for (week_start_value, doc_id, title), queries in sorted(clusters.items())
        ]

    def usage_summary(self) -> UsageSummary:
        with self._lock, self._connection as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS trace_count,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(prompt_cost), 0) AS prompt_cost,
                       COALESCE(SUM(completion_cost), 0) AS completion_cost,
                       COALESCE(SUM(total_cost), 0) AS total_cost
                FROM qa_traces
                """
            ).fetchone()
        return UsageSummary(
            trace_count=row["trace_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            prompt_cost=row["prompt_cost"],
            completion_cost=row["completion_cost"],
            total_cost=row["total_cost"],
        )
