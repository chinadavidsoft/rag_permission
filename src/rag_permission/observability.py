import math
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Span:
    name: str
    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @property
    def duration_ms(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000


class Tracer(Protocol):
    @contextmanager
    def span(self, name: str, trace_id: str, attributes: dict[str, Any] | None = None):
        yield  # pragma: no cover


class NullTracer:
    @contextmanager
    def span(self, name: str, trace_id: str, attributes: dict[str, Any] | None = None):
        yield Span(name=name, trace_id=trace_id, span_id="")


class InMemoryTracer:
    def __init__(self):
        self.spans: list[Span] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, trace_id: str, attributes: dict[str, Any] | None = None):
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex,
            attributes=dict(attributes or {}),
            started_at=time.perf_counter(),
        )
        try:
            yield span
        finally:
            span.ended_at = time.perf_counter()
            with self._lock:
                self.spans.append(span)

    def latency_percentiles(self, p50: float = 50, p95: float = 95) -> dict[str, dict[str, float]]:
        durations: dict[str, list[float]] = {}
        for span in self.spans:
            if span.duration_ms is not None:
                durations.setdefault(span.name, []).append(span.duration_ms)
        return {
            name: {
                "p50_ms": nearest_rank_percentile(values, p50),
                "p95_ms": nearest_rank_percentile(values, p95),
            }
            for name, values in sorted(durations.items())
        }


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float = 0.0,
    completion_cost_per_1k: float = 0.0,
) -> float:
    return prompt_tokens / 1000 * prompt_cost_per_1k + completion_tokens / 1000 * completion_cost_per_1k
