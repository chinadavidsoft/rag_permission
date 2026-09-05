from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class User:
    id: str
    groups: frozenset[str] = frozenset()

    def can_access(self, acl_groups: tuple[str, ...] | frozenset[str] | list[str]) -> bool:
        return bool(self.groups & frozenset(acl_groups))


@dataclass(frozen=True, slots=True)
class ParsedElement:
    kind: Literal["heading", "paragraph", "table_row"]
    text: str
    section_path: tuple[str, ...]
    locator: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    doc_id: str
    title: str
    source: str
    acl_groups: tuple[str, ...]
    elements: tuple[ParsedElement, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: Literal["fixed", "recursive", "section", "child"]
    section_path: tuple[str, ...]
    source: str
    title: str
    acl_groups: tuple[str, ...]
    locator: str
    parent_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "section_path": list(self.section_path),
            "source": self.source,
            "title": self.title,
            "acl_groups": list(self.acl_groups),
            "locator": self.locator,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: DocumentChunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
