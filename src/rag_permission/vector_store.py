import uuid
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_permission.models import DocumentChunk, SearchHit


def _chunk_from_payload(payload: dict) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        chunk_type=payload["chunk_type"],
        section_path=tuple(payload.get("section_path", [])),
        source=payload.get("source", ""),
        title=payload.get("title", ""),
        acl_groups=tuple(payload.get("acl_groups", [])),
        locator=payload.get("locator", ""),
        parent_id=payload.get("parent_id"),
        parent_text=payload.get("parent_text"),
        metadata=payload.get("metadata", {}),
    )


class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], limit: int, acl_groups: frozenset[str]
    ) -> list[tuple[DocumentChunk, float]]: ...


def create_qdrant_client(qdrant_url: str) -> QdrantClient:
    if not qdrant_url or qdrant_url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=qdrant_url)


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-permission:{chunk_id}"))

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            qmodels.PointStruct(
                id=self.point_id(chunk.chunk_id), vector=vector, payload=chunk.payload
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _acl_filter(self, acl_groups: frozenset[str]) -> qmodels.Filter | None:
        if not acl_groups:
            return qmodels.Filter(must=[])
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="acl_groups", match=qmodels.MatchAny(any=sorted(acl_groups))
                )
            ]
        )

    def search(
        self, vector: list[float], limit: int, acl_groups: frozenset[str]
    ) -> list[tuple[SearchHit, float]]:
        if not acl_groups or limit <= 0:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=self._acl_filter(acl_groups),
            with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                (
                    SearchHit(chunk=_chunk_from_payload(payload), score=float(point.score or 0.0)),
                    float(point.score or 0.0),
                )
            )
        return hits
