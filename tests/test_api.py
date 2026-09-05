from fastapi.testclient import TestClient

from rag_permission.auth import create_access_token
from rag_permission.config import Settings
from rag_permission.api import create_app
from rag_permission.citations import Citation, CitationReport
from rag_permission.generation import AnswerResult
from rag_permission.llm import LLMUsage
from rag_permission.models import SearchHit, User


class FakeRAG:
    def __init__(self):
        self.feedback_store = _FakeStore()

    def ask(self, query: str, user: User) -> AnswerResult:
        return AnswerResult(
            "答案 [1]。",
            [Citation(1, "c1", "d1", "title", "source", ("section",))],
            CitationReport(True, (1,), (), False, False, ()),
            LLMUsage(2, 3),
            "trace-1",
        )

    def ask_stream(self, query: str, user: User):
        yield _Token("答案 ")
        yield _Token("[1]。")
        yield _Complete()


class _Token:
    def __init__(self, text: str):
        self.text = text


class _Complete:
    answer = "答案 [1]。"
    citations = [Citation(1, "c1", "d1", "title", "source", ("section",))]
    citation_report = CitationReport(True, (1,), (), False, False, ())
    usage = LLMUsage(2, 3)
    trace_id = "trace-1"


class _FakeStore:
    def save_feedback(self, trace_id: str, rating: str, comment: str = "") -> bool:
        return trace_id == "trace-1"


def _settings(auth_mode: str = "jwt") -> Settings:
    secret = "test-secret-that-is-at-least-32-bytes"
    return Settings(auth_mode=auth_mode, auth_secret=secret)


def test_api_requires_authentication():
    app = create_app(settings=_settings(), runtime=FakeRAG())
    with TestClient(app) as client:
        assert client.post("/ask", json={"query": "q"}).status_code == 401


def test_api_ask_uses_verified_jwt_groups():
    app = create_app(settings=_settings(), runtime=FakeRAG())
    token = create_access_token("u", ("public",), "test-secret-that-is-at-least-32-bytes")
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"query": "q"},
            headers={"Authorization": f"Bearer {token}", "X-User-Groups": "admin"},
        )
        assert response.status_code == 200
        assert response.json()["citations"][0]["number"] == 1


def test_api_rejects_invalid_jwt():
    app = create_app(settings=_settings(), runtime=FakeRAG())
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"query": "q"},
            headers={"Authorization": "Bearer invalid", "X-User-Groups": "admin"},
        )
        assert response.status_code == 401


def test_api_trusted_header_mode_requires_gateway_user():
    app = create_app(settings=_settings("trusted_header"), runtime=FakeRAG())
    with TestClient(app) as client:
        denied = client.post("/ask", json={"query": "q"}, headers={"X-User-Groups": "public"})
        assert denied.status_code == 401
        allowed = client.post(
            "/ask",
            json={"query": "q"},
            headers={"X-User-Id": "u"},
        )
        assert allowed.status_code == 200


def test_api_stream_and_feedback():
    app = create_app(settings=_settings(), runtime=FakeRAG())
    token = create_access_token("u", ("public",), "test-secret-that-is-at-least-32-bytes")
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        response = client.post(
            "/ask-stream", json={"query": "q"}, headers=headers
        )
        assert response.status_code == 200
        assert "event: token" in response.text
        assert "event: citations" in response.text
        feedback = client.post(
            "/feedback",
            json={"trace_id": "trace-1", "rating": "up"},
            headers=headers,
        )
        assert feedback.status_code == 200
