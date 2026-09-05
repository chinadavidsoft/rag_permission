import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_permission.auth import decode_access_token
from rag_permission.config import Settings
from rag_permission.models import User
from rag_permission.observability import token_cost
from rag_permission.runtime import RAGService, build_runtime


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str
    comment: str = ""


def create_app(settings: Settings | None = None, runtime: RAGService | None = None) -> FastAPI:
    settings = settings or Settings()

    def current_user(
        authorization: str | None = Header(default=None),
        x_user_id: str | None = Header(default=None),
        x_user_groups: str | None = Header(default=None),
    ) -> User:
        if settings.auth_mode == "jwt":
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Bearer token required")
            try:
                return decode_access_token(authorization.removeprefix("Bearer "), settings.auth_secret)
            except Exception as error:
                raise HTTPException(status_code=401, detail="Invalid access token") from error

        if not x_user_id:
            raise HTTPException(status_code=401, detail="X-User-Id required")
        groups = frozenset(
            group.strip() for group in (x_user_groups or "").split(",") if group.strip()
        )
        return User(id=x_user_id, groups=groups)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.rag_service is None:
            app.state.rag_service = build_runtime(settings)
        yield

    app = FastAPI(title="Permission-aware Enterprise KB RAG", lifespan=lifespan)
    app.state.rag_service = runtime

    @app.get("/")
    def index() -> HTMLResponse:
        page = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(payload: AskRequest, user: User = Depends(current_user)) -> dict:
        result = request_rag_service(app).ask(payload.query, user)
        return {
            "answer": result.answer,
            "citations": [asdict(citation) for citation in result.citations],
            "citation_report": asdict(result.citation_report),
            "usage": asdict(result.usage),
            "cost": {
                "prompt": token_cost(result.usage.prompt_tokens, 0, settings.prompt_cost_per_1k),
                "completion": token_cost(
                    0, result.usage.completion_tokens, 0, settings.completion_cost_per_1k
                ),
                "total": token_cost(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    settings.prompt_cost_per_1k,
                    settings.completion_cost_per_1k,
                ),
            },
            "trace_id": result.trace_id,
        }

    @app.post("/ask-stream")
    def ask_stream(payload: AskRequest, user: User = Depends(current_user)) -> StreamingResponse:
        def event_stream() -> Iterator[str]:
            answer = ""
            for event in request_rag_service(app).ask_stream(payload.query, user):
                if hasattr(event, "text"):
                    answer += event.text
                    payload_data = {"content": event.text}
                    event_name = "token"
                elif hasattr(event, "citations"):
                    payload_data = {
                        "answer": event.answer,
                        "citations": [asdict(citation) for citation in event.citations],
                        "citation_report": asdict(event.citation_report),
                        "usage": asdict(event.usage),
                        "trace_id": event.trace_id,
                    }
                    event_name = "citations"
                elif hasattr(event, "usage"):
                    payload_data = {"prompt_tokens": event.usage.prompt_tokens, "completion_tokens": event.usage.completion_tokens}
                    event_name = "usage"
                else:
                    continue
                yield f"event: {event_name}\ndata: {json.dumps(payload_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/feedback")
    def feedback(payload: FeedbackRequest, user: User = Depends(current_user)) -> dict:
        if payload.rating not in {"up", "down"}:
            raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
        saved = request_rag_service(app).feedback_store.save_feedback(
            payload.trace_id, payload.rating, payload.comment
        )
        if not saved:
            raise HTTPException(status_code=404, detail="trace not found")
        return {"status": "saved", "trace_id": payload.trace_id}

    return app


def request_rag_service(app: FastAPI) -> RAGService:
    service = app.state.rag_service
    if service is None:
        raise RuntimeError("RAG service is not initialized")
    return service


app = create_app()
