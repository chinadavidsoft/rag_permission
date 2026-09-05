import tempfile
from pathlib import Path

from rag_permission.citations import citations_from_hits
from rag_permission.feedback import FeedbackStore
from rag_permission.llm import LLMUsage
from rag_permission.models import DocumentChunk, SearchHit, User


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = FeedbackStore(Path(directory) / "feedback.db")
        chunk = DocumentChunk(
            chunk_id="doc-a:section:1",
            doc_id="doc-a",
            text="设备维护说明",
            chunk_type="section",
            section_path=("维护",),
            source="sample.md",
            title="设备手册",
            acl_groups=("public",),
            locator="L1",
        )
        hits = [SearchHit(chunk=chunk, score=1.0)]
        user = User(id="u1", groups=frozenset({"public"}))
        store.record_trace(
            "trace-1", user, "怎么维护设备？", "维护步骤见 [1]。", hits,
            citations_from_hits(hits), LLMUsage(prompt_tokens=10, completion_tokens=5),
        )
        store.record_trace(
            "trace-2", user, "怎么更换滤芯？", "未找到权限范围内相关资料。", [],
            [], LLMUsage(),
        )
        print("feedback_saved_up=", store.save_feedback("trace-1", "up"))
        print("feedback_saved_down=", store.save_feedback("trace-2", "down", "答案没用"))
        print("weekly_failures=", store.weekly_failure_report())


if __name__ == "__main__":
    main()
