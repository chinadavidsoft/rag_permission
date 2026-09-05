from rag_permission.evaluation.runner import GoldenCase, evaluate_case
from rag_permission.models import DocumentChunk, SearchHit, User


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            doc_id=chunk_id.split(":", 1)[0],
            text="text",
            chunk_type="section",
            section_path=("section",),
            source="sample.md",
            title="sample",
            acl_groups=("public",),
            locator="L1",
        ),
        score=1.0,
    )


class FakeRetriever:
    def search(self, query: str, user: User) -> list[SearchHit]:
        if user.groups == frozenset({"public"}):
            return [hit("doc-a:section:1"), hit("doc-b:section:1")]
        return []


def main() -> None:
    case = GoldenCase(
        query="E-1002",
        user_groups=("public",),
        relevant_chunk_ids=("doc-a:section:1",),
        forbidden_chunk_ids=("doc-hr:section:1",),
    )
    result = evaluate_case(case, FakeRetriever().search(case.query, User("u", frozenset(case.user_groups))))
    print(f"recall={result.recall}, precision={result.precision}, mrr={result.mrr}, leakage={result.leakage}")


if __name__ == "__main__":
    main()
