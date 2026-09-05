from rag_permission.config import Settings
from rag_permission.embeddings import BGEEmbedding
from rag_permission.llm import OpenAILLMClient


def main() -> None:
    settings = Settings()

    embedder = BGEEmbedding(settings.embedding_model)
    vectors = embedder.encode(["E-1002 是什么故障？"])
    if len(vectors) != 1 or len(vectors[0]) != 1024:
        raise SystemExit(f"embedder returned {len(vectors)}x{len(vectors[0])}, expected 1x1024")
    print("embedder: 1024 dims OK")

    llm = OpenAILLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    response = llm.complete("你是一个测试助手。", "只回复两个字：正常")
    if not response.text.strip():
        raise SystemExit("LLM returned an empty response")
    print(f"llm: response OK ({response.usage.total_tokens} tokens)")


if __name__ == "__main__":
    main()
