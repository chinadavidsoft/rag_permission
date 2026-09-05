import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_permission.bm25 import BM25Index
from rag_permission.config import Settings
from rag_permission.embeddings import BGEEmbedding
from rag_permission.evaluation.runner import (
    EvaluationRunner,
    load_golden_set,
    summarize_evaluations,
)
from rag_permission.ingest_pipeline import IngestionPipeline
from rag_permission.reranker import BGEReranker
from rag_permission.retriever import HybridRetriever
from rag_permission.vector_store import QdrantVectorStore, create_qdrant_client


def build_retriever(
    settings: Settings, embedding: BGEEmbedding, store: QdrantVectorStore, bm25: BM25Index, rerank: bool
) -> HybridRetriever:
    return HybridRetriever(
        embedding,
        store,
        bm25,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        reranker=BGEReranker(settings.reranker_model) if rerank else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with and without reranking")
    parser.add_argument("--golden-set", default="fixtures/golden_set.json")
    parser.add_argument("--authorization-set", default="fixtures/authorization_set.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    settings = Settings()
    embedding = BGEEmbedding(settings.embedding_model)
    store = QdrantVectorStore(create_qdrant_client(":memory:"), settings.collection_name, 1024)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(embedding, store, bm25)
    for filename, groups in (
        ("sample.md", ("public",)),
        ("sample.docx", ("eng",)),
        ("sample.pdf", ("hr",)),
    ):
        pipeline.ingest(
            settings.fixture_dir / filename,
            groups,
            strategy=settings.ingestion_strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            parent_chunk_size=settings.parent_chunk_size,
        )

    baseline_retriever = build_retriever(settings, embedding, store, bm25, rerank=False)
    reranked_retriever = build_retriever(settings, embedding, store, bm25, rerank=True)
    cases = load_golden_set(args.golden_set)
    baseline = EvaluationRunner(baseline_retriever).run(cases)
    reranked = EvaluationRunner(reranked_retriever).run(cases)
    authorization = EvaluationRunner(baseline_retriever).run(load_golden_set(args.authorization_set))
    baseline_by_id = {(item.query, item.user_groups): item for item in baseline}
    rerank_target_moves = []
    for item in reranked:
        baseline_item = baseline_by_id[(item.query, item.user_groups)]
        if baseline_item.mrr and item.mrr > baseline_item.mrr:
            rerank_target_moves.append(
                {
                    "query": item.query,
                    "user_groups": list(item.user_groups),
                    "baseline_mrr": baseline_item.mrr,
                    "reranked_mrr": item.mrr,
                }
            )

    output = {
        "baseline": [asdict(result) for result in baseline],
        "reranked": [asdict(result) for result in reranked],
        "baseline_summary": asdict(summarize_evaluations(baseline)),
        "reranked_summary": asdict(summarize_evaluations(reranked)),
        "rerank_target_moves": rerank_target_moves,
        "authorization_summary": asdict(summarize_evaluations(authorization)),
        "authorization": [asdict(result) for result in authorization],
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(output["baseline_summary"], ensure_ascii=False))
    print(json.dumps(output["reranked_summary"], ensure_ascii=False))
    print(json.dumps(output["authorization_summary"], ensure_ascii=False))
    for move in rerank_target_moves:
        print(json.dumps(move, ensure_ascii=False))


if __name__ == "__main__":
    main()
