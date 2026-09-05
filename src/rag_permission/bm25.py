import math
import re
from collections import Counter

import jieba

from rag_permission.models import DocumentChunk, SearchHit


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.extend(token for token in jieba.lcut(value) if token.strip())
        else:
            tokens.append(value.lower())
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.doc_tokens: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.df: Counter[str] = Counter()
        self._chunk_ids: set[str] = set()

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self._chunk_ids:
                continue
            self._chunk_ids.add(chunk.chunk_id)
            tokens = tokenize(chunk.text)
            self.chunks.append(chunk)
            self.doc_tokens.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))
            self.df.update(set(tokens))

    def search(
        self, query: str, limit: int, acl_groups: frozenset[str], oversample: int = 4
    ) -> list[SearchHit]:
        if not acl_groups or limit <= 0 or not self.chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        average_length = sum(self.doc_lengths) / len(self.doc_lengths) or 1.0
        scores = []
        for index, token_counts in enumerate(self.doc_tokens):
            score = 0.0
            length_norm = self.k1 * (
                1 - self.b + self.b * self.doc_lengths[index] / average_length
            )
            for token in query_tokens:
                frequency = token_counts.get(token, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1
                    + (len(self.chunks) - self.df[token] + 0.5)
                    / (self.df[token] + 0.5)
                )
                score += idf * frequency * (self.k1 + 1) / (frequency + length_norm)
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for score, index in scores[: limit * oversample]:
            chunk = self.chunks[index]
            if frozenset(chunk.acl_groups) & acl_groups:
                hits.append(SearchHit(chunk=chunk, score=score))
                if len(hits) == limit:
                    break
        return hits
