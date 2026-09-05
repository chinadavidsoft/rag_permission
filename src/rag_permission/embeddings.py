import hashlib
import threading
from typing import Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


_model_cache: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()


class BGEEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        with _model_lock:
            if model_name not in _model_cache:
                _model_cache[model_name] = SentenceTransformer(model_name, device=device)
            self.model = _model_cache[model_name]
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        with self._cache_lock:
            missing = [text for text in texts if self._hash(text) not in self._cache]
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            with self._cache_lock:
                for text, vector in zip(missing, vectors, strict=True):
                    self._cache[self._hash(text)] = vector.tolist()
        return [self._cache[self._hash(text)] for text in texts]

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
