import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np


SEMANTIC_BACKEND = "sentence-transformers"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_CHUNK_CHARS = 800
SEMANTIC_CHUNK_OVERLAP = 120
SEMANTIC_SCAN_BATCH_SIZE = 256


class SemanticSetupError(RuntimeError):
    """Raised when the local embedding stack is unavailable."""


class SupportsSentenceEmbedding(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...

    def get_sentence_embedding_dimension(self) -> int:
        ...


@dataclass(frozen=True)
class SemanticIndexMeta:
    backend: str
    model_name: str
    embedding_dim: int
    chunk_chars: int
    chunk_overlap: int


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    start_char: int
    end_char: int
    text: str


_EMBEDDER: Optional[SupportsSentenceEmbedding] = None


def get_embedder() -> SupportsSentenceEmbedding:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SemanticSetupError(
            "Semantic indexing/search requires 'sentence-transformers' with a working PyTorch install. "
            "Install the dependency, then rerun indexing."
        ) from exc

    try:
        _EMBEDDER = SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception as exc:
        raise SemanticSetupError(
            f"Unable to load semantic model '{SEMANTIC_MODEL_NAME}'. "
            "The first run may need internet access to download and cache the model."
        ) from exc

    return _EMBEDDER


def build_index_meta(embedder: Optional[SupportsSentenceEmbedding] = None) -> SemanticIndexMeta:
    if embedder is None:
        embedder = get_embedder()

    return SemanticIndexMeta(
        backend=SEMANTIC_BACKEND,
        model_name=SEMANTIC_MODEL_NAME,
        embedding_dim=int(embedder.get_sentence_embedding_dimension()),
        chunk_chars=SEMANTIC_CHUNK_CHARS,
        chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
    )


def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _find_chunk_end(text: str, start: int, target_chars: int) -> int:
    hard_limit = min(len(text), start + target_chars)
    if hard_limit >= len(text):
        return len(text)

    window = text[start:hard_limit]
    whitespace_cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
    if whitespace_cut >= max(1, target_chars // 2):
        return start + whitespace_cut

    extended = hard_limit
    while extended < len(text) and (extended - start) < (target_chars + 80):
        if text[extended].isspace():
            return extended
        extended += 1

    return hard_limit


def chunk_text(
    text: str,
    *,
    chunk_chars: int = SEMANTIC_CHUNK_CHARS,
    chunk_overlap: int = SEMANTIC_CHUNK_OVERLAP,
) -> List[TextChunk]:
    if not text:
        return []

    out: List[TextChunk] = []
    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = _find_chunk_end(text, start, chunk_chars)
        trimmed_start, trimmed_end = _trim_bounds(text, start, end)

        if trimmed_end > trimmed_start:
            out.append(
                TextChunk(
                    chunk_index=chunk_index,
                    start_char=trimmed_start,
                    end_char=trimmed_end,
                    text=text[trimmed_start:trimmed_end],
                )
            )
            chunk_index += 1

        if end >= text_len:
            break

        next_start = max(start + 1, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return out

def encode_texts(
    texts: Sequence[str],
    *,
    embedder: Optional[SupportsSentenceEmbedding] = None,
) -> np.ndarray:
    if not texts:
        meta = build_index_meta(embedder)
        return np.empty((0, meta.embedding_dim), dtype=np.float32)

    if embedder is None:
        embedder = get_embedder()

    vectors = np.asarray(
        embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Keep normalization local so deterministic test doubles can skip it.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32, copy=False)


def encode_query(
    query_text: str,
    *,
    embedder: Optional[SupportsSentenceEmbedding] = None,
) -> np.ndarray:
    return encode_texts([query_text], embedder=embedder)[0]


def vector_to_blob(vector: np.ndarray) -> bytes:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Embeddings must be 1D float32 vectors")
    return arr.tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def dot_similarity(query_vector: np.ndarray, candidate_vector: np.ndarray) -> float:
    if query_vector.shape != candidate_vector.shape:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_vector.shape} candidate={candidate_vector.shape}"
        )
    return float(np.dot(query_vector, candidate_vector))


def semantic_meta_row(meta: SemanticIndexMeta) -> tuple[int, str, str, int, int, int, float]:
    return (
        1,
        meta.backend,
        meta.model_name,
        meta.embedding_dim,
        meta.chunk_chars,
        meta.chunk_overlap,
        time.time(),
    )


def fetch_semantic_meta(conn: sqlite3.Connection) -> Optional[SemanticIndexMeta]:
    try:
        row = conn.execute(
            """
            SELECT backend, model_name, embedding_dim, chunk_chars, chunk_overlap
            FROM semantic_meta
            WHERE singleton = 1
            """
        ).fetchone()
    except sqlite3.Error:
        return None

    if not row:
        return None

    return SemanticIndexMeta(
        backend=str(row[0]),
        model_name=str(row[1]),
        embedding_dim=int(row[2]),
        chunk_chars=int(row[3]),
        chunk_overlap=int(row[4]),
    )
