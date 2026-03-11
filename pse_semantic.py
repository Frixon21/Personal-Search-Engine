import sqlite3
import time
from dataclasses import dataclass
import os
import re
from typing import List, Optional, Protocol, Sequence

import numpy as np


SEMANTIC_BACKEND = "sentence-transformers"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
SEMANTIC_CHUNK_CHARS = 256
SEMANTIC_CHUNK_OVERLAP = 64
SEMANTIC_CHUNK_STRATEGY = "structured-v3"
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


class SupportsCrossEncoder(Protocol):
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SemanticIndexMeta:
    backend: str
    model_name: str
    embedding_dim: int
    chunk_chars: int
    chunk_overlap: int
    chunk_strategy: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class TextSpan:
    start_char: int
    end_char: int
    text: str


_EMBEDDER: Optional[SupportsSentenceEmbedding] = None
_RERANKER: Optional[SupportsCrossEncoder] = None


def _configure_model_loading_verbosity() -> None:
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    try:
        from huggingface_hub import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def get_embedder() -> SupportsSentenceEmbedding:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    _configure_model_loading_verbosity()

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


def get_reranker() -> SupportsCrossEncoder:
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    _configure_model_loading_verbosity()

    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise SemanticSetupError(
            "Semantic reranking requires 'sentence-transformers' with a working PyTorch install. "
            "Install the dependency, then rerun search."
        ) from exc

    try:
        _RERANKER = CrossEncoder(
            SEMANTIC_RERANK_MODEL_NAME,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise SemanticSetupError(
            f"Unable to load the semantic reranker model '{SEMANTIC_RERANK_MODEL_NAME}'. "
            "The first run may need internet access to download and cache the model."
        ) from exc

    return _RERANKER


def build_index_meta(embedder: Optional[SupportsSentenceEmbedding] = None) -> SemanticIndexMeta:
    if embedder is None:
        embedder = get_embedder()

    return SemanticIndexMeta(
        backend=SEMANTIC_BACKEND,
        model_name=SEMANTIC_MODEL_NAME,
        embedding_dim=int(embedder.get_sentence_embedding_dimension()),
        chunk_chars=SEMANTIC_CHUNK_CHARS,
        chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
        chunk_strategy=SEMANTIC_CHUNK_STRATEGY,
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


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _is_bullet_line(text: str) -> bool:
    stripped = text.lstrip()
    return bool(re.match(r"(?:[-*•]\s+|\d+[.)]\s+)", stripped))


def _is_heading_line(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(":") and _word_count(stripped) <= 4


def _iter_nonempty_line_blocks(text: str) -> List[List[TextSpan]]:
    blocks: List[List[TextSpan]] = []
    current: List[TextSpan] = []
    cursor = 0

    for raw_line in text.splitlines(keepends=True):
        line_text = raw_line.rstrip("\r\n")
        if not line_text.strip():
            if current:
                blocks.append(current)
                current = []
            cursor += len(raw_line)
            continue

        leading = len(line_text) - len(line_text.lstrip())
        trailing = len(line_text.rstrip())
        start = cursor + leading
        end = cursor + trailing
        current.append(TextSpan(start, end, text[start:end]))
        cursor += len(raw_line)

    if cursor < len(text):
        remainder = text[cursor:]
        if remainder.strip():
            leading = len(remainder) - len(remainder.lstrip())
            trailing = len(remainder.rstrip())
            start = cursor + leading
            end = cursor + trailing
            current.append(TextSpan(start, end, text[start:end]))

    if current:
        blocks.append(current)

    return blocks


def _sentence_spans(span: TextSpan) -> List[TextSpan]:
    out: List[TextSpan] = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", span.text, flags=re.DOTALL):
        rel_start, rel_end = match.span()
        start, end = _trim_bounds(span.text, rel_start, rel_end)
        if end <= start:
            continue
        abs_start = span.start_char + start
        abs_end = span.start_char + end
        out.append(TextSpan(abs_start, abs_end, span.text[start:end]))
    return out


def _split_long_span(
    span: TextSpan,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> List[TextSpan]:
    out: List[TextSpan] = []
    rel_start = 0
    text_len = len(span.text)

    while rel_start < text_len:
        rel_end = _find_chunk_end(span.text, rel_start, chunk_chars)
        trimmed_start, trimmed_end = _trim_bounds(span.text, rel_start, rel_end)
        if trimmed_end > trimmed_start:
            abs_start = span.start_char + trimmed_start
            abs_end = span.start_char + trimmed_end
            out.append(TextSpan(abs_start, abs_end, span.text[trimmed_start:trimmed_end]))

        if rel_end >= text_len:
            break

        next_start = max(rel_start + 1, rel_end - chunk_overlap)
        if next_start <= rel_start:
            next_start = rel_end
        rel_start = next_start

    return out


def _merge_heading_spans(text: str, spans: List[TextSpan], chunk_chars: int) -> List[TextSpan]:
    merged: List[TextSpan] = []
    idx = 0
    while idx < len(spans):
        current = spans[idx]
        if (
            idx + 1 < len(spans)
            and _is_heading_line(current.text)
            and (spans[idx + 1].end_char - current.start_char) <= chunk_chars
        ):
            nxt = spans[idx + 1]
            merged.append(TextSpan(current.start_char, nxt.end_char, text[current.start_char:nxt.end_char]))
            idx += 2
            continue

        merged.append(current)
        idx += 1

    return merged


def _block_to_spans(
    text: str,
    block: List[TextSpan],
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> List[TextSpan]:
    if not block:
        return []

    if len(block) == 1:
        paragraph = block[0]
        spans = _sentence_spans(paragraph) or [paragraph]
    elif any(_is_bullet_line(line.text) for line in block):
        spans = [
            TextSpan(
                block[0].start_char,
                block[-1].end_char,
                text[block[0].start_char:block[-1].end_char],
            )
        ]
    elif all(line.text.rstrip().endswith((".", "!", "?", ":")) for line in block):
        spans = block
    else:
        paragraph = TextSpan(block[0].start_char, block[-1].end_char, text[block[0].start_char:block[-1].end_char])
        spans = _sentence_spans(paragraph) or [paragraph]

    out: List[TextSpan] = []
    for span in spans:
        if len(span.text) > chunk_chars:
            out.extend(_split_long_span(span, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap))
        else:
            out.append(span)
    return out


def _span_contains_list(text: str) -> bool:
    return any(_is_bullet_line(line) for line in text.splitlines())


def _merge_adjacent_context_spans(text: str, spans: List[TextSpan], chunk_chars: int) -> List[TextSpan]:
    if len(spans) < 2:
        return spans

    merged: List[TextSpan] = []
    idx = 0
    while idx < len(spans):
        current = spans[idx]
        if (
            idx + 1 < len(spans)
            and _span_contains_list(current.text)
            and not _span_contains_list(spans[idx + 1].text)
            and (spans[idx + 1].end_char - current.start_char) <= chunk_chars
        ):
            nxt = spans[idx + 1]
            merged.append(TextSpan(current.start_char, nxt.end_char, text[current.start_char:nxt.end_char]))
            idx += 2
            continue

        merged.append(current)
        idx += 1

    return merged


def chunk_text(
    text: str,
    *,
    chunk_chars: int = SEMANTIC_CHUNK_CHARS,
    chunk_overlap: int = SEMANTIC_CHUNK_OVERLAP,
) -> List[TextChunk]:
    if not text:
        return []

    spans: List[TextSpan] = []
    for block in _iter_nonempty_line_blocks(text):
        spans.extend(
            _block_to_spans(
                text,
                block,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
            )
        )

    if not spans:
        return []

    spans = _merge_adjacent_context_spans(text, spans, chunk_chars)

    return [
        TextChunk(
            chunk_index=idx,
            start_char=span.start_char,
            end_char=span.end_char,
            text=span.text,
        )
        for idx, span in enumerate(spans)
    ]

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


def rerank_texts(
    query_text: str,
    texts: Sequence[str],
    *,
    reranker: Optional[SupportsCrossEncoder] = None,
) -> np.ndarray:
    if not texts:
        return np.empty((0,), dtype=np.float32)

    if reranker is None:
        reranker = get_reranker()

    pairs = [(query_text, text) for text in texts]
    scores = np.asarray(
        reranker.predict(
            pairs,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    return scores


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


def semantic_meta_row(meta: SemanticIndexMeta) -> tuple[int, str, str, int, int, int, str, float]:
    return (
        1,
        meta.backend,
        meta.model_name,
        meta.embedding_dim,
        meta.chunk_chars,
        meta.chunk_overlap,
        meta.chunk_strategy,
        time.time(),
    )


def fetch_semantic_meta(conn: sqlite3.Connection) -> Optional[SemanticIndexMeta]:
    try:
        row = conn.execute(
            """
            SELECT backend, model_name, embedding_dim, chunk_chars, chunk_overlap, chunk_strategy
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
        chunk_strategy=str(row[5]),
    )
