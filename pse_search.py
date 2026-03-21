from dataclasses import dataclass, replace
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pse_common import (
    db_path_for_root,
    human_size,
    metadata_bonus,
    open_db_connection,
    recency_bonus,
    score_filename,
    tokenize,
    fold_text,
)
from pse_semantic import (
    SEMANTIC_SCAN_BATCH_SIZE,
    SemanticSetupError,
    blob_to_vector,
    build_index_meta,
    dot_similarity,
    encode_query,
    fetch_semantic_meta,
    get_reranker,
    rerank_texts,
)


SEARCH_DB_PRAGMAS = ("temp_store=MEMORY",)
SEARCH_MODES = ("lexical", "semantic", "hybrid")
SEARCH_HELP_TEXT = """Usage:
  python pse_search.py <root_folder> "<query>" [max_results] [options]
  python pse_search.py <root_folder> [max_results] --interactive [options]

Search the SQLite index for a root folder.

Arguments:
  <root_folder>   Folder that contains the .pse_index.sqlite3 index.
  <query>         Query text to search for. If you do not quote it, the remaining
                  non-option tokens are joined into a single query string.
                  Not used in interactive mode.
  [max_results]   Optional positive integer result limit. Default: 20.

Options:
  --lexical       Use keyword/content ranking only.
  --semantic      Use semantic retrieval only.
                  By default, search uses a combined lexical + semantic ranking.
  --interactive   Start a simple query loop and reuse loaded models in one process.
  --debug         Print timings and scoring breakdowns. Also enables snippets.
  --snippets      Show cached snippet text for each result.
  -h, --help      Show this help message and exit.
"""

HYBRID_LEXICAL_WEIGHT = 1.0
HYBRID_SEMANTIC_WEIGHT = 1.0
HYBRID_METADATA_WEIGHT = 0.20
HYBRID_RECENCY_WEIGHT = 0.10
SEMANTIC_MULTI_LINE_LIST_MULTIPLIER = 0.95
SEMANTIC_SINGLE_BULLET_MULTIPLIER = 0.80
SEMANTIC_HEADING_MULTIPLIER = 0.85
SEMANTIC_RERANK_TOP_K = 10
SEMANTIC_RERANK_CHAR_CAP = 2_000


@dataclass(frozen=True)
class RankedDoc:
    doc_id: int
    path: str
    size: int
    mtime: float
    preview_text: Optional[str]
    debug: Tuple[int, int, int, int, int, int, int]
    total: int


@dataclass(frozen=True)
class SemanticDocHit:
    doc_id: int
    similarity: float
    chunk_text: str


@dataclass(frozen=True)
class SemanticRankedDoc:
    doc_id: int
    path: str
    size: int
    mtime: float
    preview_text: Optional[str]
    chunk_text: str
    similarity: float
    structure_multiplier: float
    base_score: float
    rerank_raw: Optional[float]
    rerank_bonus: float
    score: float
    debug: Tuple[int, int, int, int]


@dataclass(frozen=True)
class HybridRankedDoc:
    doc_id: int
    path: str
    size: int
    mtime: float
    preview_text: Optional[str]
    chunk_text: Optional[str]
    lexical_raw: float
    lexical_norm: float
    semantic_raw: float
    semantic_norm: float
    semantic_similarity: float
    semantic_structure_multiplier: float
    rerank_raw: Optional[float]
    rerank_bonus: float
    fn_hits: int
    fn_sub: int
    meta: int
    rec: int
    base_total: float
    total: float


@dataclass(frozen=True)
class SearchResult:
    rank: int
    doc_id: int
    path: str
    name: str
    extension: str
    size: int
    mtime: float
    preview_text: Optional[str]
    chunk_text: Optional[str]
    snippet_lines: Tuple[str, ...]
    score: float
    score_display: str
    debug: Dict[str, object]


@dataclass(frozen=True)
class SearchResponse:
    root: str
    index_path: Optional[str]
    query: str
    query_terms: Tuple[str, ...]
    mode: str
    max_results: int
    candidate_count: int
    results: Tuple[SearchResult, ...]
    error: Optional[str] = None
    query_time_seconds: Optional[float] = None


def gather_content_candidates(conn: sqlite3.Connection, q_terms: List[str]) -> Dict[int, Dict[str, int]]:
    if not q_terms:
        return {}

    candidates: Dict[int, Dict[str, int]] = {}
    for term in q_terms:
        rows = conn.execute("SELECT doc_id, count FROM terms WHERE term = ?", (term,)).fetchall()
        for doc_id, cnt in rows:
            doc_id = int(doc_id)
            cnt = int(cnt)
            candidates.setdefault(doc_id, {})[term] = cnt
    return candidates


def filename_candidate_docs(conn: sqlite3.Connection, query: str) -> List[int]:
    rows = conn.execute("SELECT doc_id, path, ext FROM docs").fetchall()
    out: List[int] = []
    for doc_id, path, ext in rows:
        name = Path(str(path)).stem
        fn_hits, fn_sub, _fn_pos = score_filename(query, name, str(ext))
        if fn_hits > 0 or fn_sub > 0:
            out.append(int(doc_id))
    return out


def content_score(q_terms: List[str], term_counts: Dict[str, int]) -> Tuple[int, int]:
    unique = sum(1 for t in q_terms if t in term_counts)
    total = sum(term_counts.get(t, 0) for t in q_terms)
    return unique, total


def fetch_doc_details(
    conn: sqlite3.Connection,
    doc_id: int,
) -> Optional[Tuple[str, str, int, float, Optional[str]]]:
    row = conn.execute(
        """
        SELECT d.path, d.ext, d.size, d.mtime, p.preview_text
        FROM docs AS d
        LEFT JOIN doc_previews AS p ON p.doc_id = d.doc_id
        WHERE d.doc_id = ?
        """,
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    return (
        str(row[0]),
        str(row[1]),
        int(row[2]),
        float(row[3]),
        str(row[4]) if row[4] is not None else None,
    )


def highlight_terms(line: str, q_terms: List[str]) -> str:
    out = fold_text(line)
    for t in sorted(set(q_terms), key=len, reverse=True):
        if not t:
            continue
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", f"[{t}]", out)
    return out


def best_effort_snippet(preview_text: Optional[str], q_terms: List[str], max_lines: int = 2) -> Optional[List[str]]:
    if not preview_text or not q_terms:
        return None

    lines = preview_text.splitlines()
    qset = set(q_terms)

    for i, line in enumerate(lines):
        toks = set(tokenize(line))
        if qset.intersection(toks):
            # Return at most 2 lines: the hit line and one after it.
            start = max(0, i)
            end = min(len(lines), i + 2)
            out: List[str] = []
            for j in range(start, end):
                out.append(highlight_terms(lines[j], q_terms))
                if len(out) >= max_lines:
                    break
            return out

    return None


def best_chunk_snippet(chunk_text: str, q_terms: List[str], max_lines: int = 2) -> Optional[List[str]]:
    if not chunk_text:
        return None

    raw_lines = [line.strip() for line in fold_text(chunk_text).splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [fold_text(chunk_text).strip()]

    selected_lines = raw_lines[:max_lines]
    qset = set(q_terms)
    if qset:
        for idx, line in enumerate(raw_lines):
            if qset.intersection(tokenize(line)):
                selected_lines = raw_lines[idx:idx + max_lines]
                break
        else:
            selected_lines = raw_lines[-1:]
    elif len(raw_lines) > max_lines:
        selected_lines = raw_lines[-1:]

    out: List[str] = []
    for line in selected_lines:
        excerpt = line
        if len(excerpt) > 240:
            excerpt = excerpt[:237].rstrip() + "..."
        if q_terms:
            excerpt = highlight_terms(excerpt, q_terms)
        out.append(excerpt)

    return out or None


def overall_score(
    content_unique: int,
    content_hits: int,
    fn_hits: int,
    fn_sub: int,
    meta: int,
    rec: int,
) -> int:
    """
    Single number score for clean output.

    Tunable weights:
    - content_unique: strongest signal
    - content_hits: supports repeated mentions
    - filename hits/sub: moderate
    - meta: moderate
    - rec: tiny
    """
    return lexical_evidence_score(
        content_unique,
        content_hits,
        fn_hits,
        fn_sub,
    ) + (
        meta * 8 +
        rec * 2
    )


def lexical_evidence_score(
    content_unique: int,
    content_hits: int,
    fn_hits: int,
    fn_sub: int,
) -> int:
    return (
        content_unique * 20 +
        min(content_hits, 20) * 4 +
        fn_hits * 10 +
        fn_sub * 50
    )


def normalize_scores(raw_scores: Dict[int, float]) -> Dict[int, float]:
    if not raw_scores:
        return {}

    low = min(raw_scores.values())
    high = max(raw_scores.values())
    if high == low:
        return {
            doc_id: (1.0 if score > 0 else 0.0)
            for doc_id, score in raw_scores.items()
        }

    return {
        doc_id: (score - low) / (high - low)
        for doc_id, score in raw_scores.items()
    }


def normalize_rerank_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []

    low = min(raw_scores)
    high = max(raw_scores)
    if high == low:
        return [0.0 for _ in raw_scores]

    return [(score - low) / (high - low) for score in raw_scores]


def hybrid_bonus(meta: int, rec: int) -> float:
    meta_norm = meta / 3.0 if meta > 0 else 0.0
    rec_norm = rec / 3.0 if rec > 0 else 0.0
    return (
        meta_norm * HYBRID_METADATA_WEIGHT +
        rec_norm * HYBRID_RECENCY_WEIGHT
    )


def semantic_rerank_text(path: str, chunk_text: Optional[str], preview_text: Optional[str]) -> str:
    stem = Path(path).stem
    for value in (preview_text, chunk_text):
        if value and value.strip():
            text = value.strip()
            if len(text) > SEMANTIC_RERANK_CHAR_CAP:
                text = text[:SEMANTIC_RERANK_CHAR_CAP]
            return f"{stem}\n\n{text}"
    return stem


def rerank_top_semantic_docs(
    query_str: str,
    ranked: List[SemanticRankedDoc],
    *,
    top_k: int = SEMANTIC_RERANK_TOP_K,
) -> List[SemanticRankedDoc]:
    if len(ranked) < 2:
        return ranked

    top_docs = ranked[:top_k]
    texts = [semantic_rerank_text(doc.path, doc.chunk_text, doc.preview_text) for doc in top_docs]

    try:
        raw_scores = [float(score) for score in rerank_texts(query_str, texts)]
    except SemanticSetupError:
        return ranked

    bonuses = normalize_rerank_scores(raw_scores)
    reranked = [
        replace(
            doc,
            rerank_raw=raw,
            rerank_bonus=bonus,
            score=doc.base_score + bonus,
        )
        for doc, raw, bonus in zip(top_docs, raw_scores, bonuses)
    ]
    reranked.sort(
        key=lambda doc: (
            doc.score,
            float("-inf") if doc.rerank_raw is None else doc.rerank_raw,
            doc.base_score,
            doc.similarity,
            doc.mtime,
        ),
        reverse=True,
    )
    return reranked + ranked[top_k:]


def rerank_top_hybrid_docs(
    query_str: str,
    ranked: List[HybridRankedDoc],
    *,
    top_k: int = SEMANTIC_RERANK_TOP_K,
) -> List[HybridRankedDoc]:
    if len(ranked) < 2:
        return ranked

    top_docs = ranked[:top_k]
    texts = [semantic_rerank_text(doc.path, doc.chunk_text, doc.preview_text) for doc in top_docs]

    try:
        raw_scores = [float(score) for score in rerank_texts(query_str, texts)]
    except SemanticSetupError:
        return ranked

    bonuses = normalize_rerank_scores(raw_scores)
    reranked = [
        replace(
            doc,
            rerank_raw=raw,
            rerank_bonus=bonus,
            total=doc.base_total + bonus,
        )
        for doc, raw, bonus in zip(top_docs, raw_scores, bonuses)
    ]
    reranked.sort(
        key=lambda doc: (
            doc.total,
            float("-inf") if doc.rerank_raw is None else doc.rerank_raw,
            doc.base_total,
            doc.semantic_norm,
            doc.lexical_norm,
            doc.mtime,
        ),
        reverse=True,
    )
    return reranked + ranked[top_k:]


def semantic_structure_multiplier(chunk_text: str) -> float:
    folded = fold_text(chunk_text).strip()
    if not folded:
        return 0.0

    lines = [line.strip() for line in folded.splitlines() if line.strip()]
    if not lines:
        return 1.0

    bullet_lines = [
        line for line in lines
        if re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)", line)
    ]
    if bullet_lines and len(lines) >= 3:
        return SEMANTIC_MULTI_LINE_LIST_MULTIPLIER
    if len(lines) == 1 and bullet_lines:
        return SEMANTIC_SINGLE_BULLET_MULTIPLIER
    if len(lines) == 1 and lines[0].endswith(":") and len(tokenize(lines[0])) <= 6:
        return SEMANTIC_HEADING_MULTIPLIER

    return 1.0


def _resolve_search_db(root: Path) -> Tuple[Path, Optional[str]]:
    root = root.expanduser().resolve()
    dbp = db_path_for_root(root)
    if not dbp.exists():
        return dbp, f"No index found at: {dbp}\nRun: python pse_index.py index <root>"

    return dbp, None


def validate_search_db(root: Path) -> Optional[Path]:
    dbp, error = _resolve_search_db(root)
    if error is not None:
        print(error)
        return None

    return dbp


def _semantic_ready_error(conn: sqlite3.Connection, dbp: Path) -> Optional[str]:
    expected_meta = build_index_meta()
    if fetch_semantic_meta(conn) != expected_meta:
        return (
            f"Index at {dbp} is outdated and must be rebuilt before semantic searching.\n"
            "Run: python pse_index.py index <root>"
        )

    return None


def ensure_semantic_ready(conn: sqlite3.Connection, dbp: Path) -> bool:
    error = _semantic_ready_error(conn, dbp)
    if error is not None:
        print(error)
        return False

    return True


def _normalize_search_mode(mode: str) -> str:
    normalized = (mode or "hybrid").strip().casefold()
    if normalized not in SEARCH_MODES:
        raise ValueError(
            f"Unsupported search mode: {mode}. Expected one of: {', '.join(SEARCH_MODES)}"
        )
    return normalized


def _public_result_from_ranked_doc(
    mode: str,
    rank: int,
    doc: RankedDoc | SemanticRankedDoc | HybridRankedDoc,
    q_terms: List[str],
) -> SearchResult:
    path = Path(doc.path)
    snippet_lines: Tuple[str, ...]
    score: float
    score_display: str
    debug: Dict[str, object]
    preview_text = doc.preview_text
    chunk_text: Optional[str]

    if mode == "lexical":
        lexical_doc = doc
        assert isinstance(lexical_doc, RankedDoc)
        cu, ct, fn_hits, fn_sub, meta, fn_pos, rec = lexical_doc.debug
        snippet_lines = tuple(_keyword_snippet(lexical_doc, q_terms) or ())
        score = float(lexical_doc.total)
        score_display = str(lexical_doc.total)
        chunk_text = None
        debug = {
            "content_unique": cu,
            "query_term_count": len(q_terms),
            "content_hits": ct,
            "filename_hits": fn_hits,
            "filename_substring": fn_sub,
            "metadata_bonus": meta,
            "filename_position": fn_pos,
            "recency_bonus": rec,
        }
    elif mode == "semantic":
        semantic_doc = doc
        assert isinstance(semantic_doc, SemanticRankedDoc)
        fn_hits, fn_sub, meta, rec = semantic_doc.debug
        snippet_lines = tuple(_semantic_snippet(semantic_doc, q_terms) or ())
        score = semantic_doc.score
        score_display = f"{semantic_doc.score:.4f}"
        chunk_text = semantic_doc.chunk_text
        debug = {
            "semantic_similarity": semantic_doc.similarity,
            "structure_multiplier": semantic_doc.structure_multiplier,
            "base_score": semantic_doc.base_score,
            "rerank_raw": semantic_doc.rerank_raw,
            "rerank_bonus": semantic_doc.rerank_bonus,
            "filename_hits": fn_hits,
            "filename_substring": fn_sub,
            "metadata_bonus": meta,
            "recency_bonus": rec,
        }
    else:
        hybrid_doc = doc
        assert isinstance(hybrid_doc, HybridRankedDoc)
        snippet_lines = tuple(_hybrid_snippet(hybrid_doc, q_terms) or ())
        score = hybrid_doc.total
        score_display = f"{hybrid_doc.total:.4f}"
        chunk_text = hybrid_doc.chunk_text
        debug = {
            "lexical_raw": hybrid_doc.lexical_raw,
            "lexical_norm": hybrid_doc.lexical_norm,
            "semantic_raw": hybrid_doc.semantic_raw,
            "semantic_norm": hybrid_doc.semantic_norm,
            "semantic_similarity": hybrid_doc.semantic_similarity,
            "semantic_structure_multiplier": hybrid_doc.semantic_structure_multiplier,
            "rerank_raw": hybrid_doc.rerank_raw,
            "rerank_bonus": hybrid_doc.rerank_bonus,
            "filename_hits": hybrid_doc.fn_hits,
            "filename_substring": hybrid_doc.fn_sub,
            "metadata_bonus": hybrid_doc.meta,
            "recency_bonus": hybrid_doc.rec,
        }

    return SearchResult(
        rank=rank,
        doc_id=doc.doc_id,
        path=doc.path,
        name=path.name,
        extension=path.suffix.lower(),
        size=doc.size,
        mtime=doc.mtime,
        preview_text=preview_text,
        chunk_text=chunk_text,
        snippet_lines=snippet_lines,
        score=score,
        score_display=score_display,
        debug=debug,
    )


def run_search(
    root: Path,
    query_str: str,
    *,
    mode: str = "hybrid",
    max_results: int = 20,
) -> SearchResponse:
    normalized_mode = _normalize_search_mode(mode)
    root = root.expanduser().resolve()
    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    base_response = {
        "root": str(root),
        "index_path": None,
        "query": query_str,
        "query_terms": (),
        "mode": normalized_mode,
        "max_results": max_results,
        "candidate_count": 0,
        "results": (),
    }

    if not query_str.strip():
        return SearchResponse(**base_response)

    dbp, error = _resolve_search_db(root)
    if error is not None:
        return SearchResponse(
            **{
                **base_response,
                "index_path": str(dbp),
                "error": error,
            }
        )

    build_ranked: Callable[[sqlite3.Connection, str], Tuple[List[str], List[Any]]]
    rerank_ranked: Optional[Callable[[str, List[Any]], List[Any]]]
    require_semantic = False
    if normalized_mode == "lexical":
        build_ranked = build_keyword_ranked_docs
        rerank_ranked = None
    elif normalized_mode == "semantic":
        build_ranked = build_semantic_ranked_docs
        rerank_ranked = rerank_top_semantic_docs
        require_semantic = True
    else:
        build_ranked = build_hybrid_ranked_docs
        rerank_ranked = rerank_top_hybrid_docs
        require_semantic = True

    conn = open_db_connection(dbp, SEARCH_DB_PRAGMAS)
    try:
        if require_semantic:
            semantic_error = _semantic_ready_error(conn, dbp)
            if semantic_error is not None:
                return SearchResponse(
                    **{
                        **base_response,
                        "index_path": str(dbp),
                        "error": semantic_error,
                    }
                )

        t0 = time.perf_counter()
        q_terms, ranked = build_ranked(conn, query_str)
        if rerank_ranked is not None:
            ranked = rerank_ranked(query_str, ranked)
        query_time = time.perf_counter() - t0

        results = tuple(
            _public_result_from_ranked_doc(normalized_mode, idx, doc, q_terms)
            for idx, doc in enumerate(ranked[:max_results], start=1)
        )
        return SearchResponse(
            root=str(root),
            index_path=str(dbp),
            query=query_str,
            query_terms=tuple(q_terms),
            mode=normalized_mode,
            max_results=max_results,
            candidate_count=len(ranked),
            results=results,
            query_time_seconds=query_time,
        )
    except SemanticSetupError as exc:
        return SearchResponse(
            **{
                **base_response,
                "index_path": str(dbp),
                "error": str(exc),
            }
        )
    finally:
        conn.close()


def build_keyword_ranked_docs(
    conn: sqlite3.Connection,
    query_str: str,
) -> Tuple[List[str], List[RankedDoc]]:
    q_terms = tokenize(query_str)
    cand = gather_content_candidates(conn, q_terms)

    for doc_id in filename_candidate_docs(conn, query_str):
        cand.setdefault(doc_id, {})

    ranked: List[RankedDoc] = []
    for doc_id, per_term_counts in cand.items():
        doc_details = fetch_doc_details(conn, doc_id)
        if doc_details is None:
            continue

        path, ext, size, mtime, preview_text = doc_details
        name = Path(path).stem

        cu, ct = content_score(q_terms, per_term_counts)
        fn_hits, fn_sub, fn_pos = score_filename(query_str, name, ext)
        meta = metadata_bonus(query_str, mtime)
        rec = recency_bonus(mtime)

        total = overall_score(cu, ct, fn_hits, fn_sub, meta, rec)
        dbg = (cu, ct, fn_hits, fn_sub, meta, fn_pos, rec)
        ranked.append(
            RankedDoc(
                doc_id=doc_id,
                path=path,
                size=size,
                mtime=mtime,
                preview_text=preview_text,
                debug=dbg,
                total=total,
            )
        )

    ranked.sort(key=lambda doc: (doc.total, doc.debug), reverse=True)
    return q_terms, ranked


def build_semantic_ranked_docs(
    conn: sqlite3.Connection,
    query_str: str,
) -> Tuple[List[str], List[SemanticRankedDoc]]:
    q_terms = tokenize(query_str)
    query_vector = encode_query(query_str)
    cand = gather_semantic_candidates(conn, query_vector)

    ranked: List[SemanticRankedDoc] = []
    for doc_id, hit in cand.items():
        doc_details = fetch_doc_details(conn, doc_id)
        if doc_details is None:
            continue

        path, ext, size, mtime, preview_text = doc_details
        name = Path(path).stem

        fn_hits, fn_sub, _fn_pos = score_filename(query_str, name, ext)
        meta = metadata_bonus(query_str, mtime)
        rec = recency_bonus(mtime)
        structure_multiplier = semantic_structure_multiplier(hit.chunk_text)
        score = hit.similarity * structure_multiplier

        ranked.append(
            SemanticRankedDoc(
                doc_id=doc_id,
                path=path,
                size=size,
                mtime=mtime,
                preview_text=preview_text,
                chunk_text=hit.chunk_text,
                similarity=hit.similarity,
                structure_multiplier=structure_multiplier,
                base_score=score,
                rerank_raw=None,
                rerank_bonus=0.0,
                score=score,
                debug=(fn_hits, fn_sub, meta, rec),
            )
        )

    ranked.sort(
        key=lambda doc: (doc.score, doc.similarity, doc.debug[1], doc.debug[0], doc.debug[2], doc.debug[3]),
        reverse=True,
    )
    return q_terms, ranked


def build_hybrid_ranked_docs(
    conn: sqlite3.Connection,
    query_str: str,
) -> Tuple[List[str], List[HybridRankedDoc]]:
    q_terms, lexical_ranked = build_keyword_ranked_docs(conn, query_str)
    _semantic_terms, semantic_ranked = build_semantic_ranked_docs(conn, query_str)

    lexical_by_id = {doc.doc_id: doc for doc in lexical_ranked}
    semantic_by_id = {doc.doc_id: doc for doc in semantic_ranked}

    lexical_raw = {
        doc.doc_id: float(lexical_evidence_score(doc.debug[0], doc.debug[1], doc.debug[2], doc.debug[3]))
        for doc in lexical_ranked
    }
    semantic_raw = {doc.doc_id: doc.score for doc in semantic_ranked}

    lexical_norm = normalize_scores(lexical_raw)
    semantic_norm = normalize_scores(semantic_raw)

    ranked: List[HybridRankedDoc] = []
    for doc_id in sorted(set(lexical_by_id) | set(semantic_by_id)):
        lexical_doc = lexical_by_id.get(doc_id)
        semantic_doc = semantic_by_id.get(doc_id)
        source_doc = lexical_doc if lexical_doc is not None else semantic_doc
        if source_doc is None:
            continue

        if lexical_doc is not None:
            _cu, _ct, fn_hits, fn_sub, meta, _fn_pos, rec = lexical_doc.debug
            preview_text = lexical_doc.preview_text
            structure_multiplier = semantic_doc.structure_multiplier if semantic_doc is not None else 1.0
        else:
            fn_hits, fn_sub, meta, rec = semantic_doc.debug
            preview_text = semantic_doc.preview_text
            structure_multiplier = semantic_doc.structure_multiplier

        total = (
            lexical_norm.get(doc_id, 0.0) * HYBRID_LEXICAL_WEIGHT +
            semantic_norm.get(doc_id, 0.0) * HYBRID_SEMANTIC_WEIGHT +
            hybrid_bonus(meta, rec)
        )

        ranked.append(
            HybridRankedDoc(
                doc_id=doc_id,
                path=source_doc.path,
                size=source_doc.size,
                mtime=source_doc.mtime,
                preview_text=preview_text,
                chunk_text=semantic_doc.chunk_text if semantic_doc is not None else None,
                lexical_raw=lexical_raw.get(doc_id, 0.0),
                lexical_norm=lexical_norm.get(doc_id, 0.0),
                semantic_raw=semantic_raw.get(doc_id, 0.0),
                semantic_norm=semantic_norm.get(doc_id, 0.0),
                semantic_similarity=semantic_doc.similarity if semantic_doc is not None else 0.0,
                semantic_structure_multiplier=structure_multiplier,
                rerank_raw=None,
                rerank_bonus=0.0,
                fn_hits=fn_hits,
                fn_sub=fn_sub,
                meta=meta,
                rec=rec,
                base_total=total,
                total=total,
            )
        )

    ranked.sort(
        key=lambda doc: (
            doc.total,
            doc.semantic_norm,
            doc.lexical_norm,
            doc.fn_sub,
            doc.fn_hits,
            doc.meta,
            doc.rec,
            doc.semantic_raw,
            doc.lexical_raw,
            doc.mtime,
        ),
        reverse=True,
    )
    return q_terms, ranked


def gather_semantic_candidates(
    conn: sqlite3.Connection,
    query_vector,
    batch_size: int = SEMANTIC_SCAN_BATCH_SIZE,
) -> Dict[int, SemanticDocHit]:
    hits: Dict[int, SemanticDocHit] = {}
    cursor = conn.execute(
        """
        SELECT doc_id, chunk_text, embedding
        FROM doc_chunks
        ORDER BY doc_id, chunk_index
        """
    )

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for doc_id, chunk_text, blob in rows:
            similarity = dot_similarity(query_vector, blob_to_vector(blob))
            doc_key = int(doc_id)
            current = hits.get(doc_key)
            if current is None or similarity > current.similarity:
                hits[doc_key] = SemanticDocHit(
                    doc_id=doc_key,
                    similarity=similarity,
                    chunk_text=str(chunk_text),
                )

    return hits


def _keyword_result_line(doc: RankedDoc, q_terms: List[str], debug: bool) -> str:
    if not debug:
        return f"score={doc.total}"

    cu, ct, fn_hits, fn_sub, meta, _fn_pos, rec = doc.debug
    return (
        f"score={doc.total}  "
        f"[content={cu}/{len(q_terms)} hits={ct} | "
        f"fn={fn_hits} sub={fn_sub} meta={meta} rec={rec}]"
    )


def _semantic_result_line(doc: SemanticRankedDoc, _q_terms: List[str], debug: bool) -> str:
    if not debug:
        return f"score={doc.score:.4f}"

    fn_hits, fn_sub, meta, rec = doc.debug
    rerank_raw = f"{doc.rerank_raw:.4f}" if doc.rerank_raw is not None else "-"
    return (
        f"score={doc.score:.4f}  "
        f"[semantic={doc.similarity:.4f} | structure={doc.structure_multiplier:.2f} | "
        f"rerank={rerank_raw} bonus={doc.rerank_bonus:.4f} | "
        f"fn={fn_hits} sub={fn_sub} meta={meta} rec={rec}]"
    )


def _hybrid_result_line(doc: HybridRankedDoc, _q_terms: List[str], debug: bool) -> str:
    if not debug:
        return f"score={doc.total:.4f}"

    rerank_raw = f"{doc.rerank_raw:.4f}" if doc.rerank_raw is not None else "-"
    return (
        f"score={doc.total:.4f}  "
        f"[lex={doc.lexical_norm:.4f} raw={doc.lexical_raw:.1f} | "
        f"sem={doc.semantic_norm:.4f} raw={doc.semantic_raw:.4f} "
        f"(base={doc.semantic_similarity:.4f} structure={doc.semantic_structure_multiplier:.2f}) | "
        f"rerank={rerank_raw} bonus={doc.rerank_bonus:.4f} | "
        f"meta={doc.meta} rec={doc.rec}]"
    )


def _keyword_snippet(doc: RankedDoc, q_terms: List[str]) -> Optional[List[str]]:
    return best_effort_snippet(doc.preview_text, q_terms)


def _semantic_snippet(doc: SemanticRankedDoc, q_terms: List[str]) -> Optional[List[str]]:
    return best_chunk_snippet(doc.chunk_text, q_terms)


def _hybrid_snippet(doc: HybridRankedDoc, q_terms: List[str]) -> Optional[List[str]]:
    snip = best_effort_snippet(doc.preview_text, q_terms)
    if snip is None and doc.chunk_text:
        return best_chunk_snippet(doc.chunk_text, q_terms)
    return snip


def _run_ranked_query(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
    *,
    build_ranked: Callable[[sqlite3.Connection, str], Tuple[List[str], List[Any]]],
    legend_lines: Tuple[str, ...],
    score_line: Callable[[Any, List[str], bool], str],
    snippet_builder: Callable[[Any, List[str]], Optional[List[str]]],
    rerank_ranked: Optional[Callable[[str, List[Any]], List[Any]]] = None,
    require_semantic: bool = False,
) -> None:
    dbp = validate_search_db(root)
    if dbp is None:
        return

    show_snippets = show_snippets or debug

    conn = open_db_connection(dbp, SEARCH_DB_PRAGMAS)
    try:
        if require_semantic and not ensure_semantic_ready(conn, dbp):
            return

        t0 = time.perf_counter()
        q_terms, ranked = build_ranked(conn, query_str)
        if rerank_ranked is not None:
            ranked = rerank_ranked(query_str, ranked)
        query_time = time.perf_counter() - t0

        if debug:
            print(f"Index DB: {dbp}")
            print(f"Query terms: {q_terms}")
            print(f"Candidates: {len(ranked):,} | query_time={query_time:.3f}s")
            print()
            print("Legend:")
            for line in legend_lines:
                print(f"  {line}")
            print()
            print(f"Showing top {min(max_results, len(ranked))}\n")

        for idx, doc in enumerate(ranked[:max_results], start=1):
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(doc.mtime))
            print(f"#{idx}  {score_line(doc, q_terms, debug)}")
            print(f"  {doc.path}")

            if debug:
                print(f"  modified={dt}  size={human_size(doc.size)}")

            if show_snippets:
                snip = snippet_builder(doc, q_terms)
                if snip:
                    print("  snippet:")
                    for line in snip:
                        print(f"    {line}")
            print()

        if debug:
            total_time = time.perf_counter() - t0
            print(f"total_time={total_time:.3f}s")
    finally:
        conn.close()


def query_keyword(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
) -> None:
    _run_ranked_query(
        root,
        query_str,
        max_results=max_results,
        debug=debug,
        show_snippets=show_snippets,
        build_ranked=build_keyword_ranked_docs,
        legend_lines=(
            "content=a/b  -> unique query terms matched / total query terms",
            "hits         -> total term occurrences in document",
            "fn           -> filename token matches",
            "sub          -> full query substring in filename (0/1)",
            "meta         -> date/weekday intent bonus",
            "rec          -> recency bonus (3<1d, 2<7d, 1<30d, 0>=30d)",
        ),
        score_line=_keyword_result_line,
        snippet_builder=_keyword_snippet,
    )


def query_semantic(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
) -> None:
    _run_ranked_query(
        root,
        query_str,
        max_results=max_results,
        debug=debug,
        show_snippets=show_snippets,
        build_ranked=build_semantic_ranked_docs,
        legend_lines=(
            "semantic     -> best chunk cosine similarity for the document",
            "structure    -> confidence multiplier based on chunk structure",
            f"rerank       -> normalized cross-encoder bonus over top {SEMANTIC_RERANK_TOP_K}",
            "fn           -> filename token matches",
            "sub          -> full query substring in filename (0/1)",
            "meta         -> date/weekday intent bonus",
            "rec          -> recency bonus (3<1d, 2<7d, 1<30d, 0>=30d)",
        ),
        score_line=_semantic_result_line,
        snippet_builder=_semantic_snippet,
        rerank_ranked=rerank_top_semantic_docs,
        require_semantic=True,
    )


def query(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
    semantic: bool = False,
    lexical: bool = False,
) -> None:
    if semantic and lexical:
        raise ValueError("Cannot enable both semantic-only and lexical-only search modes.")

    try:
        if semantic:
            query_semantic(
                root,
                query_str,
                max_results=max_results,
                debug=debug,
                show_snippets=show_snippets,
            )
            return

        if lexical:
            query_keyword(
                root,
                query_str,
                max_results=max_results,
                debug=debug,
                show_snippets=show_snippets,
            )
            return

        query_combined(
            root,
            query_str,
            max_results=max_results,
            debug=debug,
            show_snippets=show_snippets,
        )
    except SemanticSetupError as exc:
        print(str(exc))


def query_combined(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
) -> None:
    _run_ranked_query(
        root,
        query_str,
        max_results=max_results,
        debug=debug,
        show_snippets=show_snippets,
        build_ranked=build_hybrid_ranked_docs,
        legend_lines=(
            "lex          -> normalized keyword/content score (0..1)",
            "sem          -> normalized best chunk cosine similarity (0..1)",
            f"rerank       -> normalized cross-encoder bonus over top {SEMANTIC_RERANK_TOP_K}",
            "meta         -> date/weekday intent bonus (0..3)",
            "rec          -> recency bonus (3<1d, 2<7d, 1<30d, 0>=30d)",
        ),
        score_line=_hybrid_result_line,
        snippet_builder=_hybrid_snippet,
        rerank_ranked=rerank_top_hybrid_docs,
        require_semantic=True,
    )


def prepare_interactive_session(root: Path, *, lexical: bool = False) -> bool:
    dbp = validate_search_db(root)
    if dbp is None:
        return False

    if lexical:
        return True

    conn = open_db_connection(dbp, SEARCH_DB_PRAGMAS)
    try:
        if not ensure_semantic_ready(conn, dbp):
            return False
    finally:
        conn.close()

    try:
        get_reranker()
    except SemanticSetupError as exc:
        print(str(exc))
        return False

    return True


def interactive_session(
    root: Path,
    *,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
    semantic: bool = False,
    lexical: bool = False,
) -> None:
    if semantic and lexical:
        raise ValueError("Cannot enable both semantic-only and lexical-only search modes.")

    root = root.expanduser().resolve()
    if not prepare_interactive_session(root, lexical=lexical):
        return

    mode = "semantic" if semantic else "lexical" if lexical else "hybrid"
    print(f"Interactive search ready: {root}")
    print(f"Mode: {mode} | max_results={max_results}")
    print("Enter a query. Commands: :help, :quit")

    while True:
        try:
            raw = input("pse> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not raw:
            continue
        if raw in {":q", ":quit", ":exit"}:
            break
        if raw == ":help":
            print("Enter a query to search the current root.")
            print("Commands: :help, :quit")
            continue

        query(
            root,
            raw,
            max_results=max_results,
            debug=debug,
            show_snippets=show_snippets,
            semantic=semantic,
            lexical=lexical,
        )


def main() -> None:
    import sys

    args = sys.argv[1:]
    if any(arg in {"-h", "--help"} for arg in args):
        print(SEARCH_HELP_TEXT)
        raise SystemExit(0)

    known_flags = {"--debug", "--snippets", "--semantic", "--lexical", "--interactive"}
    unknown_flags = [arg for arg in args if arg.startswith("-") and arg not in known_flags]
    if unknown_flags:
        print(f"Unknown option: {unknown_flags[0]}")
        print()
        print(SEARCH_HELP_TEXT)
        raise SystemExit(1)

    debug = False
    show_snippets = False
    semantic = False
    lexical = False
    interactive = False
    if "--debug" in args:
        debug = True
        args = [a for a in args if a != "--debug"]
    if "--snippets" in args:
        show_snippets = True
        args = [a for a in args if a != "--snippets"]
    if "--semantic" in args:
        semantic = True
        args = [a for a in args if a != "--semantic"]
    if "--lexical" in args:
        lexical = True
        args = [a for a in args if a != "--lexical"]
    if "--interactive" in args:
        interactive = True
        args = [a for a in args if a != "--interactive"]

    if semantic and lexical:
        print("Cannot use --semantic and --lexical together.")
        print()
        print(SEARCH_HELP_TEXT)
        raise SystemExit(1)

    if interactive:
        if len(args) < 1:
            print(SEARCH_HELP_TEXT)
            raise SystemExit(1)
    elif len(args) < 2:
        print(SEARCH_HELP_TEXT)
        raise SystemExit(1)

    root = Path(args[0])
    if interactive:
        if len(args) > 2 or (len(args) == 2 and not args[1].isdigit()):
            print("Interactive mode accepts <root_folder> and optional [max_results].")
            print()
            print(SEARCH_HELP_TEXT)
            raise SystemExit(1)

        max_results = int(args[1]) if len(args) == 2 else 20
        interactive_session(
            root,
            max_results=max_results,
            debug=debug,
            show_snippets=show_snippets,
            semantic=semantic,
            lexical=lexical,
        )
        return

    # Remaining args: query tokens + maybe max_results
    if len(args) >= 3 and args[-1].isdigit():
        max_results = int(args[-1])
        query_str = " ".join(args[1:-1])
    else:
        max_results = 20
        query_str = " ".join(args[1:])

    query(
        root,
        query_str,
        max_results=max_results,
        debug=debug,
        show_snippets=show_snippets,
        semantic=semantic,
        lexical=lexical,
    )


if __name__ == "__main__":
    main()
