from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
import re
import sqlite3
from typing import Optional, Tuple

from pse_common import db_path_for_root, human_size, open_db_connection
from pse_index import IndexRunStats, index_root
from pse_search import SEARCH_MODES, SearchResponse, run_search
from pse_semantic import fetch_semantic_meta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP_K = 10
TOP_K_OPTIONS = (5, 10, 20, 50)


@dataclass(frozen=True)
class IndexSummary:
    root: str
    index_path: str
    root_exists: bool
    index_present: bool
    doc_count: int = 0
    preview_count: int = 0
    chunk_count: int = 0
    semantic_configured: bool = False
    semantic_model_name: Optional[str] = None
    last_indexed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class SearchResultView:
    rank: int
    doc_id: int
    name: str
    path: str
    extension_label: str
    score_display: str
    modified_display: str
    size_display: str
    snippet_html_lines: Tuple[str, ...]
    preview_excerpt: Optional[str]
    detail_preview_text: Optional[str]
    detail_chunk_text: Optional[str]
    metadata_badges: Tuple[str, ...]
    debug_items: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class SearchView:
    root: str
    query: str
    mode: str
    top_k: int
    submitted: bool
    error: Optional[str]
    index_path: Optional[str]
    query_terms: Tuple[str, ...]
    candidate_count: int
    result_count: int
    query_time_display: Optional[str]
    results: Tuple[SearchResultView, ...]


@dataclass(frozen=True)
class IndexRunView:
    root: str
    success: bool
    message: str
    stats: Optional[IndexRunStats]
    summary: IndexSummary


def get_default_root() -> Path:
    files_root = PROJECT_ROOT / "Files"
    if files_root.exists() and files_root.is_dir():
        return files_root
    return PROJECT_ROOT


def _normalize_root(root: str | Path | None) -> Path:
    if root is None:
        return get_default_root().expanduser().resolve()
    text = str(root).strip()
    if not text:
        return get_default_root().expanduser().resolve()
    return Path(text).expanduser().resolve()


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y %I:%M %p")


def _format_indexed_timestamp(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return _format_timestamp(float(ts))


def _preview_excerpt(value: Optional[str], limit: int = 220) -> Optional[str]:
    if not value:
        return None
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _detail_text(value: Optional[str], limit: int = 1800) -> Optional[str]:
    if not value:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _snippet_line_to_html(line: str) -> str:
    escaped = escape(line)
    return re.sub(r"\[(.+?)\]", r"<mark>\1</mark>", escaped)


def _extension_label(suffix: str) -> str:
    trimmed = suffix.lstrip(".").upper()
    return trimmed or "FILE"


def _debug_items(mode: str, debug: dict[str, object]) -> Tuple[Tuple[str, str], ...]:
    rows: list[Tuple[str, str]] = []

    def add(label: str, value: str) -> None:
        if value:
            rows.append((label, value))

    if mode == "lexical":
        add("Content match", f"{debug['content_unique']}/{debug['query_term_count']} query terms")
        add("Content hits", str(debug["content_hits"]))
        add("Filename hits", str(debug["filename_hits"]))
        add("Filename substring", "Yes" if debug["filename_substring"] else "No")
        add("Metadata bonus", str(debug["metadata_bonus"]))
        add("Recency bonus", str(debug["recency_bonus"]))
    elif mode == "semantic":
        add("Similarity", f"{float(debug['semantic_similarity']):.4f}")
        add("Structure multiplier", f"{float(debug['structure_multiplier']):.2f}")
        rerank_raw = debug.get("rerank_raw")
        if rerank_raw is not None:
            add("Rerank raw", f"{float(rerank_raw):.4f}")
        add("Rerank bonus", f"{float(debug['rerank_bonus']):.4f}")
        add("Filename hits", str(debug["filename_hits"]))
        add("Filename substring", "Yes" if debug["filename_substring"] else "No")
        add("Metadata bonus", str(debug["metadata_bonus"]))
        add("Recency bonus", str(debug["recency_bonus"]))
    else:
        add("Lexical raw", f"{float(debug['lexical_raw']):.1f}")
        add("Lexical normalized", f"{float(debug['lexical_norm']):.4f}")
        add("Semantic raw", f"{float(debug['semantic_raw']):.4f}")
        add("Semantic normalized", f"{float(debug['semantic_norm']):.4f}")
        add("Semantic similarity", f"{float(debug['semantic_similarity']):.4f}")
        add("Semantic structure", f"{float(debug['semantic_structure_multiplier']):.2f}")
        rerank_raw = debug.get("rerank_raw")
        if rerank_raw is not None:
            add("Rerank raw", f"{float(rerank_raw):.4f}")
        add("Rerank bonus", f"{float(debug['rerank_bonus']):.4f}")
        add("Filename hits", str(debug["filename_hits"]))
        add("Filename substring", "Yes" if debug["filename_substring"] else "No")
        add("Metadata bonus", str(debug["metadata_bonus"]))
        add("Recency bonus", str(debug["recency_bonus"]))

    return tuple(rows)


def _metadata_badges(mode: str, extension_label: str, debug: dict[str, object], has_chunk: bool) -> Tuple[str, ...]:
    badges = [extension_label]
    if has_chunk and mode in {"semantic", "hybrid"}:
        badges.append("Best chunk")
    if int(debug.get("metadata_bonus", 0)) > 0:
        badges.append("Date match")
    if int(debug.get("recency_bonus", 0)) > 0:
        badges.append("Recent")
    return tuple(badges)


def _search_result_view(mode: str, result) -> SearchResultView:
    extension_label = _extension_label(result.extension)
    return SearchResultView(
        rank=result.rank,
        doc_id=result.doc_id,
        name=result.name,
        path=result.path,
        extension_label=extension_label,
        score_display=result.score_display,
        modified_display=_format_timestamp(result.mtime),
        size_display=human_size(result.size),
        snippet_html_lines=tuple(_snippet_line_to_html(line) for line in result.snippet_lines),
        preview_excerpt=_preview_excerpt(result.preview_text),
        detail_preview_text=_detail_text(result.preview_text),
        detail_chunk_text=_detail_text(result.chunk_text, limit=1200),
        metadata_badges=_metadata_badges(mode, extension_label, result.debug, bool(result.chunk_text)),
        debug_items=_debug_items(mode, result.debug),
    )


def get_index_summary(root: str | Path | None) -> IndexSummary:
    root_path = _normalize_root(root)
    index_path = db_path_for_root(root_path)
    if not root_path.exists() or not root_path.is_dir():
        return IndexSummary(
            root=str(root_path),
            index_path=str(index_path),
            root_exists=False,
            index_present=False,
            error=f"Folder not found: {root_path}",
        )

    if not index_path.exists():
        return IndexSummary(
            root=str(root_path),
            index_path=str(index_path),
            root_exists=True,
            index_present=False,
        )

    conn = open_db_connection(index_path)
    try:
        doc_count = int(conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
        preview_count = int(conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0])
        chunk_count = int(conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0])
        last_indexed_raw = conn.execute("SELECT MAX(indexed_at) FROM docs").fetchone()[0]
        semantic_meta = fetch_semantic_meta(conn)
        return IndexSummary(
            root=str(root_path),
            index_path=str(index_path),
            root_exists=True,
            index_present=True,
            doc_count=doc_count,
            preview_count=preview_count,
            chunk_count=chunk_count,
            semantic_configured=semantic_meta is not None,
            semantic_model_name=None if semantic_meta is None else semantic_meta.model_name,
            last_indexed_at=_format_indexed_timestamp(last_indexed_raw),
        )
    except sqlite3.DatabaseError as exc:
        return IndexSummary(
            root=str(root_path),
            index_path=str(index_path),
            root_exists=True,
            index_present=True,
            error=f"Unable to read index summary: {exc}",
        )
    finally:
        conn.close()


def search_documents(root: str | Path | None, query: str, mode: str, top_k: int) -> SearchView:
    root_path = _normalize_root(root)
    submitted = bool((query or "").strip())
    try:
        response: SearchResponse = run_search(root_path, query or "", mode=mode, max_results=top_k)
        normalized_mode = response.mode
    except ValueError as exc:
        normalized_mode = mode if mode in SEARCH_MODES else "hybrid"
        return SearchView(
            root=str(root_path),
            query=query or "",
            mode=normalized_mode,
            top_k=top_k,
            submitted=submitted,
            error=str(exc),
            index_path=str(db_path_for_root(root_path)),
            query_terms=(),
            candidate_count=0,
            result_count=0,
            query_time_display=None,
            results=(),
        )

    return SearchView(
        root=response.root,
        query=response.query,
        mode=normalized_mode,
        top_k=response.max_results,
        submitted=submitted,
        error=response.error,
        index_path=response.index_path,
        query_terms=response.query_terms,
        candidate_count=response.candidate_count,
        result_count=len(response.results),
        query_time_display=None if response.query_time_seconds is None else f"{response.query_time_seconds:.3f}s",
        results=tuple(_search_result_view(normalized_mode, result) for result in response.results),
    )


def run_index(root: str | Path | None) -> IndexRunView:
    root_path = _normalize_root(root)
    try:
        stats = index_root(root_path, quiet=True)
        summary = get_index_summary(root_path)
        return IndexRunView(
            root=str(root_path),
            success=True,
            message=f"Index updated for {root_path}",
            stats=stats,
            summary=summary,
        )
    except Exception as exc:
        summary = get_index_summary(root_path)
        return IndexRunView(
            root=str(root_path),
            success=False,
            message=str(exc),
            stats=None,
            summary=summary,
        )
