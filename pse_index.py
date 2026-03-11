import json
import sqlite3
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pse_common import (
    DEFAULT_ALLOWED_EXTS,
    DEFAULT_IGNORE_FOLDERS,
    DEFAULT_MAX_BYTES,
    INDEXABLE_EXTS,
    INDEX_CONFIG_NAME,
    INDEX_LOG_DIR_NAME,
    PREVIEW_CHAR_CAP,
    db_path_for_root,
    index_config_path_for_root,
    index_jsonl_log_path_for_root,
    index_text_log_path_for_root,
    iter_files,
    open_db_connection,
    reset_index_db,
    tokenize,
)
from pse_extract import extract_text, supports_partial_byte_cap
from pse_semantic import (
    SemanticIndexMeta,
    build_index_meta,
    chunk_text,
    encode_texts,
    fetch_semantic_meta,
    semantic_meta_row,
    vector_to_blob,
)


# Shared extractor keeps indexing and snippet behavior aligned.
INDEX_CHAR_CAP = 2_000_000
INDEX_DB_PRAGMAS = (
    "journal_mode=WAL",
    "synchronous=NORMAL",
    "temp_store=MEMORY",
    "foreign_keys=ON",
)
RUN_SAMPLE_LIMIT = 10
SUCCESS_STATUS = "success"
ABORTED_STATUS = "aborted"
CONFIG_KEYS = {"allowed_extensions", "max_bytes", "ignore_folders"}
INDEX_HELP_TEXT = """Usage: python pse_index.py index <root_folder> [options]

Build or refresh the SQLite index for a root folder.

Arguments:
  index           Command that runs indexing.
  <root_folder>   Folder to scan and index.

Options:
  -h, --help      Show this help message and exit.

Behavior:
  - Creates or updates .pse_index.sqlite3 in the root folder.
  - Reads pse_index.toml from the root folder when present.
  - Writes run logs under .pse_index_logs.
"""


@dataclass(frozen=True)
class IndexConfig:
    allowed_extensions: frozenset[str]
    max_bytes: int
    ignore_folders: frozenset[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "allowed_extensions": sorted(self.allowed_extensions),
            "max_bytes": self.max_bytes,
            "ignore_folders": sorted(self.ignore_folders),
        }


@dataclass
class IndexRunStats:
    root: str
    db_path: str
    config: Optional[IndexConfig]
    started_at: str
    status: str = SUCCESS_STATUS
    finished_at: str = ""
    scanned: int = 0
    indexed: int = 0
    empty: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    duration_seconds: float = 0.0
    skipped_unchanged: int = 0
    skipped_not_indexable: int = 0
    skipped_size: int = 0
    not_indexable_samples: List[str] = field(default_factory=list)
    size_skipped_samples: List[str] = field(default_factory=list)
    failed_samples: List[str] = field(default_factory=list)
    removed_samples: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def finalize(self, status: str, finished_at: str, duration_seconds: float, error: Optional[str] = None) -> None:
        self.status = status
        self.finished_at = finished_at
        self.duration_seconds = round(max(0.0, duration_seconds), 3)
        self.skipped = (
            self.skipped_unchanged +
            self.skipped_not_indexable +
            self.skipped_size
        )
        self.error = error

    def to_log_record(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "root": self.root,
            "db_path": self.db_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "config": self.config.to_dict() if self.config is not None else None,
            "scanned": self.scanned,
            "indexed": self.indexed,
            "empty": self.empty,
            "skipped": self.skipped,
            "failed": self.failed,
            "removed": self.removed,
            "skip_breakdown": {
                "unchanged": self.skipped_unchanged,
                "extension": self.skipped_not_indexable,
                "size": self.skipped_size,
            },
            "samples": {
                "not_indexable": list(self.not_indexable_samples),
                "size_skipped": list(self.size_skipped_samples),
                "failed": list(self.failed_samples),
                "removed": list(self.removed_samples),
            },
            "error": self.error,
        }


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS docs (
            doc_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            ext  TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            indexed_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_previews (
            doc_id INTEGER PRIMARY KEY,
            preview_text TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS terms (
            term   TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            count  INTEGER NOT NULL,
            PRIMARY KEY(term, doc_id),
            FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS semantic_meta (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            backend TEXT NOT NULL,
            model_name TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            chunk_chars INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_chunks (
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY(doc_id, chunk_index),
            FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
        CREATE INDEX IF NOT EXISTS idx_terms_doc  ON terms(doc_id);
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc ON doc_chunks(doc_id);
        """
    )
    conn.commit()


def term_counts_from_text(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    return counts


def get_doc_state(conn: sqlite3.Connection, path: str) -> Optional[Tuple[int, int, float]]:
    row = conn.execute(
        "SELECT doc_id, size, mtime FROM docs WHERE path = ?",
        (path,),
    ).fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1]), float(row[2])


def upsert_doc(conn: sqlite3.Connection, path: str, ext: str, size: int, mtime: float) -> int:
    now = time.time()
    conn.execute(
        """
        INSERT INTO docs(path, ext, size, mtime, indexed_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            ext=excluded.ext,
            size=excluded.size,
            mtime=excluded.mtime,
            indexed_at=excluded.indexed_at
        """,
        (path, ext, size, mtime, now),
    )
    doc_id = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (path,)).fetchone()[0]
    return int(doc_id)


def replace_terms_for_doc(conn: sqlite3.Connection, doc_id: int, counts: Dict[str, int]) -> None:
    conn.execute("DELETE FROM terms WHERE doc_id = ?", (doc_id,))
    if counts:
        conn.executemany(
            "INSERT INTO terms(term, doc_id, count) VALUES(?, ?, ?)",
            [(term, doc_id, cnt) for term, cnt in counts.items()],
        )


def replace_preview_for_doc(conn: sqlite3.Connection, doc_id: int, preview_text: Optional[str]) -> None:
    if preview_text and preview_text.strip():
        conn.execute(
            """
            INSERT INTO doc_previews(doc_id, preview_text)
            VALUES(?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET preview_text=excluded.preview_text
            """,
            (doc_id, preview_text),
        )
        return

    conn.execute("DELETE FROM doc_previews WHERE doc_id = ?", (doc_id,))


def replace_chunks_for_doc(
    conn: sqlite3.Connection,
    doc_id: int,
    text: Optional[str],
) -> None:
    conn.execute("DELETE FROM doc_chunks WHERE doc_id = ?", (doc_id,))
    if not text or not text.strip():
        return

    chunks = chunk_text(text)
    if not chunks:
        return

    embeddings = encode_texts([chunk.text for chunk in chunks])
    if len(chunks) != len(embeddings):
        raise RuntimeError("Semantic chunk embedding count did not match chunk count")

    conn.executemany(
        """
        INSERT INTO doc_chunks(doc_id, chunk_index, start_char, end_char, chunk_text, embedding)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        [
            (
                doc_id,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                chunk.text,
                vector_to_blob(vector),
            )
            for chunk, vector in zip(chunks, embeddings)
        ],
    )


def clear_doc_content(conn: sqlite3.Connection, doc_id: int) -> None:
    replace_terms_for_doc(conn, doc_id, {})
    replace_preview_for_doc(conn, doc_id, None)
    replace_chunks_for_doc(conn, doc_id, None)


def replace_semantic_meta(conn: sqlite3.Connection, meta: SemanticIndexMeta) -> None:
    conn.execute(
        """
        INSERT INTO semantic_meta(
            singleton,
            backend,
            model_name,
            embedding_dim,
            chunk_chars,
            chunk_overlap,
            updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
            backend=excluded.backend,
            model_name=excluded.model_name,
            embedding_dim=excluded.embedding_dim,
            chunk_chars=excluded.chunk_chars,
            chunk_overlap=excluded.chunk_overlap,
            updated_at=excluded.updated_at
        """,
        semantic_meta_row(meta),
    )


def _timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_sample(samples: List[str], value: str) -> None:
    if len(samples) < RUN_SAMPLE_LIMIT:
        samples.append(value)


def _normalize_extension(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("allowed_extensions entries must be strings")

    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("allowed_extensions entries must not be empty")
    if not normalized.startswith("."):
        normalized = "." + normalized
    if normalized not in INDEXABLE_EXTS:
        raise ValueError(f"Unsupported extension in {INDEX_CONFIG_NAME}: {value}")
    return normalized


def _normalize_folder_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("ignore_folders entries must be strings")

    normalized = value.strip()
    if not normalized:
        raise ValueError("ignore_folders entries must not be empty")
    if normalized in {".", ".."}:
        raise ValueError("ignore_folders entries must be directory names")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("ignore_folders entries must be directory basenames")
    return normalized


def _normalize_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _normalize_max_bytes(raw_value: object) -> int:
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise ValueError("max_bytes must be an integer")
    if raw_value <= 0:
        raise ValueError("max_bytes must be greater than zero")
    return raw_value


def load_index_config(root: Path) -> IndexConfig:
    config_path = index_config_path_for_root(root)
    allowed_extensions = frozenset(DEFAULT_ALLOWED_EXTS)
    max_bytes = DEFAULT_MAX_BYTES
    ignore_folders = frozenset(DEFAULT_IGNORE_FOLDERS)

    if not config_path.exists():
        return IndexConfig(
            allowed_extensions=allowed_extensions,
            max_bytes=max_bytes,
            ignore_folders=ignore_folders,
        )

    try:
        with config_path.open("rb") as fh:
            raw_config = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid {INDEX_CONFIG_NAME}: {exc}") from exc

    unknown_top_level = sorted(set(raw_config) - {"index"})
    if unknown_top_level:
        raise ValueError(
            f"Unknown top-level keys in {INDEX_CONFIG_NAME}: {', '.join(unknown_top_level)}"
        )

    raw_index = raw_config.get("index")
    if not isinstance(raw_index, dict):
        raise ValueError(f"{INDEX_CONFIG_NAME} must define an [index] table")

    unknown_keys = sorted(set(raw_index) - CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown [index] keys in {INDEX_CONFIG_NAME}: {', '.join(unknown_keys)}")

    if "allowed_extensions" in raw_index:
        raw_list = _normalize_list(raw_index["allowed_extensions"], "allowed_extensions")
        allowed_extensions = frozenset(_normalize_extension(value) for value in raw_list)
    if "max_bytes" in raw_index:
        max_bytes = _normalize_max_bytes(raw_index["max_bytes"])
    if "ignore_folders" in raw_index:
        raw_list = _normalize_list(raw_index["ignore_folders"], "ignore_folders")
        ignore_folders = frozenset(_normalize_folder_name(value) for value in raw_list)

    return IndexConfig(
        allowed_extensions=allowed_extensions,
        max_bytes=max_bytes,
        ignore_folders=ignore_folders,
    )


def prune_out_of_scope_docs(
    conn: sqlite3.Connection,
    seen_paths: set[str],
    removed_samples: List[str],
) -> int:
    rows = conn.execute("SELECT doc_id, path FROM docs").fetchall()
    to_delete = []
    for doc_id, path in rows:
        resolved_path = str(path)
        if resolved_path in seen_paths:
            continue
        to_delete.append((int(doc_id),))
        _append_sample(removed_samples, resolved_path)

    if to_delete:
        conn.executemany("DELETE FROM docs WHERE doc_id = ?", to_delete)
    return len(to_delete)

def _format_text_log_block(stats: IndexRunStats) -> str:
    lines = [
        f"Run started: {stats.started_at}",
        f"Run finished: {stats.finished_at}",
        f"Status: {stats.status}",
        f"Root: {stats.root}",
        f"Index DB: {stats.db_path}",
    ]

    if stats.config is not None:
        config_dict = stats.config.to_dict()
        lines.append(
            "Config: "
            f"allowed_extensions={','.join(config_dict['allowed_extensions'])} "
            f"max_bytes={config_dict['max_bytes']} "
            f"ignore_folders={','.join(config_dict['ignore_folders'])}"
        )

    lines.extend(
        [
            f"Scanned: {stats.scanned}",
            (
                f"Indexed: {stats.indexed} | Empty: {stats.empty} | "
                f"Skipped: {stats.skipped} | Failed: {stats.failed} | Removed: {stats.removed}"
            ),
            (
                "Skip breakdown: "
                f"unchanged={stats.skipped_unchanged} | "
                f"extension={stats.skipped_not_indexable} | "
                f"size={stats.skipped_size}"
            ),
            f"Time: {stats.duration_seconds:.3f}s",
        ]
    )

    if stats.error:
        lines.append(f"Error: {stats.error}")

    if stats.not_indexable_samples:
        lines.append("Extension-skip samples: " + " | ".join(stats.not_indexable_samples))
    if stats.size_skipped_samples:
        lines.append("Size-skip samples: " + " | ".join(stats.size_skipped_samples))
    if stats.failed_samples:
        lines.append("Failed samples: " + " | ".join(stats.failed_samples))
    if stats.removed_samples:
        lines.append("Removed samples: " + " | ".join(stats.removed_samples))

    return "\n".join(lines) + "\n\n"


def append_run_logs(root: Path, stats: IndexRunStats) -> None:
    jsonl_path = index_jsonl_log_path_for_root(root)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stats.to_log_record(), ensure_ascii=True, sort_keys=True) + "\n")

    text_path = index_text_log_path_for_root(root)
    with text_path.open("a", encoding="utf-8") as fh:
        fh.write(_format_text_log_block(stats))


def print_run_summary(stats: IndexRunStats) -> None:
    print(f"Index DB: {stats.db_path}")
    print(f"Scanned: {stats.scanned:,} files")
    print(
        f"Indexed: {stats.indexed:,} | Empty: {stats.empty:,} | "
        f"Skipped: {stats.skipped:,} | Failed: {stats.failed:,} | Removed: {stats.removed:,}"
    )
    print(
        "Skip breakdown: "
        f"unchanged={stats.skipped_unchanged:,} | "
        f"extension={stats.skipped_not_indexable:,} | "
        f"size={stats.skipped_size:,}"
    )
    print(f"Time: {stats.duration_seconds:.2f}s")
    for title, samples in (
        ("Examples skipped by extension:", stats.not_indexable_samples),
        ("Examples skipped by size:", stats.size_skipped_samples),
        ("Examples failed to extract:", stats.failed_samples),
        ("Examples pruned from index:", stats.removed_samples),
    ):
        if not samples:
            continue
        print(title)
        for sample in samples:
            print(f"  - {sample}")


def _build_aborted_stats(
    root: Path,
    db_path: Path,
    started_at: str,
    config: Optional[IndexConfig],
    started_perf: float,
    exc: Exception,
) -> IndexRunStats:
    stats = IndexRunStats(
        root=str(root),
        db_path=str(db_path),
        config=config,
        started_at=started_at,
    )
    stats.finalize(
        status=ABORTED_STATUS,
        finished_at=_timestamp_now(),
        duration_seconds=time.perf_counter() - started_perf,
        error=str(exc),
    )
    return stats


def index_root(root: Path) -> IndexRunStats:
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Index root does not exist or is not a directory: {root}")

    dbp = db_path_for_root(root)
    started_at = _timestamp_now()
    started_perf = time.perf_counter()
    config: Optional[IndexConfig] = None
    conn: Optional[sqlite3.Connection] = None
    semantic_meta: Optional[SemanticIndexMeta] = None

    try:
        config = load_index_config(root)
        semantic_meta = build_index_meta()

        if dbp.exists():
            prior_conn = open_db_connection(dbp, ("foreign_keys=ON",))
            try:
                existing_meta = fetch_semantic_meta(prior_conn)
            finally:
                prior_conn.close()

            if existing_meta != semantic_meta:
                print(f"Index DB: {dbp}")
                print("Semantic configuration changed; rebuilding index.")
                reset_index_db(dbp)

        stats = IndexRunStats(
            root=str(root),
            db_path=str(dbp),
            config=config,
            started_at=started_at,
        )

        conn = open_db_connection(dbp, INDEX_DB_PRAGMAS)
        init_schema(conn)
        replace_semantic_meta(conn, semantic_meta)

        seen_paths: set[str] = set()

        for entry in iter_files(
            [root],
            ignore_folders=config.ignore_folders,
            internal_ignore_dirs={INDEX_LOG_DIR_NAME},
        ):
            stats.scanned += 1

            if entry.ext not in config.allowed_extensions:
                stats.skipped_not_indexable += 1
                _append_sample(
                    stats.not_indexable_samples,
                    f"{entry.path} (ext={entry.ext or '<none>'})",
                )
                continue

            seen_paths.add(entry.path)

            prev = get_doc_state(conn, entry.path)
            if prev is not None:
                _doc_id, prev_size, prev_mtime = prev
                if prev_size == entry.size and abs(prev_mtime - entry.mtime) < 1e-6:
                    stats.skipped_unchanged += 1
                    continue

            doc_id = upsert_doc(conn, entry.path, entry.ext, entry.size, entry.mtime)

            if entry.size > config.max_bytes and not supports_partial_byte_cap(entry.ext):
                clear_doc_content(conn, doc_id)
                stats.skipped_size += 1
                _append_sample(
                    stats.size_skipped_samples,
                    f"{entry.path} ({entry.size} bytes)",
                )
                continue

            text = extract_text(
                entry.path,
                entry.ext,
                INDEX_CHAR_CAP,
                max_bytes=config.max_bytes,
            )
            if text is None:
                clear_doc_content(conn, doc_id)
                stats.failed += 1
                _append_sample(stats.failed_samples, entry.path)
            elif not text.strip():
                clear_doc_content(conn, doc_id)
                stats.empty += 1
            else:
                counts = term_counts_from_text(text)
                replace_terms_for_doc(conn, doc_id, counts)
                replace_preview_for_doc(conn, doc_id, text[:PREVIEW_CHAR_CAP])
                replace_chunks_for_doc(conn, doc_id, text)
                stats.indexed += 1

        stats.removed = prune_out_of_scope_docs(conn, seen_paths, stats.removed_samples)
        conn.commit()

        stats.finalize(
            status=SUCCESS_STATUS,
            finished_at=_timestamp_now(),
            duration_seconds=time.perf_counter() - started_perf,
        )
        print_run_summary(stats)
        append_run_logs(root, stats)
        return stats
    except Exception as exc:
        if conn is not None:
            conn.rollback()

        aborted_stats = _build_aborted_stats(root, dbp, started_at, config, started_perf, exc)
        append_run_logs(root, aborted_stats)
        raise
    finally:
        if conn is not None:
            conn.close()

def main() -> None:
    import sys

    args = sys.argv[1:]
    if any(arg in {"-h", "--help"} for arg in args):
        print(INDEX_HELP_TEXT)
        raise SystemExit(0)

    unknown_flags = [arg for arg in args if arg.startswith("-")]
    if unknown_flags:
        print(f"Unknown option: {unknown_flags[0]}")
        print()
        print(INDEX_HELP_TEXT)
        raise SystemExit(1)

    if len(args) < 2 or args[0].lower() != "index":
        print(INDEX_HELP_TEXT)
        raise SystemExit(1)

    root = Path(args[1])
    try:
        index_root(root)
    except Exception as exc:
        print(f"Indexing failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
