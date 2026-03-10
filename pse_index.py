import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pse_common import (
    INDEXABLE_EXTS,
    INDEX_SCHEMA_VERSION,
    PREVIEW_CHAR_CAP,
    db_path_for_root,
    get_index_schema_version,
    iter_files,
    open_db_connection,
    reset_index_db,
    set_index_schema_version,
    tokenize,
)
from pse_extract import extract_text


# Shared extractor keeps indexing and snippet behavior aligned.
INDEX_CHAR_CAP = 2_000_000
INDEX_DB_PRAGMAS = (
    "journal_mode=WAL",
    "synchronous=NORMAL",
    "temp_store=MEMORY",
    "foreign_keys=ON",
)


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

        CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
        CREATE INDEX IF NOT EXISTS idx_terms_doc  ON terms(doc_id);
        """
    )
    set_index_schema_version(conn)
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


def clear_doc_content(conn: sqlite3.Connection, doc_id: int) -> None:
    replace_terms_for_doc(conn, doc_id, {})
    replace_preview_for_doc(conn, doc_id, None)


def prune_deleted_files(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT doc_id, path FROM docs").fetchall()
    to_delete = [(int(doc_id),) for doc_id, path in rows if not Path(str(path)).exists()]
    if to_delete:
        conn.executemany("DELETE FROM docs WHERE doc_id = ?", to_delete)
        conn.commit()
    return len(to_delete)


def index_root(root: Path) -> None:
    root = root.expanduser().resolve()
    dbp = db_path_for_root(root)

    existing_version = get_index_schema_version(dbp)
    if dbp.exists() and existing_version > INDEX_SCHEMA_VERSION:
        print(f"Index DB: {dbp}")
        print(
            f"Index schema version {existing_version} is newer than this code supports "
            f"({INDEX_SCHEMA_VERSION})."
        )
        return

    if dbp.exists() and existing_version < INDEX_SCHEMA_VERSION:
        print(f"Index DB: {dbp}")
        print(
            f"Schema upgrade detected ({existing_version} -> {INDEX_SCHEMA_VERSION}); "
            "rebuilding index."
        )
        reset_index_db(dbp)

    conn = open_db_connection(dbp, INDEX_DB_PRAGMAS)
    try:
        init_schema(conn)

        t0 = time.time()
        scanned = 0
        indexed = 0
        empty = 0
        skipped = 0
        failed = 0
        changed = 0
        not_indexable = 0
        not_indexable_samples: List[str] = []

        for e in iter_files([root]):
            scanned += 1

            if e.ext not in INDEXABLE_EXTS:
                not_indexable += 1
                if len(not_indexable_samples) < 10:
                    not_indexable_samples.append(f"{e.path} (ext={e.ext or '<none>'})")
                continue

            prev = get_doc_state(conn, e.path)
            if prev is not None:
                _doc_id, prev_size, prev_mtime = prev
                if prev_size == e.size and abs(prev_mtime - e.mtime) < 1e-6:
                    skipped += 1
                    continue

            # Keep supported files in docs even if content extraction fails so
            # filename and metadata search still works.
            doc_id = upsert_doc(conn, e.path, e.ext, e.size, e.mtime)
            changed += 1

            text = extract_text(e.path, e.ext, INDEX_CHAR_CAP)
            if text is None:
                # Clear content terms for unreadable files but preserve the doc row.
                clear_doc_content(conn, doc_id)
                failed += 1
            elif not text.strip():
                # Empty or whitespace-only content is valid but contributes no terms.
                clear_doc_content(conn, doc_id)
                empty += 1
            else:
                counts = term_counts_from_text(text)
                replace_terms_for_doc(conn, doc_id, counts)
                replace_preview_for_doc(conn, doc_id, text[:PREVIEW_CHAR_CAP])
                indexed += 1

            if changed % 200 == 0:
                conn.commit()

        conn.commit()
        removed = prune_deleted_files(conn)
        t1 = time.time()

        print(f"Index DB: {dbp}")
        print(f"Scanned: {scanned:,} files")
        print(
            f"Indexed: {indexed:,} | Empty: {empty:,} | "
            f"Skipped: {skipped:,} | Failed: {failed:,} | Removed: {removed:,}"
        )
        print(f"Time: {(t1 - t0):.2f}s")
        print(f"Not indexable (by ext): {not_indexable:,}")
        if not_indexable_samples:
            print("Examples not indexed:")
            for s in not_indexable_samples:
                print(f"  - {s}")
    finally:
        conn.close()


def main() -> None:
    import sys

    if len(sys.argv) < 3 or sys.argv[1].lower() != "index":
        print("Usage: python pse_index.py index <root_folder>")
        raise SystemExit(1)

    root = Path(sys.argv[2])
    index_root(root)


if __name__ == "__main__":
    main()
