import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import docx

from pse_common import INDEXABLE_EXTS, db_path_for_root, iter_files, tokenize


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


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
    conn.commit()


def extract_text(path: str, ext: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """
    Extract text from supported files.
    text files: read bytes with cap
    docx: extract paragraphs using python-docx
    """
    if ext not in INDEXABLE_EXTS:
        return None

    if ext == ".docx":
        try:
            from docx import Document  # python-docx
        except Exception:
            # If python-docx isn't available, skip docx gracefully
            return None

        try:
            doc = Document(path)
        except Exception:
            return None

        # Join paragraph text, keep it bounded so huge docs do not explode indexing time
        parts: List[str] = []
        total_chars = 0
        char_cap = 2_000_000  # roughly similar scale to max_bytes
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if not t:
                continue
            parts.append(t)
            total_chars += len(t) + 1
            if total_chars >= char_cap:
                break
        return "\n".join(parts)

    # default: treat as plain text-like
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except (PermissionError, FileNotFoundError, OSError):
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")


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

    conn = connect_db(dbp)
    try:
        init_schema(conn)

        t0 = time.time()
        scanned = 0
        indexed = 0
        skipped = 0
        failed = 0
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

            text = extract_text(e.path, e.ext)
            if text is None:
                failed += 1
                continue

            counts = term_counts_from_text(text)
            doc_id = upsert_doc(conn, e.path, e.ext, e.size, e.mtime)
            replace_terms_for_doc(conn, doc_id, counts)
            indexed += 1

            if indexed % 200 == 0:
                conn.commit()

        conn.commit()
        removed = prune_deleted_files(conn)
        t1 = time.time()

        print(f"Index DB: {dbp}")
        print(f"Scanned: {scanned:,} files")
        print(f"Indexed: {indexed:,} | Skipped: {skipped:,} | Failed: {failed:,} | Removed: {removed:,}")
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