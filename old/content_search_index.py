# content_search_index.py
# v1: Local keyword content indexing + querying with SQLite
#
# Supports text-like files: .txt .md .py .js .ts .json .yaml .yml .csv .log
#
# Usage:
#   python content_search_index.py index  "D:\Path\To\Root"
#   python content_search_index.py query  "D:\Path\To\Root"  "tuesday deploy issue"  20
#
# Notes:
# - This is keyword content search only (no embeddings yet).
# - Incremental: skips files if (mtime,size) unchanged.
# - Results are hybrid ranked using:
#     content term counts + your existing filename score + your metadata bonus

import os
import re
import sys
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Data model (matches your style)
# -----------------------------

@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    ext: str
    size: int
    mtime: float


# -----------------------------
# Query intent helpers (copied from your script)
# -----------------------------

WEEKDAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """
    Very simple tokenizer:
    - normalize
    - keep alnum tokens
    - drop very short tokens (except digits)
    """
    s = normalize(s)
    if not s:
        return []
    # Split on anything that is not a-z0-9
    raw = re.split(r"[^a-z0-9]+", s)
    toks = [t for t in raw if t]
    # Drop 1-char alpha tokens, keep 1-char digits
    out = []
    for t in toks:
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)
    return out

def extract_weekday_from_query(query: str) -> Optional[int]:
    q = normalize(query)
    for token, wd in WEEKDAY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            return wd
    return None

def infer_target_date_from_query(query: str) -> Optional[datetime.date]:
    q = normalize(query)
    if not q:
        return None

    today = datetime.now().date()

    for token, wd in WEEKDAY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            delta = (today.weekday() - wd) % 7
            return today - timedelta(days=delta)

    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", q)
    if not m:
        m = re.search(r"\b(\d{1,2})\b", q)

    if m:
        dom = int(m.group(1))
        if 1 <= dom <= 31:
            y, mo = today.year, today.month
            try:
                cand = datetime(y, mo, dom).date()
                if cand <= today:
                    return cand
            except ValueError:
                pass

            if mo == 1:
                y, mo = y - 1, 12
            else:
                mo -= 1

            try:
                return datetime(y, mo, dom).date()
            except ValueError:
                return None

    return None

def metadata_bonus(query: str, mtime: float) -> int:
    """
    Boost if query implies a weekday or day-of-month and file mtime matches.
    - Weekday: match weekday of mtime (any week)
    - Day-of-month: match near inferred target date (most recent dom)
    """
    wd = extract_weekday_from_query(query)
    mdt = datetime.fromtimestamp(mtime).date()

    if wd is not None:
        return 3 if mdt.weekday() == wd else 0

    target = infer_target_date_from_query(query)
    if target is None:
        return 0

    diff = abs((mdt - target).days)
    if diff == 0:
        return 3
    if diff == 1:
        return 2
    if diff == 2:
        return 1
    return 0


# -----------------------------
# File walking
# -----------------------------

SKIP_DIRS = {".git", "node_modules", "__pycache__"}
INDEXABLE_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".log"
}

def iter_files(roots: Iterable[Path]) -> Iterable[FileEntry]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fn in filenames:
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                try:
                    st = p.stat()
                except (PermissionError, FileNotFoundError):
                    continue

                yield FileEntry(
                    path=str(p),
                    name=p.stem,
                    ext=ext,
                    size=st.st_size,
                    mtime=st.st_mtime,
                )


# -----------------------------
# SQLite schema
# -----------------------------

def db_path_for_root(root: Path) -> Path:
    # Keep DB in root folder (easy for now). Adjust later if you prefer a central app-data folder.
    return root / ".pse_index.sqlite3"

def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
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


# -----------------------------
# Content extraction (v1)
# -----------------------------

def extract_text(path: str, ext: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """
    v1: read text-like files only.
    - max_bytes cap prevents giant logs from dominating indexing time.
    """
    if ext not in INDEXABLE_EXTS:
        return None

    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except (PermissionError, FileNotFoundError, OSError):
        return None

    # Best effort decode.
    # utf-8 first, then latin-1 fallback.
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


# -----------------------------
# Indexing logic (incremental)
# -----------------------------

def get_doc_state(conn: sqlite3.Connection, path: str) -> Optional[Tuple[int, int, float]]:
    """
    Returns (doc_id, size, mtime) if known.
    """
    row = conn.execute(
        "SELECT doc_id, size, mtime FROM docs WHERE path = ?",
        (path,),
    ).fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1]), float(row[2])

def upsert_doc(conn: sqlite3.Connection, e: FileEntry) -> int:
    """
    Insert doc if missing, else update. Returns doc_id.
    """
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
        (e.path, e.ext, e.size, e.mtime, now),
    )
    doc_id = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (e.path,)).fetchone()[0]
    return int(doc_id)

def replace_terms_for_doc(conn: sqlite3.Connection, doc_id: int, counts: Dict[str, int]) -> None:
    conn.execute("DELETE FROM terms WHERE doc_id = ?", (doc_id,))
    if not counts:
        return
    conn.executemany(
        "INSERT INTO terms(term, doc_id, count) VALUES(?, ?, ?)",
        [(term, doc_id, cnt) for term, cnt in counts.items()],
    )

def prune_deleted_files(conn: sqlite3.Connection) -> int:
    """
    Remove docs that no longer exist on disk.
    """
    rows = conn.execute("SELECT doc_id, path FROM docs").fetchall()
    to_delete = []
    for doc_id, path in rows:
        if not os.path.exists(path):
            to_delete.append((int(doc_id),))
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
        skipped = 0
        indexed = 0
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
                prev_doc_id, prev_size, prev_mtime = prev
                if prev_size == e.size and abs(prev_mtime - e.mtime) < 1e-6:
                    skipped += 1
                    continue

            text = extract_text(e.path, e.ext)
            if text is None:
                failed += 1
                continue

            counts = term_counts_from_text(text)
            doc_id = upsert_doc(conn, e)
            replace_terms_for_doc(conn, doc_id, counts)
            indexed += 1

            if indexed % 200 == 0:
                conn.commit()

        conn.commit()
        removed = prune_deleted_files(conn)
        t1 = time.time()

        print(f"Index DB: {dbp}")
        print(f"Scanned: {scanned:,} files")
        print(f"Indexed: {indexed:,} | Skipped (unchanged): {skipped:,} | Failed: {failed:,} | Removed: {removed:,}")
        print(f"Time: {(t1 - t0):.2f}s")
        print(f"Not indexable (by ext): {not_indexable:,}")
        if not_indexable_samples:
            print("Examples not indexed:")
            for s in not_indexable_samples:
                print(f"  - {s}")
    finally:
        conn.close()


# -----------------------------
# Querying + hybrid scoring
# -----------------------------

def score_filename(query: str, name: str, ext: str) -> Tuple[int, int, int]:
    """
    Smaller tuple, just for filename matching:
      (token_hits, substring_bonus, -position)
    """
    q = normalize(query)
    target = normalize(name + " " + ext.replace(".", ""))

    if not q:
        return (0, 0, 0)

    q_tokens = [t for t in q.split(" ") if t]
    token_hits = sum(1 for t in q_tokens if t in target)
    substring_bonus = 1 if q in target else 0

    pos = target.find(q_tokens[0]) if q_tokens else -1
    pos = pos if pos >= 0 else 10_000_000
    return (token_hits, substring_bonus, -pos)

def gather_candidates(conn: sqlite3.Connection, q_terms: List[str]) -> Dict[int, Dict[str, int]]:
    """
    Returns:
      doc_id -> { term -> count }
    """
    if not q_terms:
        return {}

    # Query each term and merge results.
    candidates: Dict[int, Dict[str, int]] = {}
    for term in q_terms:
        rows = conn.execute(
            "SELECT doc_id, count FROM terms WHERE term = ?",
            (term,),
        ).fetchall()
        for doc_id, cnt in rows:
            doc_id = int(doc_id)
            cnt = int(cnt)
            if doc_id not in candidates:
                candidates[doc_id] = {}
            candidates[doc_id][term] = cnt
    return candidates

def content_score(q_terms: List[str], term_counts: Dict[str, int]) -> Tuple[int, int]:
    """
    Returns (unique_terms_matched, total_term_hits)
    """
    unique = sum(1 for t in q_terms if t in term_counts)
    total = sum(term_counts.get(t, 0) for t in q_terms)
    return unique, total

def filename_candidate_docs(conn: sqlite3.Connection, query: str) -> List[int]:
    """
    Returns doc_ids where filename has at least one token hit or substring match.
    Uses docs table only (fast).
    """
    q = normalize(query)
    if not q:
        return []

    q_tokens = [t for t in q.split(" ") if t]
    if not q_tokens:
        return []

    rows = conn.execute("SELECT doc_id, path, ext FROM docs").fetchall()
    out: List[int] = []
    for doc_id, path, ext in rows:
        name = Path(str(path)).stem
        fn_hits, fn_sub, _fn_pos = score_filename(query, name, str(ext))
        if fn_hits > 0 or fn_sub > 0:
            out.append(int(doc_id))
    return out

def query_index(root: Path, query: str, max_results: int = 20) -> None:
    root = root.expanduser().resolve()
    dbp = db_path_for_root(root)
    if not dbp.exists():
        print(f"No index found at: {dbp}")
        print("Run: python content_search_index.py index <root>")
        return

    conn = connect_db(dbp)
    try:
        t0 = time.time()
        q_terms = tokenize(query)
        cand = gather_candidates(conn, q_terms)
        fn_docs = filename_candidate_docs(conn, query)
        for doc_id in fn_docs:
            cand.setdefault(doc_id, {})

        ranked: List[Tuple[Tuple[int, int, int, int, int, int, int], int]] = []
        # score tuple fields (bigger is better):
        # 0 content_unique_terms
        # 1 content_total_hits
        # 2 filename_token_hits
        # 3 filename_substring_bonus
        # 4 metadata_bonus
        # 5 filename_position (already negative)
        # 6 recency_bonus (small tie-breaker, newer slightly higher)
        now = time.time()

        for doc_id, per_term_counts in cand.items():
            row = conn.execute(
                "SELECT path, ext, size, mtime FROM docs WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                continue

            path, ext, size, mtime = row
            path = str(path)
            ext = str(ext)
            size = int(size)
            mtime = float(mtime)

            name = Path(path).stem

            cu, ct = content_score(q_terms, per_term_counts)
            fn_hits, fn_sub, fn_pos = score_filename(query, name, ext)
            m_bonus = metadata_bonus(query, mtime)

            # very small recency tie-breaker:
            # 0..3 based on how recent within last 30 days
            age_days = (now - mtime) / 86400.0
            if age_days < 1:
                rec = 3
            elif age_days < 7:
                rec = 2
            elif age_days < 30:
                rec = 1
            else:
                rec = 0

            sc = (cu, ct, fn_hits, fn_sub, m_bonus, fn_pos, rec)
            ranked.append((sc, doc_id))

        ranked.sort(key=lambda x: x[0], reverse=True)
        t1 = time.time()

        print(f"Index DB: {dbp}")
        print(f"Query terms: {q_terms}")
        print(f"Candidates: {len(ranked):,} | Time: {(t1 - t0):.3f}s")
        print(f"Showing top {min(max_results, len(ranked))}\n")

        for sc, doc_id in ranked[:max_results]:
            row = conn.execute(
                "SELECT path, ext, size, mtime FROM docs WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                continue

            path, ext, size, mtime = row
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(mtime)))

            print(f"[content={sc[0]}/{len(q_terms)} hits={sc[1]} | fn={sc[2]} sub={sc[3]} meta={sc[4]} rec={sc[6]}] {path}")
            print(f"  size={human_size(int(size))}  modified={dt}")

            snippet = best_effort_snippet(path, ext, q_terms)
            if snippet:
                print("  snippet:")
                for line in snippet:
                    print(f"    {line}")
            print()
    finally:
        conn.close()


# -----------------------------
# Snippets (best effort)
# -----------------------------

def best_effort_snippet(path: str, ext: str, q_terms: List[str], max_lines: int = 3) -> Optional[List[str]]:
    if ext not in INDEXABLE_EXTS or not q_terms:
        return None

    text = extract_text(path, ext, max_bytes=300_000)
    if not text:
        return None

    # Search line by line for first hit
    lines = text.splitlines()
    qset = set(q_terms)

    for i, line in enumerate(lines):
        toks = set(tokenize(line))
        if qset.intersection(toks):
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            out = []
            for j in range(start, end):
                out.append(highlight_terms(lines[j], q_terms))
                if len(out) >= max_lines:
                    break
            return out

    return None

def highlight_terms(line: str, q_terms: List[str]) -> str:
    # Simple highlighter: wraps exact token occurrences in brackets (case-insensitive-ish)
    out = line
    for t in sorted(set(q_terms), key=len, reverse=True):
        if not t:
            continue
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", f"[{t}]", out)
    return out


# -----------------------------
# Misc
# -----------------------------

def human_size(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.0f}{unit}" if unit == "B" else f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python content_search_index.py index <root_folder>")
        print('  python content_search_index.py query <root_folder> "<query>" [max_results]')
        sys.exit(1)

    mode = sys.argv[1].lower()
    root = Path(sys.argv[2])

    if mode == "index":
        index_root(root)
        return

    if mode == "query":
        if len(sys.argv) < 4:
            print('Missing query. Example: python content_search_index.py query "C:\\Docs" "deploy issue" 20')
            sys.exit(1)
        query = " ".join(sys.argv[3:-1]) if len(sys.argv) > 4 and sys.argv[-1].isdigit() else " ".join(sys.argv[3:])
        max_results = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 20
        query_index(root, query, max_results=max_results)
        return

    print(f"Unknown mode: {mode}. Use 'index' or 'query'.")
    sys.exit(1)


if __name__ == "__main__":
    main()