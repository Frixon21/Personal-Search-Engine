import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pse_common import (
    INDEXABLE_EXTS,
    db_path_for_root,
    human_size,
    metadata_bonus,
    recency_bonus,
    score_filename,
    tokenize,
)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


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


def extract_text_best_effort(path: str, ext: str, max_bytes: int = 300_000) -> Optional[str]:
    if ext not in INDEXABLE_EXTS:
        return None

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


def highlight_terms(line: str, q_terms: List[str]) -> str:
    out = line
    for t in sorted(set(q_terms), key=len, reverse=True):
        if not t:
            continue
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", f"[{t}]", out)
    return out


def best_effort_snippet(path: str, ext: str, q_terms: List[str], max_lines: int = 2) -> Optional[List[str]]:
    text = extract_text_best_effort(path, ext)
    if not text or not q_terms:
        return None

    lines = text.splitlines()
    qset = set(q_terms)

    for i, line in enumerate(lines):
        toks = set(tokenize(line))
        if qset.intersection(toks):
            # Return at most 2 lines: the hit line and one around it
            start = max(0, i - 0)
            end = min(len(lines), i + 2)
            out: List[str] = []
            for j in range(start, end):
                out.append(highlight_terms(lines[j], q_terms))
                if len(out) >= max_lines:
                    break
            return out

    return None


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
    return (
        content_unique * 20 +
        min(content_hits, 20) * 4 +
        fn_hits * 10 +
        fn_sub * 50 +
        meta * 8 +
        rec * 2
    )


def query(root: Path, query_str: str, max_results: int = 20, debug: bool = False) -> None:
    root = root.expanduser().resolve()
    dbp = db_path_for_root(root)
    if not dbp.exists():
        print(f"No index found at: {dbp}")
        print("Run: python pse_index.py index <root>")
        return

    conn = connect_db(dbp)
    try:
        t0 = time.time()

        q_terms = tokenize(query_str)
        cand = gather_content_candidates(conn, q_terms)

        # Union with filename-only candidates
        for doc_id in filename_candidate_docs(conn, query_str):
            cand.setdefault(doc_id, {})

        ranked: List[Tuple[Tuple[int, int, int, int, int, int, int], int, int]] = []
        # Store:
        # - debug tuple: (cu, ct, fn_hits, fn_sub, meta, fn_pos, rec)
        # - doc_id
        # - overall_score int
        for doc_id, per_term_counts in cand.items():
            row = conn.execute(
                "SELECT path, ext, size, mtime FROM docs WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                continue

            path, ext, size, mtime = str(row[0]), str(row[1]), int(row[2]), float(row[3])
            name = Path(path).stem

            cu, ct = content_score(q_terms, per_term_counts)
            fn_hits, fn_sub, fn_pos = score_filename(query_str, name, ext)
            meta = metadata_bonus(query_str, mtime)
            rec = recency_bonus(mtime)

            total = overall_score(cu, ct, fn_hits, fn_sub, meta, rec)
            dbg = (cu, ct, fn_hits, fn_sub, meta, fn_pos, rec)
            ranked.append((dbg, doc_id, total))

        # Sort primarily by overall score, then by debug tuple
        ranked.sort(key=lambda x: (x[2], x[0]), reverse=True)
        t1 = time.time()

        if debug:
            print(f"Index DB: {dbp}")
            print(f"Query terms: {q_terms}")
            print(f"Candidates: {len(ranked):,} | Time: {(t1 - t0):.3f}s")
            print()
            print("Legend:")
            print("  content=a/b  -> unique query terms matched / total query terms")
            print("  hits         -> total term occurrences in document")
            print("  fn           -> filename token matches")
            print("  sub          -> full query substring in filename (0/1)")
            print("  meta         -> date/weekday intent bonus")
            print("  rec          -> recency bonus (3<1d, 2<7d, 1<30d, 0>=30d)")
            print()
            print(f"Showing top {min(max_results, len(ranked))}\n")

        for idx, (dbg, doc_id, total) in enumerate(ranked[:max_results], start=1):
            row = conn.execute(
                "SELECT path, ext, size, mtime FROM docs WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                continue

            path, ext, size, mtime = str(row[0]), str(row[1]), int(row[2]), float(row[3])
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))

            cu, ct, fn_hits, fn_sub, meta, fn_pos, rec = dbg

            if debug:
                print(
                    f"#{idx}  score={total}  "
                    f"[content={cu}/{len(q_terms)} hits={ct} | "
                    f"fn={fn_hits} sub={fn_sub} meta={meta} rec={rec}]"
                )
            else:
                print(f"#{idx}  score={total}")

            print(f"  {path}")

            if debug:
                print(f"  modified={dt}  size={human_size(size)}")

            snip = best_effort_snippet(path, ext, q_terms)
            if snip:
                print("  snippet:")
                for line in snip:
                    print(f"    {line}")
            print()

    finally:
        conn.close()


def main() -> None:
    import sys

    # pse_search.py <root> "<query>" [max_results] [--debug]
    if len(sys.argv) < 3:
        print('Usage: python pse_search.py <root_folder> "<query>" [max_results] [--debug]')
        raise SystemExit(1)

    args = sys.argv[1:]
    debug = False
    if "--debug" in args:
        debug = True
        args = [a for a in args if a != "--debug"]

    root = Path(args[0])
    # Remaining args: query tokens + maybe max_results
    if len(args) >= 3 and args[-1].isdigit():
        max_results = int(args[-1])
        query_str = " ".join(args[1:-1])
    else:
        max_results = 20
        query_str = " ".join(args[1:])

    query(root, query_str, max_results=max_results, debug=debug)


if __name__ == "__main__":
    main()