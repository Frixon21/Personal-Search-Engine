from dataclasses import dataclass
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pse_common import (
    INDEX_SCHEMA_VERSION,
    db_path_for_root,
    get_index_schema_version,
    human_size,
    metadata_bonus,
    open_db_connection,
    recency_bonus,
    score_filename,
    tokenize,
    fold_text,
)


SEARCH_DB_PRAGMAS = ("temp_store=MEMORY",)


@dataclass(frozen=True)
class RankedDoc:
    path: str
    size: int
    mtime: float
    preview_text: Optional[str]
    debug: Tuple[int, int, int, int, int, int, int]
    total: int


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


def query(
    root: Path,
    query_str: str,
    max_results: int = 20,
    debug: bool = False,
    show_snippets: bool = False,
) -> None:
    root = root.expanduser().resolve()
    dbp = db_path_for_root(root)
    if not dbp.exists():
        print(f"No index found at: {dbp}")
        print("Run: python pse_index.py index <root>")
        return

    schema_version = get_index_schema_version(dbp)
    if schema_version < INDEX_SCHEMA_VERSION:
        print(f"Index at {dbp} is outdated and must be rebuilt before searching.")
        print("Run: python pse_index.py index <root>")
        return
    if schema_version > INDEX_SCHEMA_VERSION:
        print(
            f"Index at {dbp} uses schema version {schema_version}, "
            f"but this code expects {INDEX_SCHEMA_VERSION}."
        )
        return

    show_snippets = show_snippets or debug

    conn = open_db_connection(dbp, SEARCH_DB_PRAGMAS)
    try:
        t0 = time.perf_counter()

        q_terms = tokenize(query_str)
        cand = gather_content_candidates(conn, q_terms)

        # Union with filename-only candidates.
        for doc_id in filename_candidate_docs(conn, query_str):
            cand.setdefault(doc_id, {})

        # Store:
        # - debug tuple: (cu, ct, fn_hits, fn_sub, meta, fn_pos, rec)
        # - doc details needed for output
        # - overall_score int
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
                    path=path,
                    size=size,
                    mtime=mtime,
                    preview_text=preview_text,
                    debug=dbg,
                    total=total,
                )
            )

        # Sort primarily by overall score, then by debug tuple.
        ranked.sort(key=lambda doc: (doc.total, doc.debug), reverse=True)
        query_time = time.perf_counter() - t0

        if debug:
            print(f"Index DB: {dbp}")
            print(f"Query terms: {q_terms}")
            print(f"Candidates: {len(ranked):,} | query_time={query_time:.3f}s")
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

        for idx, doc in enumerate(ranked[:max_results], start=1):
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(doc.mtime))

            cu, ct, fn_hits, fn_sub, meta, fn_pos, rec = doc.debug

            if debug:
                print(
                    f"#{idx}  score={doc.total}  "
                    f"[content={cu}/{len(q_terms)} hits={ct} | "
                    f"fn={fn_hits} sub={fn_sub} meta={meta} rec={rec}]"
                )
            else:
                print(f"#{idx}  score={doc.total}")

            print(f"  {doc.path}")

            if debug:
                print(f"  modified={dt}  size={human_size(doc.size)}")

            if show_snippets:
                snip = best_effort_snippet(doc.preview_text, q_terms)
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


def main() -> None:
    import sys

    # pse_search.py <root> "<query>" [max_results] [--debug] [--snippets]
    if len(sys.argv) < 3:
        print('Usage: python pse_search.py <root_folder> "<query>" [max_results] [--debug] [--snippets]')
        raise SystemExit(1)

    args = sys.argv[1:]
    debug = False
    show_snippets = False
    if "--debug" in args:
        debug = True
        args = [a for a in args if a != "--debug"]
    if "--snippets" in args:
        show_snippets = True
        args = [a for a in args if a != "--snippets"]

    root = Path(args[0])
    # Remaining args: query tokens + maybe max_results
    if len(args) >= 3 and args[-1].isdigit():
        max_results = int(args[-1])
        query_str = " ".join(args[1:-1])
    else:
        max_results = 20
        query_str = " ".join(args[1:])

    query(root, query_str, max_results=max_results, debug=debug, show_snippets=show_snippets)


if __name__ == "__main__":
    main()
