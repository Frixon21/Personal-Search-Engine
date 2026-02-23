import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime, timedelta
from typing import Optional

@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    ext: str
    size: int
    mtime: float

WEEKDAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

def extract_weekday_from_query(query: str) -> Optional[int]:
    q = normalize(query)
    for token, wd in WEEKDAY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            return wd
    return None

def infer_target_date_from_query(query: str) -> Optional[datetime.date]:
    """
    Minimal date intent:
      - weekday words -> most recent weekday (including today)
      - '21st' / '21' -> most recent day-of-month in current or previous month
    """
    q = normalize(query)
    if not q:
        return None

    today = datetime.now().date()

    # weekday
    for token, wd in WEEKDAY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            delta = (today.weekday() - wd) % 7
            return today - timedelta(days=delta)

    # ordinal like 21st, 3rd, 1st
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", q)
    if not m:
        # plain 1-31 number as a fallback
        m = re.search(r"\b(\d{1,2})\b", q)

    if m:
        dom = int(m.group(1))
        if 1 <= dom <= 31:
            # try this month first, else previous month
            y, mo = today.year, today.month
            try:
                cand = datetime(y, mo, dom).date()
                if cand <= today:
                    return cand
            except ValueError:
                pass

            # previous month
            if mo == 1:
                y, mo = y - 1, 12
            else:
                mo -= 1
            try:
                return datetime(y, mo, dom).date()
            except ValueError:
                return None

    return None

def metadata_bonus(query: str, entry: FileEntry) -> int:
    """
    Boost if query implies a weekday or day-of-month and file mtime matches.
    - Weekday: match the weekday of mtime (any week)
    - Day-of-month: match near the inferred target date (most recent dom)
    """
    # A) Weekday intent: boost any file whose mtime weekday matches
    wd = extract_weekday_from_query(query)
    if wd is not None:
        mdt = datetime.fromtimestamp(entry.mtime).date()
        if mdt.weekday() == wd:
            return 3
        return 0

    # B) Day-of-month intent: keep your previous "most recent date" logic
    target = infer_target_date_from_query(query)
    if target is None:
        return 0

    mdt = datetime.fromtimestamp(entry.mtime).date()
    diff = abs((mdt - target).days)

    if diff == 0:
        return 3
    if diff == 1:
        return 2
    if diff == 2:
        return 1
    return 0


def iter_files(roots: Iterable[Path]) -> Iterable[FileEntry]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip some noisy folders
            dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__"}]

            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except (PermissionError, FileNotFoundError):
                    continue

                yield FileEntry(
                    path=str(p),
                    name=p.stem,
                    ext=p.suffix.lower(),
                    size=st.st_size,
                    mtime=st.st_mtime,
                )

def normalize(s: str) -> str:
    s = s.lower()
    # treat separators as spaces
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def score_filename(query: str, entry: FileEntry) -> Tuple[int, int, int, int]:
    """
    Higher is better.
    Returns a tuple so Python sorts lexicographically:
      (token_hits, substring_bonus, -position)
    """
    q = normalize(query)
    target = normalize(entry.name + " " + entry.ext.replace(".", ""))

    if not q:
        return (0, 0, 0, 0)

    q_tokens = [t for t in q.split(" ") if t]
    token_hits = sum(1 for t in q_tokens if t in target)

    # bonus if the full normalized query is a substring
    substring_bonus = 1 if q in target else 0

    # earlier match is better (smaller position)
    pos = target.find(q_tokens[0]) if q_tokens else -1
    pos = pos if pos >= 0 else 10_000_000

    m_bonus = metadata_bonus(query, entry)
    
    return (token_hits, substring_bonus, m_bonus, -pos)

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def main():
    if len(sys.argv) < 3:
        print("Usage: python name_search_poc.py <root_folder> <query> [max_results]")
        print(r'Example: python name_search_poc.py "C:\Users\Alex\Documents" "deploy issue" 20')
        sys.exit(1)

    root = Path(sys.argv[1])
    query = " ".join(sys.argv[2:-1]) if len(sys.argv) > 3 and sys.argv[-1].isdigit() else " ".join(sys.argv[2:])
    max_results = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 20

    t0 = time.time()
    entries = list(iter_files([root]))
    t1 = time.time()

    ranked: List[Tuple[Tuple[int, int, int, int], FileEntry]] = []
    for e in entries:
        sc = score_filename(query, e)
        if sc[0] > 0 or sc[1] > 0 or sc[2] > 0:
            ranked.append((sc, e))

    ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"Scanned: {len(entries):,} files in {(t1 - t0):.2f}s")
    print(f"Matches: {len(ranked):,} | Showing top {min(max_results, len(ranked))}\n")

    for sc, e in ranked[:max_results]:
        dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.mtime))
        print(f"[hits={sc[0]} sub={sc[1]} meta={sc[2]}] {e.path}")
        print(f"  size={human_size(e.size)}  modified={dt}\n")

if __name__ == "__main__":
    main()
