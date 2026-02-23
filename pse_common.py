import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    ext: str
    size: int
    mtime: float


SKIP_DIRS = {".git", "node_modules", "__pycache__"}
INDEXABLE_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".log", ".docx"
}

INDEX_DB_NAME = ".pse_index.sqlite3"


WEEKDAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def db_path_for_root(root: Path) -> Path:
    return root.expanduser().resolve() / INDEX_DB_NAME


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize(s)
    if not s:
        return []
    raw = re.split(r"[^a-z0-9]+", s)
    toks = [t for t in raw if t]

    out: List[str] = []
    for t in toks:
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)
    return out


def iter_files(roots: Iterable[Path]) -> Iterable[FileEntry]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fn in filenames:
                # Skip our own index database artifacts (db/wal/shm)
                if fn.startswith(INDEX_DB_NAME):
                    continue

                p = Path(dirpath) / fn
                ext = p.suffix.lower()

                try:
                    st = p.stat()
                except (PermissionError, FileNotFoundError, OSError):
                    continue

                yield FileEntry(
                    path=str(p),
                    name=p.stem,
                    ext=ext,
                    size=st.st_size,
                    mtime=st.st_mtime,
                )


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


def score_filename(query: str, name: str, ext: str) -> Tuple[int, int, int]:
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


def human_size(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.0f}{unit}" if unit == "B" else f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


def recency_bonus(mtime: float) -> int:
    now = time.time()
    age_days = (now - mtime) / 86400.0
    if age_days < 1:
        return 3
    if age_days < 7:
        return 2
    if age_days < 30:
        return 1
    return 0