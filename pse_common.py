import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import unicodedata


@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    ext: str
    size: int
    mtime: float


DEFAULT_IGNORE_FOLDERS = {".git", "node_modules", "__pycache__"}
INDEXABLE_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".log",
    ".pdf", ".doc", ".docx", ".rtf", ".odt", ".tex", ".html", ".htm",
    ".xls", ".xlsx", ".ppt", ".pptx",
}
DEFAULT_ALLOWED_EXTS = frozenset(INDEXABLE_EXTS)
SKIP_DIRS = set(DEFAULT_IGNORE_FOLDERS)

INDEX_DB_NAME = ".pse_index.sqlite3"
INDEX_CONFIG_NAME = "pse_index.toml"
INDEX_LOG_DIR_NAME = ".pse_index_logs"
INDEX_RUN_JSONL_NAME = "index_runs.jsonl"
INDEX_RUN_TEXT_NAME = "index_runs.log"
INDEX_SCHEMA_VERSION = 2
DEFAULT_MAX_BYTES = 2_000_000
PREVIEW_CHAR_CAP = 10_000


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


def index_config_path_for_root(root: Path) -> Path:
    return root.expanduser().resolve() / INDEX_CONFIG_NAME


def index_log_dir_for_root(root: Path) -> Path:
    return root.expanduser().resolve() / INDEX_LOG_DIR_NAME


def index_jsonl_log_path_for_root(root: Path) -> Path:
    return index_log_dir_for_root(root) / INDEX_RUN_JSONL_NAME


def index_text_log_path_for_root(root: Path) -> Path:
    return index_log_dir_for_root(root) / INDEX_RUN_TEXT_NAME


def index_db_artifact_paths(db_path: Path) -> List[Path]:
    return [
        db_path,
        Path(str(db_path) + "-wal"),
        Path(str(db_path) + "-shm"),
    ]


def get_index_schema_version(db_path: Path) -> int:
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error:
        return 0
    finally:
        conn.close()

    return int(row[0]) if row else 0


def open_db_connection(db_path: Path, pragmas: Iterable[str] = ()) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    for pragma in pragmas:
        conn.execute(f"PRAGMA {pragma};")
    return conn


def set_index_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")


def reset_index_db(db_path: Path) -> None:
    for artifact in index_db_artifact_paths(db_path):
        try:
            artifact.unlink()
        except FileNotFoundError:
            continue


def fold_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)

    # Remove invisible / layout chars commonly seen in PDFs
    s = s.translate({
        0x00AD: None,  # soft hyphen
        0x200B: None,  # zero width space
        0x200C: None,  # zero width non-joiner
        0x200D: None,  # zero width joiner
        0xFEFF: None,  # BOM
    })

    # Normalize common whitespace variants to space
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("\u2009", " ")  # thin space
    s = s.replace("\u202F", " ")  # narrow no-break space

    # Normalize punctuation that often varies in PDFs
    for old, new in {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2212": "-",
        "\u2014": "-",
    }.items():
        s = s.replace(old, new)

    # Strip diacritics so accented text still matches plain ASCII queries.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s)

    return s


def normalize(s: str) -> str:
    s = fold_text(s)
    s = s.casefold()  # better than lower() for Unicode
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


def _normalize_name_set(names: Optional[Iterable[str]]) -> set[str]:
    out: set[str] = set()
    if not names:
        return out

    for name in names:
        text = str(name).strip()
        if text:
            out.add(text.casefold())
    return out


def iter_files(
    roots: Iterable[Path],
    ignore_folders: Optional[Iterable[str]] = None,
    internal_ignore_dirs: Optional[Iterable[str]] = None,
) -> Iterable[FileEntry]:
    ignored_dirs = _normalize_name_set(
        DEFAULT_IGNORE_FOLDERS if ignore_folders is None else ignore_folders
    )
    ignored_dirs.update(_normalize_name_set(internal_ignore_dirs))

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d.casefold() not in ignored_dirs)

            for fn in sorted(filenames):
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

