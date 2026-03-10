import io
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "indexing_fixture"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_index
import pse_search
from pse_common import (
    DEFAULT_IGNORE_FOLDERS,
    INDEX_SCHEMA_VERSION,
    PREVIEW_CHAR_CAP,
    db_path_for_root,
    index_jsonl_log_path_for_root,
    index_log_dir_for_root,
    index_text_log_path_for_root,
)


class IndexingFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run_index(self) -> tuple[pse_index.IndexRunStats, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            stats = pse_index.index_root(self.root)
        return stats, out.getvalue()

    def _write_config(self, content: str) -> Path:
        path = self.root / "pse_index.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def _copy_fixture_tree(self) -> None:
        shutil.copytree(FIXTURE_ROOT, self.root, dirs_exist_ok=True)

    def test_supported_file_stays_searchable_by_filename_when_extraction_fails(self) -> None:
        legacy_file = self.root / "legacy_report.doc"
        legacy_file.write_bytes(b"not really a doc")

        with mock.patch.object(pse_index, "extract_text", return_value=None):
            self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT doc_id, path, ext FROM docs").fetchone()
            self.assertIsNotNone(row)
            doc_id, path, ext = int(row[0]), str(row[1]), str(row[2])
            self.assertEqual(str(legacy_file.resolve()), path)
            self.assertEqual(".doc", ext)

            term_count = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            self.assertEqual(0, term_count)
            preview_count = conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0]
            self.assertEqual(0, preview_count)

            candidates = pse_search.filename_candidate_docs(conn, "legacy report")
            self.assertIn(doc_id, candidates)
        finally:
            conn.close()

    def test_empty_file_counts_as_empty_not_failed(self) -> None:
        empty_file = self.root / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        stats, output = self._run_index()

        self.assertEqual(1, stats.empty)
        self.assertIn("Indexed: 0 | Empty: 1 | Skipped: 0 | Failed: 0 | Removed: 0", output)

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(empty_file.resolve()),)).fetchone()
            self.assertIsNotNone(row)
            term_count = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            preview_count = conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0]
            self.assertEqual(0, term_count)
            self.assertEqual(0, preview_count)
        finally:
            conn.close()

    def test_preview_text_is_capped(self) -> None:
        long_text = "alpha " * (PREVIEW_CHAR_CAP + 100)
        doc = self.root / "long.txt"
        doc.write_text(long_text, encoding="utf-8")

        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            preview = conn.execute("SELECT preview_text FROM doc_previews").fetchone()
            self.assertIsNotNone(preview)
            self.assertEqual(PREVIEW_CHAR_CAP, len(preview[0]))
        finally:
            conn.close()

    def test_reindex_rebuilds_old_schema_and_populates_previews_for_unchanged_file(self) -> None:
        doc = self.root / "meeting_notes.txt"
        doc.write_text("Alpha beta\nSecond line", encoding="utf-8")
        st = doc.stat()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(
                """
                CREATE TABLE docs (
                    doc_id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    ext  TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    indexed_at REAL NOT NULL
                );

                CREATE TABLE terms (
                    term   TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    count  INTEGER NOT NULL,
                    PRIMARY KEY(term, doc_id)
                );
                """
            )
            conn.execute(
                """
                INSERT INTO docs(path, ext, size, mtime, indexed_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (str(doc.resolve()), ".txt", st.st_size, st.st_mtime, 0.0),
            )
            conn.execute(
                "INSERT INTO terms(term, doc_id, count) VALUES(?, ?, ?)",
                ("alpha", 1, 1),
            )
            conn.execute("PRAGMA user_version = 0")
            conn.commit()
        finally:
            conn.close()

        self._run_index()

        conn = sqlite3.connect(str(db_path))
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            preview = conn.execute("SELECT preview_text FROM doc_previews").fetchone()
            self.assertEqual(INDEX_SCHEMA_VERSION, version)
            self.assertIsNotNone(preview)
            self.assertIn("Alpha beta", preview[0])
        finally:
            conn.close()

    def test_deleted_file_is_pruned_from_fixture_tree(self) -> None:
        self._copy_fixture_tree()
        remove_path = (self.root / "nested" / "remove_me.txt").resolve()
        keep_path = (self.root / "keep.txt").resolve()

        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            doc_id = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(remove_path),)).fetchone()[0]
        finally:
            conn.close()

        remove_path.unlink()
        stats, output = self._run_index()

        self.assertEqual(1, stats.removed)
        self.assertIn("Removed: 1", output)

        conn = sqlite3.connect(str(db_path))
        try:
            removed_doc = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(remove_path),)).fetchone()
            kept_doc = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(keep_path),)).fetchone()
            term_rows = conn.execute("SELECT COUNT(*) FROM terms WHERE doc_id = ?", (doc_id,)).fetchone()[0]
            preview_rows = conn.execute("SELECT COUNT(*) FROM doc_previews WHERE doc_id = ?", (doc_id,)).fetchone()[0]
            self.assertIsNone(removed_doc)
            self.assertIsNotNone(kept_doc)
            self.assertEqual(0, term_rows)
            self.assertEqual(0, preview_rows)
        finally:
            conn.close()

    def test_config_allowed_extensions_prunes_previously_indexed_docs(self) -> None:
        txt_doc = self.root / "keep.txt"
        md_doc = self.root / "drop.md"
        txt_doc.write_text("Alpha text", encoding="utf-8")
        md_doc.write_text("Beta markdown", encoding="utf-8")

        self._run_index()
        self._write_config(
            "[index]\n"
            'allowed_extensions = [".txt"]\n'
        )

        stats, _output = self._run_index()

        self.assertEqual(1, stats.removed)
        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            kept_doc = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(txt_doc.resolve()),)).fetchone()
            removed_doc = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(md_doc.resolve()),)).fetchone()
            self.assertIsNotNone(kept_doc)
            self.assertIsNone(removed_doc)
        finally:
            conn.close()

    def test_config_ignore_folders_prunes_previously_indexed_subtree(self) -> None:
        keep_doc = self.root / "keep.txt"
        archive_dir = self.root / "archive"
        archive_doc = archive_dir / "hidden.txt"
        archive_dir.mkdir()
        keep_doc.write_text("Keep me", encoding="utf-8")
        archive_doc.write_text("Hide me", encoding="utf-8")

        self._run_index()
        ignore_folders = sorted(DEFAULT_IGNORE_FOLDERS | {"archive"})
        ignore_values = ", ".join(f'"{name}"' for name in ignore_folders)
        self._write_config(
            "[index]\n"
            f"ignore_folders = [{ignore_values}]\n"
        )

        stats, _output = self._run_index()

        self.assertEqual(1, stats.removed)
        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            kept_doc_row = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(keep_doc.resolve()),)).fetchone()
            removed_doc_row = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(archive_doc.resolve()),)).fetchone()
            self.assertIsNotNone(kept_doc_row)
            self.assertIsNone(removed_doc_row)
        finally:
            conn.close()

    def test_max_bytes_caps_text_content(self) -> None:
        doc = self.root / "capped.txt"
        doc.write_text("alpha beta gamma", encoding="utf-8")
        self._write_config(
            "[index]\n"
            "max_bytes = 5\n"
        )

        stats, _output = self._run_index()

        self.assertEqual(1, stats.indexed)
        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            terms = dict(conn.execute("SELECT term, count FROM terms").fetchall())
            preview = conn.execute("SELECT preview_text FROM doc_previews").fetchone()[0]
            self.assertEqual({"alpha": 1}, terms)
            self.assertEqual("alpha", preview)
        finally:
            conn.close()

    def test_oversized_rich_format_keeps_metadata_only_and_counts_size_skip(self) -> None:
        legacy_pdf = self.root / "legacy.pdf"
        legacy_pdf.write_bytes(b"x" * 64)
        self._write_config(
            "[index]\n"
            'allowed_extensions = [".pdf"]\n'
            "max_bytes = 8\n"
        )

        with mock.patch.object(pse_index, "extract_text", side_effect=AssertionError("extract_text should not run")):
            stats, output = self._run_index()

        self.assertEqual(1, stats.skipped_size)
        self.assertIn("Skip breakdown: unchanged=0 | extension=1 | size=1", output)

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT path, ext FROM docs WHERE path = ?", (str(legacy_pdf.resolve()),)).fetchone()
            term_count = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            preview_count = conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0]
            self.assertEqual((str(legacy_pdf.resolve()), ".pdf"), row)
            self.assertEqual(0, term_count)
            self.assertEqual(0, preview_count)
        finally:
            conn.close()

    def test_invalid_config_aborts_without_mutating_existing_index_and_logs_abort(self) -> None:
        doc = self.root / "stable.txt"
        doc.write_text("alpha", encoding="utf-8")
        self._run_index()
        db_path = db_path_for_root(self.root)

        conn = sqlite3.connect(str(db_path))
        try:
            before_rows = conn.execute("SELECT path, size, mtime FROM docs").fetchall()
        finally:
            conn.close()

        self._write_config(
            "[index]\n"
            'max_bytes = "bad"\n'
        )

        with self.assertRaisesRegex(ValueError, "max_bytes must be an integer"):
            pse_index.index_root(self.root)

        conn = sqlite3.connect(str(db_path))
        try:
            after_rows = conn.execute("SELECT path, size, mtime FROM docs").fetchall()
        finally:
            conn.close()

        self.assertEqual(before_rows, after_rows)

        log_path = index_jsonl_log_path_for_root(self.root)
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual("aborted", records[-1]["status"])
        self.assertIn("max_bytes must be an integer", records[-1]["error"])

    def test_run_logs_are_appended_and_internal_log_dir_is_excluded(self) -> None:
        doc = self.root / "alpha.txt"
        doc.write_text("alpha", encoding="utf-8")

        _stats1, _output1 = self._run_index()
        log_dir = index_log_dir_for_root(self.root)
        ignored_file = log_dir / "ignored.txt"
        ignored_file.write_text("should never be indexed", encoding="utf-8")

        stats2, _output2 = self._run_index()

        jsonl_path = index_jsonl_log_path_for_root(self.root)
        text_log_path = index_text_log_path_for_root(self.root)
        self.assertTrue(jsonl_path.exists())
        self.assertTrue(text_log_path.exists())

        json_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(2, len(json_records))
        self.assertEqual(stats2.scanned, json_records[-1]["scanned"])
        self.assertEqual(stats2.skipped, json_records[-1]["skipped"])
        self.assertEqual(stats2.removed, json_records[-1]["removed"])

        text_log = text_log_path.read_text(encoding="utf-8")
        self.assertEqual(2, text_log.count("Status: success"))

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            ignored_row = conn.execute("SELECT doc_id FROM docs WHERE path = ?", (str(ignored_file.resolve()),)).fetchone()
            self.assertIsNone(ignored_row)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
