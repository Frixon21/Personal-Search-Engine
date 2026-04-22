import sys
import tempfile
import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_common


class FixedMarch10DateTime(real_datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls(2026, 3, 10, tzinfo=tz)
        return cls(2026, 3, 10)


class FixedMarch1DateTime(real_datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls(2026, 3, 1, tzinfo=tz)
        return cls(2026, 3, 1)


class CommonPathAndDbHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_path_helpers_and_db_artifact_paths_use_expected_names(self) -> None:
        db_path = pse_common.db_path_for_root(self.root)

        self.assertEqual(".pse_index.sqlite3", db_path.name)
        self.assertEqual("pse_index.toml", pse_common.index_config_path_for_root(self.root).name)
        self.assertEqual(".pse_index_logs", pse_common.index_log_dir_for_root(self.root).name)
        self.assertEqual("index_runs.jsonl", pse_common.index_jsonl_log_path_for_root(self.root).name)
        self.assertEqual("index_runs.log", pse_common.index_text_log_path_for_root(self.root).name)
        self.assertEqual(
            [
                db_path,
                Path(str(db_path) + "-wal"),
                Path(str(db_path) + "-shm"),
            ],
            pse_common.index_db_artifact_paths(db_path),
        )

    def test_open_db_connection_applies_pragmas_and_reset_index_db_removes_artifacts(self) -> None:
        db_path = pse_common.db_path_for_root(self.root)
        conn = pse_common.open_db_connection(db_path, ("foreign_keys=ON",))
        try:
            self.assertEqual(1, conn.execute("PRAGMA foreign_keys").fetchone()[0])
            conn.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()

        for artifact in pse_common.index_db_artifact_paths(db_path)[1:]:
            artifact.write_bytes(b"x")

        pse_common.reset_index_db(db_path)

        for artifact in pse_common.index_db_artifact_paths(db_path):
            self.assertFalse(artifact.exists())


class CommonTextAndIterationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_fold_text_normalizes_unicode_whitespace_punctuation_and_diacritics(self) -> None:
        folded = pse_common.fold_text("Cafe\u0301\u00a0soft\u00adhyphen\u200b \u201cquote\u201d \u2014 dash")

        self.assertEqual('Cafe softhyphen "quote" - dash', folded)

    def test_normalize_and_tokenize_helpers_preserve_text_behavior(self) -> None:
        self.assertEqual("hello world 2026", pse_common.normalize("  Hello-World_2026  "))
        self.assertEqual(["42", "beta"], pse_common.tokenize("g o 42 beta"))

    def test_iter_files_skips_ignored_directories_internal_dirs_and_db_artifacts(self) -> None:
        keep = self.root / "Keep.TXT"
        skip_git = self.root / ".git" / "ignored.py"
        skip_logs = self.root / "temp_logs" / "ignored.md"
        skip_db = self.root / ".pse_index.sqlite3"
        self.root.joinpath(".git").mkdir()
        self.root.joinpath("temp_logs").mkdir()

        keep.write_text("alpha", encoding="utf-8")
        skip_git.write_text("beta", encoding="utf-8")
        skip_logs.write_text("gamma", encoding="utf-8")
        skip_db.write_text("db", encoding="utf-8")

        entries = list(
            pse_common.iter_files(
                [self.root, self.root / "does-not-exist"],
                ignore_folders=[" .GIT "],
                internal_ignore_dirs=[" temp_logs "],
            )
        )

        self.assertEqual(1, len(entries))
        self.assertEqual(str(keep), entries[0].path)
        self.assertEqual("Keep", entries[0].name)
        self.assertEqual(".txt", entries[0].ext)


class CommonDateAndScoreTests(unittest.TestCase):
    def test_extract_weekday_and_infer_target_date_cover_alias_and_previous_month_logic(self) -> None:
        self.assertEqual(1, pse_common.extract_weekday_from_query("notes from Tuesday"))
        self.assertIsNone(pse_common.extract_weekday_from_query("notes from someday"))

        with mock.patch.object(pse_common, "datetime", FixedMarch10DateTime):
            self.assertEqual(real_datetime(2026, 3, 9).date(), pse_common.infer_target_date_from_query("monday plan"))
            self.assertEqual(real_datetime(2026, 2, 15).date(), pse_common.infer_target_date_from_query("15th update"))

        with mock.patch.object(pse_common, "datetime", FixedMarch1DateTime):
            self.assertIsNone(pse_common.infer_target_date_from_query("31"))

    def test_metadata_bonus_filename_score_human_size_and_recency_bonus(self) -> None:
        monday_mtime = real_datetime(2026, 3, 9, 12, 0).timestamp()
        exact_mtime = real_datetime(2026, 3, 10, 12, 0).timestamp()
        next_day_mtime = real_datetime(2026, 3, 11, 12, 0).timestamp()

        with mock.patch.object(pse_common, "datetime", FixedMarch10DateTime):
            self.assertEqual(3, pse_common.metadata_bonus("monday notes", monday_mtime))
            self.assertEqual(3, pse_common.metadata_bonus("10th notes", exact_mtime))
            self.assertEqual(2, pse_common.metadata_bonus("10th notes", next_day_mtime))
            self.assertEqual(0, pse_common.metadata_bonus("plain query", exact_mtime))

        self.assertEqual((2, 1, 0), pse_common.score_filename("meeting notes", "Meeting_Notes_2026", ".md"))
        self.assertEqual((0, 0, 0), pse_common.score_filename("", "whatever", ".txt"))
        self.assertEqual("500B", pse_common.human_size(500))
        self.assertEqual("1.5KB", pse_common.human_size(1536))

        now = 2_000_000.0
        with mock.patch.object(pse_common.time, "time", return_value=now):
            self.assertEqual(3, pse_common.recency_bonus(now - 3600))
            self.assertEqual(2, pse_common.recency_bonus(now - (3 * 86400)))
            self.assertEqual(1, pse_common.recency_bonus(now - (10 * 86400)))
            self.assertEqual(0, pse_common.recency_bonus(now - (60 * 86400)))


if __name__ == "__main__":
    unittest.main()
