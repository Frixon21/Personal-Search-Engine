import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_index
from pse_semantic import SemanticIndexMeta, TextChunk


class IndexHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.conn = sqlite3.connect(":memory:")
        pse_index.init_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmpdir.cleanup()

    def test_term_counts_doc_state_and_upsert_doc_update_existing_rows(self) -> None:
        self.assertEqual({"alpha": 2, "beta": 1}, pse_index.term_counts_from_text("alpha beta alpha"))
        self.assertIsNone(pse_index.get_doc_state(self.conn, "missing.txt"))

        path = str((self.root / "doc.txt").resolve())
        first_id = pse_index.upsert_doc(self.conn, path, ".txt", size=10, mtime=1.0)
        second_id = pse_index.upsert_doc(self.conn, path, ".txt", size=20, mtime=2.0)
        self.conn.commit()

        self.assertEqual(first_id, second_id)
        self.assertEqual((first_id, 20, 2.0), pse_index.get_doc_state(self.conn, path))

    def test_replace_helpers_and_clear_doc_content_update_rows_consistently(self) -> None:
        path = str((self.root / "doc.txt").resolve())
        doc_id = pse_index.upsert_doc(self.conn, path, ".txt", size=10, mtime=1.0)

        pse_index.replace_terms_for_doc(self.conn, doc_id, {"alpha": 2})
        pse_index.replace_preview_for_doc(self.conn, doc_id, "preview text")

        chunks = [
            TextChunk(chunk_index=0, start_char=0, end_char=5, text="alpha"),
            TextChunk(chunk_index=1, start_char=6, end_char=10, text="beta"),
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with mock.patch.object(pse_index, "chunk_text", return_value=chunks):
            with mock.patch.object(pse_index, "encode_texts", return_value=embeddings):
                pse_index.replace_chunks_for_doc(self.conn, doc_id, "alpha beta")
        self.conn.commit()

        self.assertEqual([("alpha", 2)], self.conn.execute("SELECT term, count FROM terms").fetchall())
        self.assertEqual(("preview text",), self.conn.execute("SELECT preview_text FROM doc_previews").fetchone())
        self.assertEqual(
            [(0, "alpha"), (1, "beta")],
            self.conn.execute("SELECT chunk_index, chunk_text FROM doc_chunks ORDER BY chunk_index").fetchall(),
        )

        pse_index.clear_doc_content(self.conn, doc_id)
        self.conn.commit()

        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0])
        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0])
        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0])

        pse_index.replace_preview_for_doc(self.conn, doc_id, "   ")
        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM doc_previews").fetchone()[0])

    def test_replace_chunks_for_doc_raises_when_embedding_count_does_not_match(self) -> None:
        doc_id = pse_index.upsert_doc(self.conn, str((self.root / "doc.txt").resolve()), ".txt", size=10, mtime=1.0)

        with mock.patch.object(
            pse_index,
            "chunk_text",
            return_value=[
                TextChunk(chunk_index=0, start_char=0, end_char=5, text="alpha"),
                TextChunk(chunk_index=1, start_char=6, end_char=10, text="beta"),
            ],
        ):
            with mock.patch.object(pse_index, "encode_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)):
                with self.assertRaisesRegex(RuntimeError, "chunk embedding count"):
                    pse_index.replace_chunks_for_doc(self.conn, doc_id, "alpha beta")

    def test_replace_semantic_meta_and_config_normalizers_cover_valid_and_invalid_values(self) -> None:
        meta = SemanticIndexMeta("backend", "model", 8, 100, 20, "structured-v3")
        pse_index.replace_semantic_meta(self.conn, meta)
        pse_index.replace_semantic_meta(self.conn, SemanticIndexMeta("backend2", "model2", 9, 200, 30, "structured-v4"))
        self.conn.commit()

        self.assertEqual(
            ("backend2", "model2", 9, 200, 30, "structured-v4"),
            self.conn.execute(
                "SELECT backend, model_name, embedding_dim, chunk_chars, chunk_overlap, chunk_strategy FROM semantic_meta"
            ).fetchone(),
        )

        self.assertEqual(".txt", pse_index._normalize_extension("TXT"))
        self.assertEqual("archive", pse_index._normalize_folder_name("archive"))
        self.assertEqual([1, 2], pse_index._normalize_list([1, 2], "field"))
        self.assertEqual(16, pse_index._normalize_max_bytes(16))

        with self.assertRaisesRegex(ValueError, "strings"):
            pse_index._normalize_extension(1)
        with self.assertRaisesRegex(ValueError, "Unsupported extension"):
            pse_index._normalize_extension(".exe")
        with self.assertRaisesRegex(ValueError, "directory basenames"):
            pse_index._normalize_folder_name("nested/path")
        with self.assertRaisesRegex(ValueError, "must be a list"):
            pse_index._normalize_list("bad", "field")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            pse_index._normalize_max_bytes(True)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            pse_index._normalize_max_bytes(0)

    def test_load_index_config_pruning_and_logging_helpers_render_expected_output(self) -> None:
        default_config = pse_index.load_index_config(self.root)
        self.assertIn(".txt", default_config.allowed_extensions)

        config_path = self.root / "pse_index.toml"
        config_path.write_text(
            "[index]\n"
            'allowed_extensions = [".txt", "md"]\n'
            "max_bytes = 42\n"
            'ignore_folders = ["archive"]\n',
            encoding="utf-8",
        )
        loaded = pse_index.load_index_config(self.root)
        self.assertEqual(frozenset({".txt", ".md"}), loaded.allowed_extensions)
        self.assertEqual(42, loaded.max_bytes)
        self.assertEqual(frozenset({"archive"}), loaded.ignore_folders)

        config_path.write_text("top_level = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown top-level keys"):
            pse_index.load_index_config(self.root)

        config_path.write_text("[index]\nunknown = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown \\[index\\] keys"):
            pse_index.load_index_config(self.root)

        path_one = str((self.root / "one.txt").resolve())
        path_two = str((self.root / "two.txt").resolve())
        doc_one = pse_index.upsert_doc(self.conn, path_one, ".txt", size=10, mtime=1.0)
        pse_index.upsert_doc(self.conn, path_two, ".txt", size=10, mtime=1.0)
        self.conn.commit()

        removed_samples: list[str] = []
        removed = pse_index.prune_out_of_scope_docs(self.conn, {path_one}, removed_samples)
        self.conn.commit()

        self.assertEqual(1, removed)
        self.assertEqual([path_two], removed_samples)
        self.assertIsNotNone(self.conn.execute("SELECT doc_id FROM docs WHERE doc_id = ?", (doc_one,)).fetchone())
        self.assertIsNone(self.conn.execute("SELECT doc_id FROM docs WHERE path = ?", (path_two,)).fetchone())

        samples: list[str] = []
        for i in range(pse_index.RUN_SAMPLE_LIMIT + 2):
            pse_index._append_sample(samples, f"sample-{i}")
        self.assertEqual(pse_index.RUN_SAMPLE_LIMIT, len(samples))

        stats = pse_index.IndexRunStats(
            root=str(self.root),
            db_path=str(self.root / ".pse_index.sqlite3"),
            config=loaded,
            started_at="2026-03-10T10:00:00+00:00",
            finished_at="2026-03-10T10:01:00+00:00",
            scanned=5,
            indexed=2,
            empty=1,
            failed=1,
            removed=1,
            skipped_unchanged=1,
            skipped_not_indexable=2,
            skipped_size=3,
            duration_seconds=1.234,
            not_indexable_samples=["skip.ext"],
            size_skipped_samples=["big.pdf"],
            failed_samples=["bad.doc"],
            removed_samples=["gone.txt"],
            error="boom",
        )
        stats.finalize(pse_index.ABORTED_STATUS, stats.finished_at, stats.duration_seconds, error=stats.error)

        block = pse_index._format_text_log_block(stats)
        self.assertIn("Status: aborted", block)
        self.assertIn("Extension-skip samples: skip.ext", block)
        self.assertIn("Failed samples: bad.doc", block)

        out = io.StringIO()
        with redirect_stdout(out):
            pse_index.print_run_summary(stats)
        output = out.getvalue()
        self.assertIn("Examples skipped by extension:", output)
        self.assertIn("Examples pruned from index:", output)

    def test_build_aborted_stats_and_index_root_missing_dir_cover_error_paths(self) -> None:
        with mock.patch.object(pse_index, "_timestamp_now", return_value="done"):
            stats = pse_index._build_aborted_stats(
                self.root,
                self.root / ".pse_index.sqlite3",
                "start",
                None,
                1.0,
                ValueError("bad config"),
            )

        self.assertEqual(pse_index.ABORTED_STATUS, stats.status)
        self.assertEqual("done", stats.finished_at)
        self.assertEqual("bad config", stats.error)

        missing_root = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            pse_index.index_root(missing_root)


if __name__ == "__main__":
    unittest.main()
