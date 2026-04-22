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
import pse_search
from pse_common import db_path_for_root, tokenize


class FakeEmbedder:
    DIMENSION = 8
    GROUPS = (
        {"hello", "hi", "greeting"},
        {"world", "planet", "earth"},
        {"meeting", "sync", "standup"},
        {"notes", "memo", "summary"},
        {"deploy", "release", "shipment"},
        {"issue", "bug", "problem"},
    )

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIMENSION

    def encode(
        self,
        texts,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        rows = []
        for text in texts:
            tokens = tokenize(text)
            vector = np.zeros(self.DIMENSION, dtype=np.float32)
            for idx, group in enumerate(self.GROUPS):
                vector[idx] = sum(1 for token in tokens if token in group)
            for token in tokens:
                vector[6 + (sum(ord(ch) for ch in token) % 2)] += 1.0
            if not tokens:
                vector[-1] = 1.0
            rows.append(vector)

        arr = np.vstack(rows)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = arr / norms

        return arr.astype(np.float32, copy=False)


def semantic_embedder_patch() -> mock._patch:
    return mock.patch("pse_semantic.get_embedder", return_value=FakeEmbedder())


class SearchSnippetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_text(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _run_index(self) -> None:
        with semantic_embedder_patch():
            with redirect_stdout(io.StringIO()):
                pse_index.index_root(self.root)

    def _run_query(self, query: str, **kwargs) -> str:
        out = io.StringIO()
        with semantic_embedder_patch():
            with redirect_stdout(out):
                pse_search.query(self.root, query, max_results=5, **kwargs)
        return out.getvalue()

    def test_search_without_snippets_does_not_print_snippet_block(self) -> None:
        self._write_text("alpha.txt", "Alpha beta\nSecond line")
        self._run_index()

        output = self._run_query("alpha")

        self.assertIn("#1", output)
        self.assertNotIn("snippet:", output)

    def test_search_with_snippets_uses_cached_preview_text(self) -> None:
        self._write_text("alpha.txt", "Alpha beta\nSecond line")
        self._run_index()

        output = self._run_query("alpha", show_snippets=True)

        self.assertIn("snippet:", output)
        self.assertIn("[alpha] beta", output)

    def test_debug_search_prints_timings_and_snippets(self) -> None:
        self._write_text("alpha.txt", "Alpha beta\nSecond line")
        self._run_index()

        output = self._run_query("alpha", debug=True)

        self.assertIn("query_time=", output)
        self.assertIn("total_time=", output)
        self.assertIn("snippet:", output)

    def test_search_snippets_do_not_reopen_files(self) -> None:
        self._write_text("alpha.txt", "Alpha beta\nSecond line")
        self._run_index()

        with mock.patch("pse_extract.extract_text", side_effect=AssertionError("search should use cached previews")):
            output = self._run_query("alpha", show_snippets=True)

        self.assertIn("snippet:", output)

    def test_search_ignores_legacy_user_version_values(self) -> None:
        self._write_text("alpha.txt", "Alpha beta\nSecond line")
        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA user_version = 0")
            conn.commit()
        finally:
            conn.close()

        output = self._run_query("alpha")

        self.assertIn("#1", output)
        self.assertNotIn("must be rebuilt", output)

    def test_default_query_uses_combined_mode_and_can_return_semantic_only_hit(self) -> None:
        doc = self._write_text("hello_world.txt", "hello world")
        self._write_text("release_notes.txt", "deploy issue notes")
        self._run_index()

        output = self._run_query("greeting planet")

        self.assertIn(str(doc.resolve()), output)
        self.assertIn("#1", output)

    def test_lexical_flag_keeps_keyword_only_mode(self) -> None:
        doc = self._write_text("hello_world.txt", "hello world")
        self._write_text("release_notes.txt", "deploy issue notes")
        self._run_index()

        output = self._run_query("greeting planet", lexical=True)

        self.assertNotIn(str(doc.resolve()), output)
        self.assertNotIn("#1", output)


class SearchCliHelpTests(unittest.TestCase):
    def test_help_lists_all_search_options(self) -> None:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["pse_search.py", "--help"]):
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as exc:
                    pse_search.main()

        output = out.getvalue()
        self.assertEqual(0, exc.exception.code)
        self.assertIn("Usage:", output)
        self.assertIn('python pse_search.py <root_folder> "<query>" [max_results] [options]', output)
        self.assertIn("--lexical", output)
        self.assertIn("--semantic", output)
        self.assertIn("--debug", output)
        self.assertIn("--snippets", output)
        self.assertIn("--help", output)

    def test_unknown_option_prints_help(self) -> None:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["pse_search.py", "--bogus"]):
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as exc:
                    pse_search.main()

        output = out.getvalue()
        self.assertEqual(1, exc.exception.code)
        self.assertIn("Unknown option: --bogus", output)
        self.assertIn("--lexical", output)
        self.assertIn("--semantic", output)

    def test_conflicting_mode_flags_are_rejected(self) -> None:
        out = io.StringIO()
        with mock.patch.object(
            sys,
            "argv",
            ["pse_search.py", "C:\\tmp", "alpha", "--semantic", "--lexical"],
        ):
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as exc:
                    pse_search.main()

        output = out.getvalue()
        self.assertEqual(1, exc.exception.code)
        self.assertIn("Cannot use --semantic and --lexical together.", output)
        self.assertIn("--lexical", output)
        self.assertIn("--semantic", output)


if __name__ == "__main__":
    unittest.main()
