import io
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
from pse_common import tokenize
from pse_web import services


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


class WebServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_text(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _run_index(self, *, quiet: bool = False) -> pse_index.IndexRunStats:
        with semantic_embedder_patch():
            return pse_index.index_root(self.root, quiet=quiet)

    def test_index_root_quiet_suppresses_console_output(self) -> None:
        self._write_text("alpha_notes.txt", "alpha beta gamma")

        out = io.StringIO()
        with redirect_stdout(out):
            stats = self._run_index(quiet=True)

        self.assertEqual("", out.getvalue())
        self.assertEqual(1, stats.indexed)

    def test_run_search_returns_structured_hybrid_results(self) -> None:
        doc = self._write_text("alpha_notes.txt", "alpha beta gamma")
        self._run_index(quiet=True)

        with semantic_embedder_patch():
            response = pse_search.run_search(self.root, "alpha", mode="hybrid", max_results=5)

        self.assertIsNone(response.error)
        self.assertEqual("hybrid", response.mode)
        self.assertEqual(1, response.candidate_count)
        self.assertEqual(("alpha",), response.query_terms)
        self.assertEqual(1, len(response.results))
        result = response.results[0]
        self.assertEqual(doc.name, result.name)
        self.assertEqual(str(doc.resolve()), result.path)
        self.assertTrue(any("alpha" in line.casefold() for line in result.snippet_lines))
        self.assertIn("semantic_norm", result.debug)

    def test_run_search_blank_query_invalid_mode_and_invalid_limit(self) -> None:
        blank = pse_search.run_search(self.root, "   ", mode="hybrid", max_results=5)
        self.assertIsNone(blank.error)
        self.assertEqual((), blank.results)
        self.assertIsNone(blank.index_path)

        with self.assertRaisesRegex(ValueError, "Unsupported search mode"):
            pse_search.run_search(self.root, "alpha", mode="bogus", max_results=5)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            pse_search.run_search(self.root, "alpha", mode="hybrid", max_results=0)

    def test_run_search_reports_missing_index_without_raising(self) -> None:
        response = pse_search.run_search(self.root, "alpha", mode="lexical", max_results=5)

        self.assertIn("No index found", response.error)
        self.assertEqual(str(self.root.resolve() / ".pse_index.sqlite3"), response.index_path)
        self.assertEqual(0, response.candidate_count)
        self.assertEqual((), response.results)

    def test_run_search_supports_lexical_semantic_and_reranked_hybrid_modes(self) -> None:
        self._write_text("alpha_notes.txt", "alpha beta gamma")
        self._write_text("beta_notes.txt", "alpha beta delta")
        self._run_index(quiet=True)

        with semantic_embedder_patch():
            lexical = pse_search.run_search(self.root, "alpha", mode="lexical", max_results=1)
        self.assertEqual("lexical", lexical.mode)
        self.assertEqual(1, len(lexical.results))
        self.assertIsNone(lexical.results[0].chunk_text)
        self.assertIn("content_hits", lexical.results[0].debug)

        with semantic_embedder_patch():
            with mock.patch.object(pse_search, "rerank_texts", return_value=np.array([0.1, 0.9], dtype=np.float32)):
                semantic = pse_search.run_search(self.root, "alpha", mode="semantic", max_results=5)
        self.assertEqual("semantic", semantic.mode)
        self.assertEqual(2, len(semantic.results))
        self.assertTrue(all(result.chunk_text for result in semantic.results))
        self.assertTrue(any(result.debug.get("rerank_raw") is not None for result in semantic.results))

        with semantic_embedder_patch():
            with mock.patch.object(pse_search, "rerank_texts", return_value=np.array([0.4, 0.6], dtype=np.float32)):
                hybrid = pse_search.run_search(self.root, "alpha", mode="hybrid", max_results=5)
        self.assertEqual("hybrid", hybrid.mode)
        self.assertEqual(2, len(hybrid.results))
        self.assertTrue(any(result.debug.get("rerank_raw") is not None for result in hybrid.results))

    def test_run_search_reports_outdated_semantic_index_and_runtime_errors(self) -> None:
        self._write_text("alpha_notes.txt", "alpha beta gamma")
        self._run_index(quiet=True)

        with mock.patch.object(pse_search, "_semantic_ready_error", return_value="rebuild required"):
            response = pse_search.run_search(self.root, "alpha", mode="semantic", max_results=5)
        self.assertEqual("rebuild required", response.error)

        with semantic_embedder_patch():
            with mock.patch.object(
                pse_search,
                "build_semantic_ranked_docs",
                side_effect=pse_search.SemanticSetupError("semantic unavailable"),
            ):
                response = pse_search.run_search(self.root, "alpha", mode="semantic", max_results=5)
        self.assertEqual("semantic unavailable", response.error)

    def test_index_summary_and_search_documents_format_template_data(self) -> None:
        self._write_text("alpha_notes.txt", "alpha beta gamma")
        self._run_index(quiet=True)

        summary = services.get_index_summary(self.root)
        self.assertTrue(summary.index_present)
        self.assertEqual(1, summary.doc_count)
        self.assertGreaterEqual(summary.chunk_count, 1)
        self.assertTrue(summary.semantic_configured)

        with semantic_embedder_patch():
            search = services.search_documents(self.root, "alpha", "hybrid", 10)

        self.assertTrue(search.submitted)
        self.assertFalse(search.error)
        self.assertEqual(1, search.result_count)
        self.assertIn("<mark>alpha</mark>", search.results[0].snippet_html_lines[0].casefold())
        self.assertIn("TXT", search.results[0].metadata_badges)
        self.assertTrue(search.results[0].debug_items)

    def test_search_documents_formats_errors_escaping_and_preview_limits(self) -> None:
        htmlish = "alpha <script>alert(1)</script>\n" + ("tail " * 500)
        self._write_text("alpha_notes.txt", htmlish)
        self._run_index(quiet=True)

        with semantic_embedder_patch():
            search = services.search_documents(self.root, "alpha", "hybrid", 10)

        self.assertIsNone(search.error)
        self.assertEqual(1, search.result_count)
        first = search.results[0]
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", first.snippet_html_lines[0])
        self.assertIn("<mark>alpha</mark>", first.snippet_html_lines[0].casefold())
        self.assertTrue(first.preview_excerpt.endswith("..."))
        self.assertTrue(first.detail_preview_text.endswith("..."))

        invalid = services.search_documents(self.root, "alpha", "bogus", 10)
        self.assertIn("Unsupported search mode", invalid.error)
        self.assertEqual("hybrid", invalid.mode)

    def test_get_index_summary_handles_missing_root_missing_index_and_broken_db(self) -> None:
        missing = services.get_index_summary(self.root / "missing")
        self.assertFalse(missing.root_exists)
        self.assertFalse(missing.index_present)
        self.assertIn("Folder not found", missing.error)

        empty_dir = self.root / "empty"
        empty_dir.mkdir()
        no_index = services.get_index_summary(empty_dir)
        self.assertTrue(no_index.root_exists)
        self.assertFalse(no_index.index_present)
        self.assertIsNone(no_index.error)

        broken = self.root / "broken"
        broken.mkdir()
        (broken / ".pse_index.sqlite3").write_text("not a sqlite db", encoding="utf-8")
        broken_summary = services.get_index_summary(broken)
        self.assertTrue(broken_summary.index_present)
        self.assertIn("Unable to read index summary", broken_summary.error)

    def test_run_index_returns_error_for_missing_root(self) -> None:
        missing = self.root / "missing"

        result = services.run_index(missing)

        self.assertFalse(result.success)
        self.assertIn("does not exist", result.message)
        self.assertFalse(result.summary.root_exists)

    def test_run_index_success_populates_summary_and_default_root_fallback(self) -> None:
        self._write_text("meeting_notes.txt", "project sync recap")

        result = services.run_index(self.root)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.stats)
        self.assertEqual(1, result.summary.doc_count)
        self.assertTrue(result.summary.index_present)

        with tempfile.TemporaryDirectory() as tmp:
            patched_root = Path(tmp)
            with mock.patch("pse_web.services.PROJECT_ROOT", patched_root):
                self.assertEqual(patched_root.resolve(), services.get_default_root())
