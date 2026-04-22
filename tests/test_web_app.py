import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_index
from pse_common import tokenize
from pse_web.app import app
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


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.search_root = self.root / "search_root"
        self.search_root.mkdir()
        (self.search_root / "alpha_notes.txt").write_text("alpha beta gamma", encoding="utf-8")
        with semantic_embedder_patch():
            pse_index.index_root(self.search_root, quiet=True)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.tmpdir.cleanup()

    def test_home_page_renders_summary_with_default_root(self) -> None:
        with mock.patch("pse_web.app.services.get_default_root", return_value=self.search_root):
            response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn("Personal Search Engine", response.text)
        self.assertIn("Corpus Status", response.text)
        self.assertIn("Documents", response.text)

    def test_search_endpoint_returns_htmx_partial_results(self) -> None:
        with semantic_embedder_patch():
            response = self.client.get(
                "/search",
                params={
                    "root": str(self.search_root),
                    "query": "alpha",
                    "mode": "hybrid",
                    "top_k": 10,
                },
                headers={"HX-Request": "true"},
            )

        self.assertEqual(200, response.status_code)
        self.assertNotIn("<!DOCTYPE html>", response.text)
        self.assertIn("alpha_notes.txt", response.text)
        self.assertIn("<mark>alpha</mark>", response.text.casefold())

    def test_search_endpoint_full_page_empty_state_and_invalid_mode(self) -> None:
        empty = self.client.get(
            "/search",
            params={
                "root": str(self.search_root),
                "query": "   ",
                "mode": "hybrid",
                "top_k": 10,
            },
        )
        self.assertEqual(200, empty.status_code)
        self.assertIn("<!DOCTYPE html>", empty.text)
        self.assertIn("Start with a question or topic", empty.text)

        invalid = self.client.get(
            "/search",
            params={
                "root": str(self.search_root),
                "query": "alpha",
                "mode": "bogus",
                "top_k": 10,
            },
            headers={"HX-Request": "true"},
        )
        self.assertEqual(200, invalid.status_code)
        self.assertIn("Unsupported search mode", invalid.text)

    def test_search_endpoint_falls_back_to_default_top_k_and_uses_threadpool(self) -> None:
        real_search = services.search_documents
        seen_top_k: list[int] = []

        def wrapped_search(root, query, mode, top_k):
            seen_top_k.append(top_k)
            return real_search(root, query, mode, top_k)

        async_mock = mock.AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
        with mock.patch("pse_web.app.run_in_threadpool", async_mock):
            with mock.patch("pse_web.app.services.search_documents", side_effect=wrapped_search):
                with semantic_embedder_patch():
                    response = self.client.get(
                        "/search",
                        params={
                            "root": str(self.search_root),
                            "query": "alpha",
                            "mode": "hybrid",
                            "top_k": 999,
                        },
                        headers={"HX-Request": "true"},
                    )

        self.assertEqual(200, response.status_code)
        self.assertEqual([services.DEFAULT_TOP_K], seen_top_k)
        self.assertEqual(1, async_mock.await_count)

    def test_index_endpoint_runs_and_returns_status_partial(self) -> None:
        new_root = self.root / "new_root"
        new_root.mkdir()
        (new_root / "meeting_notes.txt").write_text("project sync recap", encoding="utf-8")

        with semantic_embedder_patch():
            response = self.client.post(
                "/index",
                data={"root": str(new_root)},
                headers={"HX-Request": "true"},
            )

        self.assertEqual(200, response.status_code)
        self.assertIn("Index complete", response.text)
        self.assertIn("Updated corpus summary", response.text)
        self.assertIn("Index updated for", response.text)

    def test_index_endpoint_full_page_uses_default_root_and_threadpool(self) -> None:
        default_root = self.root / "default_root"
        default_root.mkdir()
        (default_root / "fallback.txt").write_text("alpha beta", encoding="utf-8")

        async_mock = mock.AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
        with mock.patch("pse_web.app.run_in_threadpool", async_mock):
            with mock.patch("pse_web.app.services.get_default_root", return_value=default_root):
                with semantic_embedder_patch():
                    response = self.client.post("/index", data={"root": ""})

        self.assertEqual(200, response.status_code)
        self.assertIn("<!DOCTYPE html>", response.text)
        self.assertIn("Index complete", response.text)
        self.assertIn(str(default_root), response.text)
        self.assertIn("Updated corpus summary", response.text)
        self.assertEqual(1, async_mock.await_count)

    def test_index_endpoint_surfaces_invalid_root_errors(self) -> None:
        missing_root = self.root / "missing_root"

        response = self.client.post(
            "/index",
            data={"root": str(missing_root)},
            headers={"HX-Request": "true"},
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("Indexing failed", response.text)
        self.assertIn("does not exist", response.text)

    def test_home_page_handles_missing_default_index_cleanly(self) -> None:
        empty_root = self.root / "empty_root"
        empty_root.mkdir()

        with mock.patch("pse_web.app.services.get_default_root", return_value=empty_root):
            response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn("Missing", response.text)
        self.assertIn(str(empty_root), response.text)
