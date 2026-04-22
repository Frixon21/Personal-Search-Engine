import builtins
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_semantic


class StaticEmbedder:
    def __init__(self, vectors: np.ndarray, dimension: int) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.dimension = dimension

    def encode(
        self,
        texts,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self.vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class SemanticHelperTests(unittest.TestCase):
    def test_get_embedder_reports_import_and_model_load_failures(self) -> None:
        original_import = builtins.__import__

        def missing_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        with mock.patch.object(pse_semantic, "_EMBEDDER", None):
            with mock.patch("builtins.__import__", side_effect=missing_import):
                with self.assertRaisesRegex(pse_semantic.SemanticSetupError, "sentence-transformers"):
                    pse_semantic.get_embedder()

        class FailingSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                raise RuntimeError("boom")

        def failing_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                return types.SimpleNamespace(SentenceTransformer=FailingSentenceTransformer)
            return original_import(name, *args, **kwargs)

        with mock.patch.object(pse_semantic, "_EMBEDDER", None):
            with mock.patch("builtins.__import__", side_effect=failing_import):
                with self.assertRaisesRegex(pse_semantic.SemanticSetupError, "Unable to load semantic model"):
                    pse_semantic.get_embedder()

    def test_build_index_meta_and_chunk_helpers_cover_trim_and_overlap_rules(self) -> None:
        meta = pse_semantic.build_index_meta(StaticEmbedder(np.zeros((1, 3), dtype=np.float32), 3))
        self.assertEqual(3, meta.embedding_dim)
        self.assertEqual(pse_semantic.SEMANTIC_MODEL_NAME, meta.model_name)

        self.assertEqual((2, 7), pse_semantic._trim_bounds("  alpha ", 0, 8))
        self.assertEqual(5, pse_semantic._find_chunk_end("alpha beta gamma", 0, 10))
        self.assertEqual(10, pse_semantic._find_chunk_end("abcdefghij klm", 0, 8))

        chunks = pse_semantic.chunk_text("  alpha beta gamma delta epsilon  ", chunk_chars=12, chunk_overlap=4)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(list(range(len(chunks))), [chunk.chunk_index for chunk in chunks])
        self.assertTrue(all(chunk.text == chunk.text.strip() for chunk in chunks))
        self.assertLess(chunks[1].start_char, chunks[0].end_char)

    def test_encode_helpers_normalize_vectors_and_handle_empty_inputs(self) -> None:
        empty = pse_semantic.encode_texts([], embedder=StaticEmbedder(np.zeros((0, 3), dtype=np.float32), 3))
        self.assertEqual((0, 3), empty.shape)

        embedder = StaticEmbedder(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32), 2)
        vectors = pse_semantic.encode_texts(["a", "b"], embedder=embedder)
        np.testing.assert_allclose(np.array([[0.6, 0.8], [0.0, 0.0]], dtype=np.float32), vectors)

        query = pse_semantic.encode_query("query", embedder=StaticEmbedder(np.array([[5.0, 0.0]], dtype=np.float32), 2))
        np.testing.assert_allclose(np.array([1.0, 0.0], dtype=np.float32), query)

        single = pse_semantic.encode_texts(["only"], embedder=StaticEmbedder(np.array([1.0, 2.0], dtype=np.float32), 2))
        self.assertEqual((1, 2), single.shape)
        np.testing.assert_allclose(np.array([[0.4472136, 0.8944272]], dtype=np.float32), single)

    def test_vector_blob_similarity_and_meta_helpers_cover_errors_and_round_trips(self) -> None:
        vector = np.array([1.5, 2.5], dtype=np.float32)
        blob = pse_semantic.vector_to_blob(vector)
        np.testing.assert_allclose(vector, pse_semantic.blob_to_vector(blob))

        with self.assertRaisesRegex(ValueError, "1D"):
            pse_semantic.vector_to_blob(np.array([[1.0, 2.0]], dtype=np.float32))

        self.assertAlmostEqual(5.0, pse_semantic.dot_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            pse_semantic.dot_similarity(np.array([1.0, 2.0]), np.array([1.0]))

        meta = pse_semantic.SemanticIndexMeta("backend", "model", 4, 100, 20, "structured-v3")
        with mock.patch.object(pse_semantic.time, "time", return_value=123.0):
            row = pse_semantic.semantic_meta_row(meta)
        self.assertEqual((1, "backend", "model", 4, 100, 20, "structured-v3", 123.0), row)

    def test_fetch_semantic_meta_returns_none_without_table_and_reads_existing_row(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            self.assertIsNone(pse_semantic.fetch_semantic_meta(conn))
            conn.execute(
                """
                CREATE TABLE semantic_meta (
                    singleton INTEGER PRIMARY KEY,
                    backend TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    chunk_chars INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    chunk_strategy TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self.assertIsNone(pse_semantic.fetch_semantic_meta(conn))
            conn.execute(
                """
                INSERT INTO semantic_meta(
                    singleton,
                    backend,
                    model_name,
                    embedding_dim,
                    chunk_chars,
                    chunk_overlap,
                    chunk_strategy,
                    updated_at
                )
                VALUES(1, 'backend', 'model', 8, 200, 50, 'structured-v3', 99.0)
                """
            )
            conn.commit()

            meta = pse_semantic.fetch_semantic_meta(conn)
        finally:
            conn.close()

        self.assertEqual(
            pse_semantic.SemanticIndexMeta(
                backend="backend",
                model_name="model",
                embedding_dim=8,
                chunk_chars=200,
                chunk_overlap=50,
                chunk_strategy="structured-v3",
            ),
            meta,
        )


if __name__ == "__main__":
    unittest.main()
