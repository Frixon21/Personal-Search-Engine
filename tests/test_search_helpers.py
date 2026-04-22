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
from pse_semantic import SemanticIndexMeta, vector_to_blob


class SearchHelperTests(unittest.TestCase):
    def test_highlight_and_snippet_helpers_cover_matches_fallbacks_and_truncation(self) -> None:
        highlighted = pse_search.highlight_terms("Alpha beta alphabet", ["alphabet", "alpha"])
        self.assertEqual("[alpha] beta [alphabet]", highlighted)

        preview = "First line\nAlpha beta\nThird line\nFourth line"
        self.assertEqual(
            ["[alpha] beta", "Third line"],
            pse_search.best_effort_snippet(preview, ["alpha"]),
        )
        self.assertIsNone(pse_search.best_effort_snippet(preview, ["missing"]))

        long_line = "x" * 300
        self.assertEqual(
            [long_line[:237] + "..."],
            pse_search.best_chunk_snippet("one\ntwo\n" + long_line, ["missing"]),
        )
        self.assertIsNone(pse_search.best_chunk_snippet("", ["alpha"]))

    def test_score_helpers_cover_weighting_normalization_and_bonus_scaling(self) -> None:
        self.assertEqual(98, pse_search.lexical_evidence_score(1, 2, 2, 1))
        self.assertEqual(120, pse_search.overall_score(1, 2, 2, 1, 2, 3))

        self.assertEqual({}, pse_search.normalize_scores({}))
        self.assertEqual({1: 1.0, 2: 1.0}, pse_search.normalize_scores({1: 5.0, 2: 5.0}))
        self.assertEqual({1: 0.0, 2: 0.0}, pse_search.normalize_scores({1: 0.0, 2: 0.0}))
        self.assertEqual({1: 0.0, 2: 0.5, 3: 1.0}, pse_search.normalize_scores({1: 2.0, 2: 4.0, 3: 6.0}))

        self.assertAlmostEqual(0.0, pse_search.hybrid_bonus(0, 0))
        self.assertAlmostEqual(0.3, pse_search.hybrid_bonus(3, 3))


class SearchDatabaseHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.conn = sqlite3.connect(":memory:")
        pse_index.init_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmpdir.cleanup()

    def _insert_doc(
        self,
        name: str,
        *,
        ext: str = ".txt",
        size: int = 10,
        mtime: float = 1000.0,
        preview_text: str | None = None,
        terms: dict[str, int] | None = None,
    ) -> tuple[int, str]:
        path = str((self.root / name).resolve())
        doc_id = pse_index.upsert_doc(self.conn, path, ext, size=size, mtime=mtime)
        if preview_text is not None:
            pse_index.replace_preview_for_doc(self.conn, doc_id, preview_text)
        if terms is not None:
            pse_index.replace_terms_for_doc(self.conn, doc_id, terms)
        self.conn.commit()
        return doc_id, path

    def _insert_chunk(self, doc_id: int, chunk_index: int, text: str, vector: np.ndarray) -> None:
        self.conn.execute(
            """
            INSERT INTO doc_chunks(doc_id, chunk_index, start_char, end_char, chunk_text, embedding)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (doc_id, chunk_index, 0, len(text), text, vector_to_blob(vector.astype(np.float32))),
        )
        self.conn.commit()

    def test_candidate_helpers_fetch_details_and_best_semantic_hits(self) -> None:
        doc_id, path = self._insert_doc(
            "alpha_notes.txt",
            preview_text="alpha preview",
            terms={"alpha": 2, "beta": 1},
        )
        filename_only_id, _ = self._insert_doc("meeting_alpha.md", ext=".md")
        self._insert_chunk(doc_id, 0, "weak chunk", np.array([0.2, 0.0], dtype=np.float32))
        self._insert_chunk(doc_id, 1, "best chunk", np.array([0.9, 0.0], dtype=np.float32))
        self._insert_chunk(filename_only_id, 0, "other chunk", np.array([0.5, 0.0], dtype=np.float32))

        self.assertEqual({doc_id: {"alpha": 2}}, pse_search.gather_content_candidates(self.conn, ["alpha"]))
        self.assertIn(filename_only_id, pse_search.filename_candidate_docs(self.conn, "meeting alpha"))
        self.assertEqual((1, 2), pse_search.content_score(["alpha", "missing"], {"alpha": 2}))
        self.assertEqual((path, ".txt", 10, 1000.0, "alpha preview"), pse_search.fetch_doc_details(self.conn, doc_id))
        self.assertIsNone(pse_search.fetch_doc_details(self.conn, 9999))

        hits = pse_search.gather_semantic_candidates(self.conn, np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual("best chunk", hits[doc_id].chunk_text)
        self.assertAlmostEqual(0.9, hits[doc_id].similarity)

    def test_build_keyword_ranked_docs_includes_filename_only_candidates(self) -> None:
        literal_id, literal_path = self._insert_doc(
            "alpha_report.txt",
            preview_text="alpha keyword",
            terms={"alpha": 3, "keyword": 1},
        )
        filename_only_id, filename_only_path = self._insert_doc("alpha_summary.md", ext=".md")

        q_terms, ranked = pse_search.build_keyword_ranked_docs(self.conn, "alpha keyword")

        self.assertEqual(["alpha", "keyword"], q_terms)
        self.assertEqual([literal_id, filename_only_id], [doc.doc_id for doc in ranked])
        self.assertEqual([literal_path, filename_only_path], [doc.path for doc in ranked])
        self.assertGreater(ranked[0].total, ranked[1].total)

    def test_build_semantic_ranked_docs_orders_by_similarity(self) -> None:
        first_id, first_path = self._insert_doc("first.txt", preview_text="first preview")
        second_id, second_path = self._insert_doc("second.txt", preview_text="second preview")
        self._insert_chunk(first_id, 0, "strong", np.array([0.9, 0.0], dtype=np.float32))
        self._insert_chunk(second_id, 0, "weak", np.array([0.3, 0.0], dtype=np.float32))

        with mock.patch.object(pse_search, "encode_query", return_value=np.array([1.0, 0.0], dtype=np.float32)):
            q_terms, ranked = pse_search.build_semantic_ranked_docs(self.conn, "alpha")

        self.assertEqual(["alpha"], q_terms)
        self.assertEqual([first_id, second_id], [doc.doc_id for doc in ranked])
        self.assertEqual([first_path, second_path], [doc.path for doc in ranked])
        self.assertEqual("first preview", ranked[0].preview_text)
        self.assertGreater(ranked[0].similarity, ranked[1].similarity)

    def test_build_hybrid_ranked_docs_merges_lexical_and_semantic_results(self) -> None:
        lexical_docs = [
            pse_search.RankedDoc(
                doc_id=1,
                path="one.txt",
                size=1,
                mtime=1.0,
                preview_text="one",
                debug=(2, 3, 1, 0, 0, 0, 0),
                total=0,
            ),
            pse_search.RankedDoc(
                doc_id=2,
                path="two.txt",
                size=1,
                mtime=2.0,
                preview_text="two",
                debug=(1, 1, 0, 0, 0, 0, 0),
                total=0,
            ),
        ]
        semantic_docs = [
            pse_search.SemanticRankedDoc(
                doc_id=2,
                path="two.txt",
                size=1,
                mtime=2.0,
                preview_text="two",
                chunk_text="two chunk",
                similarity=0.2,
                structure_multiplier=1.0,
                base_score=0.2,
                rerank_raw=None,
                rerank_bonus=0.0,
                score=0.2,
                debug=(0, 0, 0, 0),
            ),
            pse_search.SemanticRankedDoc(
                doc_id=3,
                path="three.txt",
                size=1,
                mtime=3.0,
                preview_text=None,
                chunk_text="three chunk",
                similarity=0.8,
                structure_multiplier=1.0,
                base_score=0.8,
                rerank_raw=None,
                rerank_bonus=0.0,
                score=0.8,
                debug=(0, 0, 0, 0),
            ),
        ]

        with mock.patch.object(pse_search, "build_keyword_ranked_docs", return_value=(["alpha"], lexical_docs)):
            with mock.patch.object(pse_search, "build_semantic_ranked_docs", return_value=(["alpha"], semantic_docs)):
                q_terms, ranked = pse_search.build_hybrid_ranked_docs(self.conn, "alpha")

        self.assertEqual(["alpha"], q_terms)
        self.assertEqual([3, 1, 2], [doc.doc_id for doc in ranked])
        self.assertEqual(1.0, ranked[0].semantic_norm)
        self.assertEqual(1.0, ranked[1].lexical_norm)
        self.assertEqual(0.0, ranked[2].total)

    def test_validate_search_db_ensure_semantic_ready_and_query_error_paths(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertIsNone(pse_search.validate_search_db(self.root))
        self.assertIn("No index found", out.getvalue())

        db_path = self.root / ".pse_index.sqlite3"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY)")
            conn.execute("PRAGMA user_version = 999")
            conn.commit()
        finally:
            conn.close()

        self.assertEqual(db_path.resolve(), pse_search.validate_search_db(self.root))

        expected_meta = SemanticIndexMeta("backend", "model", 8, 100, 20, "structured-v3")
        self.conn.execute(
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
            VALUES(1, 'wrong', 'model', 8, 100, 20, 'structured-v3', 1.0)
            """
        )
        self.conn.commit()

        out = io.StringIO()
        with mock.patch.object(pse_search, "build_index_meta", return_value=expected_meta):
            with redirect_stdout(out):
                self.assertFalse(pse_search.ensure_semantic_ready(self.conn, Path("index.sqlite3")))
        self.assertIn("must be rebuilt", out.getvalue())

        out = io.StringIO()
        with mock.patch.object(pse_search, "query_combined", side_effect=pse_search.SemanticSetupError("semantic unavailable")):
            with redirect_stdout(out):
                pse_search.query(self.root, "alpha")
        self.assertIn("semantic unavailable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
