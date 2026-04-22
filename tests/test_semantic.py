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
from pse_semantic import (
    SEMANTIC_BACKEND,
    SEMANTIC_CHUNK_CHARS,
    SEMANTIC_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_STRATEGY,
    SEMANTIC_MODEL_NAME,
    chunk_text,
)


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


class FakeReranker:
    GROUPS = FakeEmbedder.GROUPS

    def predict(
        self,
        sentences,
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        scores = []
        for query_text, candidate_text in sentences:
            query_tokens = set(tokenize(query_text))
            candidate_tokens = set(tokenize(candidate_text))
            score = 0.0
            for group in self.GROUPS:
                if query_tokens.intersection(group) and candidate_tokens.intersection(group):
                    score += 1.0
            score += 0.05 * len(query_tokens.intersection(candidate_tokens))
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)


def semantic_embedder_patch() -> mock._patch:
    return mock.patch("pse_semantic.get_embedder", return_value=FakeEmbedder())


def semantic_reranker_patch() -> mock._patch:
    return mock.patch("pse_semantic.get_reranker", return_value=FakeReranker())


class SemanticSearchTests(unittest.TestCase):
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
            with semantic_reranker_patch():
                with redirect_stdout(out):
                    pse_search.query(self.root, query, max_results=5, **kwargs)
        return out.getvalue()

    def test_indexing_writes_semantic_chunks_and_metadata(self) -> None:
        repeated = "intro filler " * 120
        self._write_text("story.txt", repeated + "\nmeeting notes summary about launch")

        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            meta_row = conn.execute(
                """
                SELECT backend, model_name, embedding_dim, chunk_chars, chunk_overlap, chunk_strategy
                FROM semantic_meta
                WHERE singleton = 1
                """
            ).fetchone()
            chunk_rows = conn.execute(
                """
                SELECT chunk_index, start_char, end_char, chunk_text, length(embedding)
                FROM doc_chunks
                ORDER BY chunk_index
                """
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(
            (
                SEMANTIC_BACKEND,
                SEMANTIC_MODEL_NAME,
                FakeEmbedder.DIMENSION,
                SEMANTIC_CHUNK_CHARS,
                SEMANTIC_CHUNK_OVERLAP,
                SEMANTIC_CHUNK_STRATEGY,
            ),
            meta_row,
        )
        self.assertGreater(len(chunk_rows), 1)
        self.assertEqual(list(range(len(chunk_rows))), [int(row[0]) for row in chunk_rows])
        self.assertTrue(all(int(row[1]) < int(row[2]) for row in chunk_rows))
        self.assertTrue(all(int(row[4]) == FakeEmbedder.DIMENSION * 4 for row in chunk_rows))

    def test_chunk_text_splits_short_multisection_note(self) -> None:
        text = (
            "Today we reviewed the current status of the capstone search engine.\n"
            "I discussed recent progress, unresolved issues, and next actions.\n\n"
            "Main points:\n"
            "- indexing pipeline was stabilized\n"
            "- semantic retrieval was implemented\n"
        )

        chunks = chunk_text(text)

        self.assertGreater(len(chunks), 2)
        self.assertIn(
            "I discussed recent progress, unresolved issues, and next actions.",
            [chunk.text for chunk in chunks],
        )
        self.assertTrue(any(chunk.text.startswith("Main points:") for chunk in chunks))

    def test_reindex_replaces_chunk_rows_for_changed_document(self) -> None:
        doc = self._write_text("mutable.txt", ("old topic " * 160) + "ending")
        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            initial_chunks = conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE doc_id = (SELECT doc_id FROM docs WHERE path = ?)",
                (str(doc.resolve()),),
            ).fetchone()[0]
        finally:
            conn.close()

        doc.write_text("hello world", encoding="utf-8")
        self._run_index()

        conn = sqlite3.connect(str(db_path))
        try:
            chunk_rows = conn.execute(
                """
                SELECT chunk_text
                FROM doc_chunks
                WHERE doc_id = (SELECT doc_id FROM docs WHERE path = ?)
                ORDER BY chunk_index
                """,
                (str(doc.resolve()),),
            ).fetchall()
        finally:
            conn.close()

        self.assertGreater(initial_chunks, 1)
        self.assertEqual([("hello world",)], chunk_rows)

    def test_semantic_query_returns_doc_without_literal_term_overlap(self) -> None:
        doc = self._write_text("hello_world.txt", "hello world")
        self._write_text("release_notes.txt", "deploy issue notes")
        self._run_index()

        output = self._run_query("greeting planet", semantic=True)

        self.assertIn(str(doc.resolve()), output)
        self.assertIn("#1", output)

    def test_semantic_query_snippet_uses_best_chunk_text(self) -> None:
        intro = "unrelated filler " * 90
        target = "meeting notes summary about launch"
        self._write_text("notes.txt", intro + "\n" + target)
        self._run_index()

        output = self._run_query("sync memo", semantic=True, show_snippets=True)

        self.assertIn("snippet:", output)
        self.assertIn(target, output)

    def test_semantic_ranking_penalizes_low_context_list_chunks(self) -> None:
        recap = self._write_text("recap.txt", "placeholder text")
        roadmap = self._write_text("roadmap.txt", "placeholder text")
        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            recap_doc_id = conn.execute(
                "SELECT doc_id FROM docs WHERE path = ?",
                (str(recap.resolve()),),
            ).fetchone()[0]
            roadmap_doc_id = conn.execute(
                "SELECT doc_id FROM docs WHERE path = ?",
                (str(roadmap.resolve()),),
            ).fetchone()[0]

            fake_hits = {
                int(roadmap_doc_id): pse_search.SemanticDocHit(
                    doc_id=int(roadmap_doc_id),
                    similarity=0.3132,
                    chunk_text=(
                        "Upcoming milestones for the project:\n"
                        "- complete the web interface\n"
                        "- improve testing and evaluation\n"
                        "- finalize the demonstration workflow\n"
                        "- prepare the final report and user guide\n\n"
                        "The presentation outlines the future plan for the next development phase"
                    ),
                ),
                int(recap_doc_id): pse_search.SemanticDocHit(
                    doc_id=int(recap_doc_id),
                    similarity=0.3019,
                    chunk_text="I discussed recent progress, unresolved issues, and next actions.",
                ),
            }

            with mock.patch.object(pse_search, "encode_query", return_value=np.zeros(FakeEmbedder.DIMENSION, dtype=np.float32)):
                with mock.patch.object(pse_search, "gather_semantic_candidates", return_value=fake_hits):
                    with semantic_reranker_patch():
                        _q_terms, ranked = pse_search.build_semantic_ranked_docs(conn, "meeting summary")
        finally:
            conn.close()

        self.assertEqual(str(recap.resolve()), ranked[0].path)
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertEqual(1.0, ranked[0].structure_multiplier)
        self.assertLess(ranked[1].structure_multiplier, 1.0)

    def test_semantic_reranking_uses_top_ten_candidates(self) -> None:
        for idx in range(11):
            self._write_text(f"doc_{idx}.txt", f"placeholder {idx}")
        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute("SELECT doc_id, path FROM docs ORDER BY path").fetchall()
            fake_hits = {}
            for rank, (doc_id, path) in enumerate(rows, start=1):
                fake_hits[int(doc_id)] = pse_search.SemanticDocHit(
                    doc_id=int(doc_id),
                    similarity=float(100 - rank),
                    chunk_text=f"candidate {rank}",
                )

            class RecordingReranker:
                def __init__(self) -> None:
                    self.calls: list[int] = []

                def predict(self, sentences, *, show_progress_bar: bool = False) -> np.ndarray:
                    self.calls.append(len(sentences))
                    return np.asarray(list(range(len(sentences), 0, -1)), dtype=np.float32)

            reranker = RecordingReranker()

            with mock.patch.object(pse_search, "encode_query", return_value=np.zeros(FakeEmbedder.DIMENSION, dtype=np.float32)):
                with mock.patch.object(pse_search, "gather_semantic_candidates", return_value=fake_hits):
                    _q_terms, ranked = pse_search.build_semantic_ranked_docs(conn, "meeting summary")
                    with mock.patch("pse_semantic.get_reranker", return_value=reranker):
                        reranked = pse_search.rerank_top_semantic_docs("meeting summary", ranked)
            self.assertEqual([10], reranker.calls)
            self.assertEqual(len(ranked), len(reranked))
        finally:
            conn.close()

    def test_semantic_query_rejects_outdated_semantic_metadata(self) -> None:
        self._write_text("hello_world.txt", "hello world")
        self._run_index()

        db_path = db_path_for_root(self.root)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("UPDATE semantic_meta SET model_name = ?", ("broken-model",))
            conn.commit()
        finally:
            conn.close()

        output = self._run_query("greeting planet", semantic=True)

        self.assertIn("must be rebuilt", output)
        self.assertIn("semantic searching", output)


if __name__ == "__main__":
    unittest.main()
