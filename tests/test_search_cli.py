import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pse_search


class SearchCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name).resolve()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_interactive_session_runs_queries_until_quit(self) -> None:
        out = io.StringIO()
        with mock.patch.object(pse_search, "validate_search_db", return_value=self.root / ".pse_index.sqlite3"):
            with mock.patch.object(pse_search, "query") as query_mock:
                with mock.patch("builtins.input", side_effect=["meeting summary", "", ":help", "project recap", ":quit"]):
                    with redirect_stdout(out):
                        pse_search.interactive_session(
                            self.root,
                            max_results=7,
                            debug=True,
                            lexical=True,
                        )

        self.assertEqual(2, query_mock.call_count)
        first_args = query_mock.call_args_list[0]
        second_args = query_mock.call_args_list[1]
        self.assertEqual(str(self.root), str(first_args.args[0]))
        self.assertEqual("meeting summary", first_args.args[1])
        self.assertEqual("project recap", second_args.args[1])
        self.assertEqual(7, first_args.kwargs["max_results"])
        self.assertTrue(first_args.kwargs["debug"])
        self.assertTrue(first_args.kwargs["lexical"])
        self.assertIn("Interactive search ready", out.getvalue())
        self.assertIn("Commands: :help, :quit", out.getvalue())

    def test_interactive_session_preloads_semantic_runtime_once(self) -> None:
        dummy_db = self.root / ".pse_index.sqlite3"
        dummy_conn = mock.Mock()

        with mock.patch.object(pse_search, "validate_search_db", return_value=dummy_db):
            with mock.patch.object(pse_search, "open_db_connection", return_value=dummy_conn) as open_db_mock:
                with mock.patch.object(pse_search, "ensure_semantic_ready", return_value=True) as ensure_ready_mock:
                    with mock.patch.object(pse_search, "get_reranker") as get_reranker_mock:
                        with mock.patch("builtins.input", side_effect=[":quit"]):
                            with redirect_stdout(io.StringIO()):
                                pse_search.interactive_session(self.root)

        open_db_mock.assert_called_once_with(dummy_db, pse_search.SEARCH_DB_PRAGMAS)
        ensure_ready_mock.assert_called_once_with(dummy_conn, dummy_db)
        get_reranker_mock.assert_called_once_with()
        dummy_conn.close.assert_called_once_with()

    def test_main_routes_to_interactive_session(self) -> None:
        argv = [
            "pse_search.py",
            str(self.root),
            "9",
            "--interactive",
            "--semantic",
            "--debug",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(pse_search, "interactive_session") as interactive_mock:
                with redirect_stdout(io.StringIO()):
                    pse_search.main()

        interactive_mock.assert_called_once()
        args, kwargs = interactive_mock.call_args
        self.assertEqual(str(self.root), str(args[0]))
        self.assertEqual(9, kwargs["max_results"])
        self.assertTrue(kwargs["semantic"])
        self.assertTrue(kwargs["debug"])


if __name__ == "__main__":
    unittest.main()
