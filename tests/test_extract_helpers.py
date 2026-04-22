import builtins
import io
import sys
import tempfile
import types
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pse_extract


class ExtractHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_text(self, name: str, content: str, encoding: str = "utf-8") -> Path:
        path = self.root / name
        path.write_text(content, encoding=encoding)
        return path

    def _write_zip_bytes(self, name: str, members: dict[str, bytes]) -> Path:
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member_name, content in members.items():
                archive.writestr(member_name, content)
        return path

    def test_text_builder_caps_output_and_support_helpers_cover_caps_and_dispatch(self) -> None:
        builder = pse_extract.TextBuilder(5)
        builder.append("alpha")
        builder.append("beta")
        self.assertTrue(builder.full)
        self.assertEqual("alpha", builder.build())

        self.assertTrue(pse_extract.supports_partial_byte_cap(".TXT"))
        self.assertFalse(pse_extract.supports_partial_byte_cap(".pdf"))
        self.assertEqual(4096, pse_extract._byte_cap(1))
        self.assertEqual(16_000_000, pse_extract._byte_cap(10_000_000))
        self.assertEqual(50, pse_extract._effective_byte_cap(100, 50))
        self.assertEqual(1, pse_extract._effective_byte_cap(100, 1))

        txt_path = self._write_text("sample.txt", "alpha beta gamma")
        self.assertEqual("alpha", pse_extract.extract_text(str(txt_path), ".txt", 5))
        self.assertIsNone(pse_extract.extract_text(str(txt_path), ".exe", 5))

    def test_read_decode_and_cleanup_helpers_cover_missing_data_and_fallbacks(self) -> None:
        text_path = self._write_text("latin1.txt", "olá", encoding="latin-1")
        zip_path = self._write_zip_bytes("bundle.zip", {"inside.txt": b"hello"})

        self.assertEqual(b"ol\xe1", pse_extract._read_bytes(str(text_path), 10))
        self.assertIsNone(pse_extract._read_bytes(str(self.root / "missing.txt"), 10))
        self.assertEqual(b"hello", pse_extract._read_zip_member(str(zip_path), "inside.txt"))
        self.assertIsNone(pse_extract._read_zip_member(str(zip_path), "missing.txt"))

        self.assertEqual("olá", pse_extract._decode_bytes(b"ol\xe1"))
        self.assertEqual("a\nb\n\nc", pse_extract._clean_visible_text(" a \r\n b \n\n\n c "))
        self.assertEqual("tag", pse_extract._local_name("{ns}tag"))
        elem = ET.Element("{ns}node", {"{ns}c": "3"})
        self.assertEqual("3", pse_extract._attr_local(elem, "c"))
        self.assertEqual("fallback", pse_extract._attr_local(elem, "missing", "fallback"))

    def test_xml_and_odt_helpers_cover_breaks_spaces_and_parse_failures(self) -> None:
        builder = pse_extract.TextBuilder(100)
        pse_extract._append_openxml_text(
            builder,
            b"<root><p><t>Alpha</t><br/><t>Beta</t></p></root>",
            {"t"},
            {"br"},
            {"p"},
        )
        self.assertEqual("Alpha\nBeta", pse_extract._clean_visible_text(builder.build()))

        broken = pse_extract.TextBuilder(100)
        pse_extract._append_openxml_text(broken, b"<broken", {"t"}, {"br"}, {"p"})
        self.assertEqual("", broken.build())

        odt_builder = pse_extract.TextBuilder(100)
        root = ET.fromstring(
            "<text:p xmlns:text='urn:oasis:names:tc:opendocument:xmlns:text:1.0'>"
            "Alpha<text:s text:c='2'/>"
            "<text:span>Beta</text:span>"
            "<text:tab/>"
            "<text:line-break/>"
            "Gamma"
            "</text:p>"
        )
        pse_extract._append_odt_node(odt_builder, root)
        self.assertEqual("Alpha Beta\nGamma", pse_extract._clean_visible_text(odt_builder.build()))

    def test_iter_excel_rows_shape_text_safe_invoke_and_com_runner_cover_edge_paths(self) -> None:
        self.assertEqual([(1,)], list(pse_extract._iter_excel_rows(1)))
        self.assertEqual([(1, 2)], list(pse_extract._iter_excel_rows((1, 2))))
        self.assertEqual([(1, 2), (3,)], list(pse_extract._iter_excel_rows(((1, 2), 3))))

        class ShapeWithFallback:
            @property
            def TextFrame(self):
                raise RuntimeError("bad text frame")

            TextFrame2 = types.SimpleNamespace(
                HasText=True,
                TextRange=types.SimpleNamespace(Text="fallback text"),
            )

        self.assertEqual("fallback text", pse_extract._shape_text(ShapeWithFallback()))
        self.assertEqual("", pse_extract._shape_text(types.SimpleNamespace()))

        class RaisesOnCall:
            def boom(self) -> None:
                raise RuntimeError("ignore me")

        pse_extract._safe_invoke(RaisesOnCall(), "boom")
        pse_extract._safe_invoke(None, "boom")

        with mock.patch.object(pse_extract, "_import_pywin32", return_value=(None, None)):
            self.assertIsNone(pse_extract._run_with_com_app("Word.Application", lambda app: "never"))

        class FakePythonCom:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def CoInitialize(self) -> None:
                self.calls.append("init")

            def CoUninitialize(self) -> None:
                self.calls.append("uninit")

        class FakeApp:
            def __init__(self) -> None:
                self.quit_called = False

            def Quit(self) -> None:
                self.quit_called = True

        class FakeClient:
            def __init__(self, app) -> None:
                self.app = app

            def DispatchEx(self, prog_id: str):
                return self.app

        pythoncom = FakePythonCom()
        app = FakeApp()
        cleanup_called: list[str] = []

        with mock.patch.object(pse_extract, "_import_pywin32", return_value=(pythoncom, FakeClient(app))):
            result = pse_extract._run_with_com_app(
                "Word.Application",
                lambda _app: (_ for _ in ()).throw(RuntimeError("worker failed")),
                cleanup=lambda: cleanup_called.append("cleanup"),
            )

        self.assertIsNone(result)
        self.assertEqual(["cleanup"], cleanup_called)
        self.assertTrue(app.quit_called)
        self.assertEqual(["init", "uninit"], pythoncom.calls)

    def test_html_and_pdf_extractors_cover_optional_dependency_fallbacks(self) -> None:
        html_path = self._write_text(
            "sample.html",
            "<html><body><script>ignore()</script><p>Hello <b>world</b></p></body></html>",
        )

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("missing bs4")
            if name == "fitz":
                raise ImportError("missing fitz")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            text = pse_extract._extract_html_text(str(html_path), 200)
            pdf_text = pse_extract._extract_pdf_text(str(self.root / "missing.pdf"), 200)

        self.assertEqual("Hello world", text)
        self.assertIsNone(pdf_text)


if __name__ == "__main__":
    unittest.main()
