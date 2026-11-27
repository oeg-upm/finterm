"""Read-only guarantees, import isolation, encoding records and core integrity."""

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import patch

from corpus_readers import BankAuthorization, discover_corpora, read_authorized_bank
from corpus_readers._io import physical_lines

from fixtures import ReaderFixtureCase


PROJECT = Path(__file__).resolve().parents[1]


class ReaderIsolationTests(ReaderFixtureCase):
    def test_discovery_inspects_paths_only_without_opening_annotation_or_text_files(self):
        self.acter()
        self.celex()
        with patch.object(Path, "open", side_effect=AssertionError("Discovery must not read contents")):
            catalog = discover_corpora(self.root)
        self.assertEqual(len(catalog.acter_documents), 1)
        self.assertEqual(len(catalog.celex_segments), 1)

    def test_all_reader_opens_are_rb_and_fixture_bytes_remain_unchanged(self):
        self.acter()
        self.celex()
        files = sorted(path for path in self.root.rglob("*") if path.is_file())
        before = {path: path.read_bytes() for path in files}
        actual_open = Path.open
        modes = []

        def readonly(path, *args, **kwargs):
            modes.append(args)
            self.assertEqual(args, ("rb",))
            return actual_open(path, *args, **kwargs)

        with patch.object(Path, "open", readonly):
            bank = read_authorized_bank(discover_corpora(self.root), BankAuthorization({"ACTER:corp_en_01", "CELEX:32013R0549"}))
        self.assertTrue(modes)
        self.assertEqual(before, {path: path.read_bytes() for path in files})
        for snapshot in bank.sources:
            self.assertEqual(snapshot.sha256, hashlib.sha256(before[snapshot.path]).hexdigest())
            self.assertEqual(snapshot.size_bytes, len(before[snapshot.path]))

    def test_import_and_synthetic_reading_cannot_import_or_execute_models(self):
        self.acter()
        code = """
import importlib.abc, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'nltk', 'fkassim', 'spacy', 'numpy', 'scipy', 'huggingface_hub', 'accelerate'}:
            raise AssertionError('Model import forbidden: ' + fullname)
sys.meta_path.insert(0, NoModels())
from corpus_readers import discover_corpora, BankAuthorization, read_authorized_bank
catalog = discover_corpora(sys.argv[1])
bank = read_authorized_bank(catalog, BankAuthorization({'ACTER:corp_en_01'}))
assert len(bank.examples) == 2
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.root)], cwd=PROJECT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_physical_lines_preserve_unicode_separators_and_do_not_add_eof_records(self):
        self.assertEqual(physical_lines("a\r\nb\nc\r"), ((1, "a"), (2, "b"), (3, "c")))
        self.assertEqual(physical_lines("a\u2028b"), ((1, "a\u2028b"),))
        self.assertEqual(physical_lines(""), ())
        self.assertEqual(physical_lines("\n"), ((1, ""),))

    def test_source_contains_no_writes_model_calls_or_retrieval_selection(self):
        forbidden = {"write_text", "write_bytes", "mkdir", "unlink", "rename", "rmdir", "touch",
                     "download", "from_pretrained", "generate", "extract", "retrieve", "score", "tokenize"}
        for path in (PROJECT / "corpus_readers").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None
                )
                self.assertNotIn(name, forbidden, (path.name, node.lineno))
                if name == "open":
                    self.assertEqual(node.args[0].value, "rb")

    def test_packaging_adds_only_the_external_reader_package(self):
        data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["tool"]["setuptools"]["packages"]["find"]["include"],
                         ["syntactic_retrieval*", "prompts*", "llm_adapters*", "extractor*", "corpus_readers*"])
        self.assertEqual(data["project"]["dependencies"], [])

    def test_all_48_approved_source_config_test_and_report_files_are_unchanged(self):
        paths = [path for folder in ("syntactic_retrieval", "prompts", "llm_adapters", "extractor")
                 for path in (PROJECT / folder).glob("*.py")]
        paths += list((PROJECT / "config").glob("*.toml"))
        paths += list((PROJECT / "tests").glob("*.py"))
        paths += [PROJECT / ".gitignore"] + list(PROJECT.glob("0[0-4]_*.md"))
        digest = hashlib.sha256()
        for path in sorted(paths):
            name = path.relative_to(PROJECT).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        self.assertEqual(len(paths), 48)
        self.assertEqual(digest.hexdigest(), "bf9f114e34d758de5ad54e9ad9014fb2521eef6d781a7dd07e47dd9c75abe05b")
