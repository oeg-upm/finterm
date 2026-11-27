"""Only synthetic files under a temporary test directory; never real corpora."""

import os
from pathlib import Path
import tempfile
import unittest


class ReaderFixtureCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(
            prefix="finterm-readers-", dir=os.environ.get("FINTERM_TEST_TMPDIR")
        )
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    @staticmethod
    def write(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if isinstance(contents, bytes) else contents.encode("utf-8"))
        return path

    def acter(self, source_id="corp_en_01", *, annotation="Central\tB\nbank\tI\nrates\tB\nand\tO\nrates\tB\n.\tO\n\nNo\tO\nterms\tO\n.\tO\n", tokenized="Central  bank rates and rates .  \nNo terms .\n", prefix="relocated/another-level", scheme="iob_annotations", named="without_named_entities"):
        domain, language, _ = source_id.split("_")
        base = self.root / prefix / language / domain / "annotated"
        annotation_path = self.write(
            base / "annotations" / "sequential_annotations" / scheme / named / (source_id + "_seq_terms.tsv"), annotation
        )
        tokenized_path = self.write(base / "texts_tokenised" / (source_id + ".txt"), tokenized)
        original_path = self.write(base / "texts" / (source_id + ".txt"), "Synthetic untokenised source; no global term list.\n")
        return annotation_path, tokenized_path, original_path

    def celex(self, source_id="32013R0549_seg_0", *, text="CIF-value. requirements. European Central Bank.  edge  \r\nAnother sentence.", key="CIF value\r\nCIF\r\nRequirements\r\nEuropean Central Bank\r\nCIF\r\n  edge  ", prefix="relocated-finance"):
        text_path = self.write(self.root / prefix / "texts-in-unexpected-place" / (source_id + ".txt"), text)
        key_path = self.write(self.root / prefix / "references-in-another-place" / (source_id + ".key"), key)
        return text_path, key_path
