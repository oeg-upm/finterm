"""Selected compatible regressions recovered from the old tests (see provenance)."""
from contextlib import closing
import json
import sqlite3

from tests.support import CODE, WorkingTest, jsonl
from tests.recovered_support import ControlledEncoder
from finterm_linking._runtime import frozen_runtime


class RecoveredProtocolTests(WorkingTest):
    def test_decimal_abbreviation_and_twenty_word_window(self):
        with frozen_runtime(CODE) as runtime:
            for text in ("The bank reported 0.5 percent.", "The bank purchased U.S. Treasury bonds.",
                         "The bank considered Dr. Smith's report carefully."):
                rows, _, info, _, _ = runtime.stage01.process_document("doc", None, text=text + " Next sentence.", terms=["bank"])
                self.assertEqual(rows[0]["context"], text)
                self.assertEqual(info["status"], "OK")
            prefix = " ".join("word" + str(i) for i in range(35))
            rows, _, info, _, _ = runtime.stage01.process_document("doc", None, text=prefix + " bank reported 0.5 on U.S. assets. Next.", terms=["bank"])
            self.assertTrue(rows[0]["context"].startswith("word15 "))
            self.assertEqual(info["max_words_before"], 20)

    def test_repeated_mentions_preserve_historical_dedup_and_mention_audit(self):
        with frozen_runtime(CODE) as runtime:
            rows, _, info, _, _ = runtime.stage01.process_document("doc", None, text="bank and bank.", terms=["bank"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(info["mention_audit"]["total_mentions"], 2)
            self.assertEqual(info["mention_audit"]["duplicate_context_mentions"], 1)
            self.assertEqual([row["normalized_start"] for row in info["mention_audit"]["mentions"]], [0, 9])

    def test_term_origin_changes_provenance_not_ids(self):
        with frozen_runtime(CODE) as runtime:
            results = []
            for origin in ("user", "external"):
                folder = self.root / origin
                runtime.stage01.run_stage01(folder, text="The museum stores manuscripts.", terms=["museum", "manuscripts"],
                                           document_id="doc", term_origin=origin)
                results.append(runtime.stage02.load_stage01(folder)["contexts"])
            self.assertEqual([r["occurrence_uid"] for r in results[0]], [r["occurrence_uid"] for r in results[1]])
            self.assertEqual([r["context"] for r in results[0]], [r["context"] for r in results[1]])

    def test_duplicate_terms_retain_provenance(self):
        with frozen_runtime(CODE) as runtime:
            from terminlink_inputs import DocumentInput, supplied_terms
            records, metadata = supplied_terms(DocumentInput.from_text("A rover.", "doc"), terms=["rover", "rover"])
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].provenance), 2)
            self.assertEqual(metadata["duplicate_terms"], {"rover": 2})

    def test_controlled_vectors_cannot_enter_real_model_namespace_and_corruption_is_rejected(self):
        with frozen_runtime(CODE) as runtime:
            encoder = ControlledEncoder()
            item = runtime.similarity.embedding_request("controlled text", encoder.identity)
            cache_path = self.root / "vectors.sqlite"
            with runtime.similarity.EmbeddingCache(cache_path) as cache:
                for vector in ([0, 0], [float("nan"), 1], [2, 0], [1, 0, 0]):
                    with self.subTest(vector=vector), self.assertRaises(ValueError):
                        cache.put(item, vector, encoder.runtime)
                with self.assertRaises(ValueError):
                    cache.put(item, [1, 0], {"backend_kind": runtime.native.BACKEND_KIND})
                cache.put(item, [1, 0], encoder.runtime)
            with closing(sqlite3.connect(cache_path)) as db, db:
                db.execute("UPDATE vectors SET data=?", (b"bad",))
            with runtime.similarity.EmbeddingCache(cache_path) as cache, self.assertRaisesRegex(ValueError, "payload hash"):
                cache.get(item)

    def test_model_preflight_is_metadata_only(self):
        import sys
        with frozen_runtime(CODE) as runtime:
            before = set(sys.modules)
            report = runtime.native.check_environment()
            self.assertFalse(report["ml_imported_by_preflight"])
            self.assertFalse({name.split(".")[0] for name in set(sys.modules) - before} & {"torch", "sentence_transformers"})
