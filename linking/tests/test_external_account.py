"""Optional read-only real archive regression. Historical results are not shipped."""
import json
import os
from pathlib import Path
import unittest

from finterm_linking import Linker
from finterm_linking.replay import rebuild_wikidata_seed
from tests.support import CODE, WorkingTest, jsonl

PILOT = Path(os.environ["FINTERM_ACCOUNT_PILOT"]) if os.environ.get("FINTERM_ACCOUNT_PILOT") else None


@unittest.skipUnless(PILOT is not None, "Optional external account archive not configured")
class ExternalAccountTests(WorkingTest):
    def test_exact_archived_scores_associations_states_and_top_views(self):
        seed = rebuild_wikidata_seed(PILOT / "02_candidatos", self.root / "account_seed.json", code_dir=CODE)
        config = self.config(seed, lemma_map=PILOT / "02_candidatos/02_lemmas.json",
                             similarity_mode="archived_replay", archived_similarity_dir=PILOT / "06_similitud")
        case = json.loads((PILOT / "account_case.json").read_text(encoding="utf-8"))
        result = Linker(config).link(case["text"], ["account"], document_id="account_brokerage_001")
        self.assertEqual(len(result.occurrences), 1)
        self.assertEqual(len(result.rankings), 102)
        self.assertEqual(sum(row["score"] is not None for row in result.rankings), 93)
        self.assertEqual(result.rankings, jsonl(PILOT / "06_similitud/06_candidate_rankings.jsonl"))
        self.assertEqual(result.representations, jsonl(PILOT / "05_defterm/05_defterm_representations.jsonl"))
        self.assertEqual(result.execution_kind, "ARCHIVED_SCORE_REPLAY_NO_INFERENCE")
        self.assertFalse(result.real_embedding_inference_performed)
        # User-known references ONLY after ranking, never passed to Linker.
        index = {row["qid"]: row for row in result.rankings}
        self.assertEqual((index["Q2563572"]["rank_position"], index["Q2563572"]["score"]), (1, 0.6210758090019226))
        self.assertEqual(index["Q508295"]["rank_position"], 12)
        self.assertEqual(len(result.top(3)[0]["candidates"]), 3)
        self.assertEqual(len(result.top(5)[0]["candidates"]), 5)
