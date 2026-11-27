"""Native DiSTER input serialization, with no model/tokenizer/chat template."""

import json
import unittest

from prompts import build_dister_prompt, default_prompt_config


class DiSTERPromptTests(unittest.TestCase):
    def test_native_conversation_matches_the_official_fields_and_wording(self):
        prompt = build_dister_prompt("synthetic-1", "A synthetic technical text.", "test-domain")
        expected = {
            "id": "synthetic-1",
            "conversations": [
                {"from": "human", "value": "Text: A synthetic technical text."},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": "What describes (technical or scientific) terms in the text, that are relevant to the domain test-domain?"},
                {"from": "gpt", "value": ""},
            ],
        }
        self.assertEqual(prompt.to_dict(), expected)
        self.assertEqual(json.loads(prompt.to_json()), expected)
        self.assertEqual(prompt.model_id, "ElenaSenger/DiSTER-Llama-3-8B-Instruct")

    def test_json_roundtrip_preserves_quotes_newlines_braces_and_unicode(self):
        identifier = 'id "quoted" {x}'
        text = 'Line "one"\n{json} \\ € é 漢字\t'
        domain = 'a "domain" {with braces}'
        record = json.loads(build_dister_prompt(identifier, text, domain).to_json())
        self.assertEqual(record["id"], identifier)
        self.assertEqual(record["conversations"][0]["value"], "Text: " + text)
        self.assertTrue(record["conversations"][2]["value"].endswith(domain + "?"))

    def test_generation_turn_is_empty_inside_valid_complete_json(self):
        record = json.loads(build_dister_prompt("id", "Synthetic.", "test").to_json())
        self.assertEqual(record["conversations"][-1], {"from": "gpt", "value": ""})
        self.assertEqual(len(record["conversations"]), 4)
        self.assertEqual(set(record), {"id", "conversations"})

    def test_no_generalist_json_output_requirement_or_chat_roles_are_added(self):
        native = build_dister_prompt("id", "Synthetic input.", "test").to_json()
        self.assertNotIn("Return only a JSON object", native)
        self.assertNotIn('"terms"', native)
        self.assertNotIn('"role"', native)
        self.assertNotIn("### Examples", native)
        self.assertNotIn("### Rules", native)

    def test_native_baseline_does_not_accept_demonstrations_or_reference_terms(self):
        with self.assertRaises(TypeError):
            build_dister_prompt("id", "Synthetic.", "test", retrieved_examples=[])
        with self.assertRaises(TypeError):
            build_dister_prompt("id", "Synthetic.", "test", reference_terms=["answer"])

    def test_the_domain_is_supplied_independently_of_the_native_format(self):
        for key in ("finance", "corruption", "equitation_dressage", "heart_failure", "wind_energy"):
            domain = default_prompt_config(key).domain.name
            prompt = build_dister_prompt("id", "Synthetic.", domain)
            self.assertIn(f"domain {domain}?", prompt.to_dict()["conversations"][2]["value"])

    def test_native_record_exports_are_independent(self):
        prompt = build_dister_prompt("id", "Synthetic.", "test")
        record = prompt.to_dict()
        record["conversations"][0]["value"] = "changed"
        self.assertEqual(prompt.to_dict()["conversations"][0]["value"], "Text: Synthetic.")

    def test_missing_identifier_text_or_domain_is_rejected(self):
        for values in (("", "text", "domain"), ("id", " ", "domain"), ("id", "text", "")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_dister_prompt(*values)


if __name__ == "__main__":
    unittest.main()
