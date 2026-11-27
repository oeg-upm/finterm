"""Strict schema checks on literal protocol fixtures, not model predictions."""

import unittest

from llm_adapters import validate_terms_json


class ResponseValidationTests(unittest.TestCase):
    def test_valid_json_terms_are_returned_without_normalization(self):
        response = ' \n{"terms": ["MiXeD", "  unchanged  ", "MiXeD", "€ \\u00e9", "line\\nbreak"]}\n '
        parsed = validate_terms_json(response)
        self.assertEqual(parsed.response, response)
        self.assertTrue(parsed.json_valid)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.terms, ("MiXeD", "  unchanged  ", "MiXeD", "€ é", "line\nbreak"))
        self.assertEqual(parsed.errors, ())

    def test_valid_empty_list_is_distinct_from_an_invalid_response(self):
        valid = validate_terms_json('{"terms": []}')
        invalid = validate_terms_json('not JSON')
        self.assertEqual(valid.terms, ())
        self.assertEqual(valid.to_dict()["terms"], [])
        self.assertIsNone(invalid.terms)
        self.assertIsNone(invalid.to_dict()["terms"])

    def test_syntax_errors_are_reported_with_locations_and_original_text(self):
        response = '{"terms": ["fixture",]}'
        result = validate_terms_json(response)
        self.assertFalse(result.json_valid)
        self.assertFalse(result.schema_valid)
        self.assertEqual(result.response, response)
        self.assertEqual(result.errors[0].code, "invalid_json")
        self.assertIn("line", result.errors[0].location)

    def test_markdown_reasoning_and_explanations_are_not_stripped_or_repaired(self):
        for text in (
            '```json\n{"terms": ["fixture"]}\n```',
            'Explanation: {"terms": ["fixture"]}',
            '<think>reasoning fixture</think>{"terms": ["fixture"]}',
            '{"terms": []}\n{"terms": []}',
        ):
            with self.subTest(response=text):
                result = validate_terms_json(text)
                self.assertFalse(result.json_valid)
                self.assertEqual(result.response, text)
                self.assertIsNone(result.terms)

    def test_valid_json_can_still_have_an_invalid_top_level_schema(self):
        for text in ('[]', '["fixture"]', 'true', 'null', '12', '"text"'):
            with self.subTest(response=text):
                result = validate_terms_json(text)
                self.assertTrue(result.json_valid)
                self.assertFalse(result.schema_valid)
                self.assertEqual(result.errors[0].code, "not_object")
                self.assertIsNone(result.terms)

    def test_missing_wrong_type_or_extra_properties_are_format_errors(self):
        cases = (
            ('{}', "missing_terms"),
            ('{"terms": "fixture"}', "not_array"),
            ('{"terms": null}', "not_array"),
            ('{"terms": [], "extra": "fixture"}', "extra_properties"),
        )
        for text, code in cases:
            with self.subTest(response=text):
                result = validate_terms_json(text)
                self.assertTrue(result.json_valid)
                self.assertFalse(result.schema_valid)
                self.assertIn(code, [error.code for error in result.errors])
                self.assertIsNone(result.terms)

    def test_invalid_items_are_not_removed_or_coerced_to_strings(self):
        for item in ('12', 'true', 'null', '{}', '[]', '""', '"   "'):
            with self.subTest(item=item):
                result = validate_terms_json('{"terms": ["fixture", ' + item + ']}')
                self.assertTrue(result.json_valid)
                self.assertFalse(result.schema_valid)
                self.assertIsNone(result.terms)
                self.assertEqual(result.errors[0].location, "$.terms[1]")

    def test_duplicate_keys_are_detected_without_choosing_a_terms_list(self):
        text = '{"terms": ["first"], "terms": ["second"]}'
        result = validate_terms_json(text)
        self.assertTrue(result.json_valid)
        self.assertFalse(result.schema_valid)
        self.assertIn("duplicate_key", [error.code for error in result.errors])
        self.assertIsNone(result.terms)
        self.assertEqual(result.response, text)

    def test_nonstandard_numeric_constants_are_not_accepted_as_json(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                result = validate_terms_json('{"terms": [' + value + ']}')
                self.assertFalse(result.json_valid)
                self.assertIsNone(result.terms)

    def test_validation_is_not_semantic_term_filtering(self):
        text = '{"terms": ["Synthetic Institution", "2026", "word"]}'
        result = validate_terms_json(text)
        self.assertEqual(result.terms, ("Synthetic Institution", "2026", "word"))
        self.assertTrue(result.schema_valid)

    def test_non_string_responses_are_not_implicitly_converted(self):
        with self.assertRaises(TypeError):
            validate_terms_json({"terms": []})


if __name__ == "__main__":
    unittest.main()
