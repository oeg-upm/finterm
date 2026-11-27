"""Synthetic, text-only tests of the original annotation guidelines’ four generalist conditions."""

from dataclasses import replace
import json
import re
import unittest

from prompts import (
    AnnotationPolicy, DomainSpec, InsufficientExamplesError, NestedTerms,
    PromptBuilder, PromptCondition, PromptConfig, default_prompt_config,
)
from syntactic_retrieval.models import AnnotatedExample, RetrievedExample


def demonstrations(count=3):
    return tuple(
        RetrievedExample(
            example=AnnotatedExample(
                identifier=f"synthetic-{index}",
                text=f'Synthetic text {index}: "quoted" {{data}}.\nA second line €.',
                terms=(f'term "{index}"', "short term", "a short term", "a short term", "line\nbreak\\"),
                document_id=f"synthetic-document-{index}",
                domain="synthetic example domain",
            ),
            score=(index + 1) / 7,
        )
        for index in range(count)
    )


def example_outputs(content):
    return [json.loads(match) for match in re.findall(r"\nOutput: (\{[^\n]*\})", content)]


class BasicPromptTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(default_prompt_config("finance"))

    def test_basic_wording_matches_the_supplied_prompt(self):
        text = "A synthetic input with no model execution."
        expected = '''### Definition
A term is "the designation of a defined concept in a special language by a linguistic expression" (ISO 1087). A term is a word or multiword expression with a specific meaning in a specialised domain.

### Domain
Finance (banking, accounting, financial markets, financial regulation).

### Rules
- Extract only terms relevant to the specified domain.
- Extract terms exactly as they appear in the text (same word form and spelling).
- If a multiword term contains shorter terms that are also terms on their own, extract both, when permitted by the annotation policy.
- Do NOT extract: named entities (companies, people, places, institutions, products), numbers, dates, monetary amounts, percentages, or general-language words without a specialised meaning in the specified domain.
- If there are no terms, return an empty list.

### Text
A synthetic input with no model execution.

### Output
Return only a JSON object with this format, with no additional text:
{"terms": ["term 1", "term 2"]}'''
        prompt = self.builder.build_basic(text)
        self.assertEqual(prompt.to_messages(), [
            {"role": "system", "content": "You are an expert terminologist specialised in the Finance domain."},
            {"role": "user", "content": expected},
        ])
        self.assertEqual(prompt.metadata.condition, PromptCondition.BASIC)

    def test_special_characters_and_braces_are_preserved_literally(self):
        text = 'Line "one" {"terms": ["NOT an evaluated answer"]}\n\\line\t€ é 漢字'
        prompt = self.builder.build_basic(text)
        self.assertIn(f"### Text\n{text}\n\n### Output", prompt.messages[1].content)
        self.assertEqual(json.loads(prompt.to_json()), prompt.to_messages())

    def test_output_schema_is_a_valid_json_object(self):
        user = self.builder.build_basic("Synthetic input.").messages[1].content
        schema = user.rsplit("\n", 1)[1]
        self.assertEqual(json.loads(schema), {"terms": ["term 1", "term 2"]})

    def test_domain_selection_is_explicit_for_all_acter_domains(self):
        expected = {
            "corruption": "corruption", "equitation_dressage": "equitation/dressage",
            "heart_failure": "heart failure", "wind_energy": "wind energy",
        }
        for key, name in expected.items():
            with self.subTest(domain=key):
                prompt = PromptBuilder(default_prompt_config(key)).build_basic("Synthetic input.")
                self.assertEqual(
                    prompt.messages[0].content,
                    f"You are an expert terminologist specialised in the {name} domain.",
                )
                self.assertNotIn("Finance", prompt.messages[1].content)
                self.assertNotIn("financial", prompt.messages[1].content)
                self.assertIn("only the longest term span", prompt.messages[1].content)
        with self.assertRaises(KeyError):
            default_prompt_config("not-an-authorized-domain-preset")

    def test_custom_domain_description_is_not_reformatted(self):
        domain = DomainSpec("custom", "test domain", 'A "description" with {braces}.\nNext line.')
        config = PromptConfig(domain, AnnotationPolicy())
        user = PromptBuilder(config).build_basic("Synthetic input.").messages[1].content
        self.assertIn(f"### Domain\n{domain.description}\n\n### Rules", user)

    def test_no_query_reference_argument_is_accepted(self):
        with self.assertRaises(TypeError):
            self.builder.build_basic("Synthetic input.", reference_terms=["hidden answer"])
        with self.assertRaises(TypeError):
            self.builder.build_few_shot("Synthetic input.", demonstrations(), gold_terms=["hidden answer"])

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build_basic(" \n")

    def test_exported_messages_are_independent_of_the_immutable_record(self):
        prompt = self.builder.build_basic("Original synthetic text.")
        messages = prompt.to_messages()
        messages[0]["content"] = "changed"
        self.assertNotEqual(prompt.messages[0].content, "changed")


class PatternPromptTests(unittest.TestCase):
    def test_condition_two_adds_only_the_requested_block(self):
        builder = PromptBuilder(default_prompt_config("finance"))
        basic = builder.build_basic("Synthetic input.")
        patterns = builder.build_patterns("Synthetic input.")
        content = patterns.messages[1].content
        start = content.index("### Typical structures of financial terms")
        end = content.index("### Text")
        self.assertEqual(content[:start] + content[end:], basic.messages[1].content)
        self.assertEqual(patterns.messages[0], basic.messages[0])

    def test_corrected_financial_structure_examples(self):
        content = PromptBuilder(default_prompt_config("finance")).build_patterns("Synthetic.").messages[1].content
        lines = set(content.splitlines())
        expected = (
            "N: dividend, liquidity, collateral",
            "N + N: interest rate, asset management, credit risk",
            "A + N: fixed income, deferred tax, sovereign debt",
            "N + N + N: credit default swap, balance sheet item",
            "A + N + N: net asset value", "A + A + N: gross domestic product",
            "N + of + N: cost of capital", "N + of + N + N: statement of cash flows",
            "N + P + N: return on equity", "Other specialised constructions: loss given default",
            "Acronyms and abbreviations: EBITDA, NAV, LTV",
        )
        for item in expected:
            self.assertIn(f"- {item}", lines)
        self.assertNotIn("- A + N + N: net asset value, gross domestic product", lines)
        self.assertNotIn("- N + of + N: cost of capital, statement of cash flows", lines)
        self.assertNotIn("N + P + N (other prepositions): return on equity, loss given default", content)

    def test_acter_uses_general_structures_without_financial_examples(self):
        for key in ("corruption", "equitation_dressage", "heart_failure", "wind_energy"):
            with self.subTest(domain=key):
                content = PromptBuilder(default_prompt_config(key)).build_patterns("Synthetic.").messages[1].content
                self.assertIn("### Typical structures of terms", content)
                self.assertIn("- N + of + N + N\n", content)
                for phrase in ("dividend", "interest rate", "EBITDA", "financial concept"):
                    self.assertNotIn(phrase, content)

    def test_structure_descriptions_are_configurable(self):
        base = default_prompt_config("finance")
        guide = replace(base.structure_guide, structures=("N: synthetic grammatical illustration",))
        config = replace(base, structure_guide=guide)
        content = PromptBuilder(config).build_patterns("Synthetic.").messages[1].content
        self.assertIn("- N: synthetic grammatical illustration", content)
        self.assertNotIn("- N: dividend", content)

    def test_financial_illustrations_cannot_be_assigned_to_acter_by_accident(self):
        finance = default_prompt_config("finance")
        acter = default_prompt_config("heart_failure")
        with self.assertRaises(ValueError):
            replace(acter, structure_guide=finance.structure_guide)


class FewShotPromptTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(default_prompt_config("finance"))

    def test_exactly_three_existing_retrieved_objects_are_consumed_in_order(self):
        supplied = demonstrations(4)
        prompt = self.builder.build_few_shot("Synthetic query.", supplied)
        self.assertEqual(len(example_outputs(prompt.messages[1].content)), 3)
        self.assertEqual(prompt.metadata.retrieved_examples, supplied[:3])
        self.assertIs(prompt.metadata.retrieved_examples[0], supplied[0])
        self.assertIs(prompt.metadata.retrieved_examples[0].example, supplied[0].example)
        self.assertEqual(prompt.metadata.available_examples, 4)
        self.assertTrue(prompt.metadata.complete_few_shot)
        self.assertNotIn(supplied[3].example.text, prompt.messages[1].content)

    def test_text_and_terms_are_literal_and_terms_are_valid_json_lists(self):
        supplied = demonstrations()
        user = self.builder.build_few_shot("Synthetic query.", supplied).messages[1].content
        outputs = example_outputs(user)
        for index, retrieved in enumerate(supplied):
            self.assertIn(f"Text: {retrieved.example.text}\nOutput: ", user)
            self.assertEqual(outputs[index], {"terms": list(retrieved.example.terms)})
            self.assertIsInstance(outputs[index]["terms"], list)

    def test_few_shot_changes_only_the_examples_block(self):
        basic = self.builder.build_basic("Synthetic query.")
        few = self.builder.build_few_shot("Synthetic query.", demonstrations())
        content = few.messages[1].content
        start = content.index("### Examples")
        end = content.index("### Text")
        self.assertEqual(content[:start] + content[end:], basic.messages[1].content)
        self.assertEqual(few.messages[0], basic.messages[0])

    def test_scores_remain_metadata_and_never_change_messages(self):
        supplied = demonstrations()
        changed = tuple(replace(item, score=1000.123 + index) for index, item in enumerate(supplied))
        a = self.builder.build_few_shot("Synthetic query.", supplied)
        b = self.builder.build_few_shot("Synthetic query.", changed)
        self.assertEqual(a.to_messages(), b.to_messages())
        self.assertNotIn("1000.123", b.to_json())
        self.assertEqual(b.metadata.to_dict()["retrieved_examples"][0]["score"], 1000.123)

    def test_insufficient_examples_raise_instead_of_fabricating_demonstrations(self):
        for count in (0, 1, 2):
            with self.subTest(available=count):
                with self.assertRaises(InsufficientExamplesError) as caught:
                    self.builder.build_few_shot("Synthetic query.", demonstrations(count))
                self.assertEqual(caught.exception.required, 3)
                self.assertEqual(caught.exception.available, count)
                self.assertIn("No complete few-shot prompt was built", str(caught.exception))

    def test_few_shot_count_is_configurable(self):
        config = replace(default_prompt_config("finance"), few_shot_examples=2)
        prompt = PromptBuilder(config).build_few_shot("Synthetic query.", demonstrations())
        self.assertEqual(len(example_outputs(prompt.messages[1].content)), 2)
        self.assertTrue(prompt.metadata.complete_few_shot)

    def test_plain_annotations_are_not_mistaken_for_retriever_results(self):
        with self.assertRaises(TypeError):
            self.builder.build_few_shot("Synthetic.", [item.example for item in demonstrations()])
        with self.assertRaises(TypeError):
            self.builder.build_few_shot("Synthetic.", "a path or a string")

    def test_supplied_annotation_policy_mismatches_are_not_relabelled(self):
        provided = demonstrations()
        example = replace(provided[0].example, terms=("Synthetic Institution", "short", "long short"))
        provided = (replace(provided[0], example=example),) + provided[1:]
        builder = PromptBuilder(default_prompt_config("heart_failure"))
        user = builder.build_few_shot("Synthetic query.", provided).messages[1].content
        self.assertIn("only the longest term span", user)
        self.assertEqual(example_outputs(user)[0]["terms"], list(example.terms))

    def test_preannotated_empty_example_is_serialized_without_invention(self):
        supplied = demonstrations()
        empty = replace(supplied[0].example, terms=())
        supplied = (replace(supplied[0], example=empty),) + supplied[1:]
        user = self.builder.build_few_shot("Synthetic query.", supplied).messages[1].content
        self.assertEqual(example_outputs(user)[0], {"terms": []})


class HybridPromptTests(unittest.TestCase):
    def test_hybrid_matches_the_supplied_wording(self):
        builder = PromptBuilder(default_prompt_config("finance"))
        expected = '''### Definition
A term is "the designation of a defined concept in a special language by a linguistic expression" (ISO 1087). A term is a word or multiword expression with a specific meaning in a specialised domain.

### Domain
Finance (banking, accounting, financial markets, financial regulation).

### Task
Below is a text and a list of candidate expressions automatically extracted from it using part-of-speech patterns. For each candidate, decide whether it is a term relevant to the specified domain as used in this text.

### Rules
- Accept a candidate only if it designates a specialised concept in this context.
- Reject named entities, numbers, dates, monetary amounts, and general-language expressions.
- Reject candidates with incorrect boundaries (e.g., "the higher interest rate" → reject; the correct term would be "interest rate" if it is in the list).
- Do not add expressions that are not in the candidate list.

### Text
Synthetic input.

### Candidates
["synthetic candidate"]

### Output
Return only a JSON object with this format, with no additional text:
{"terms": ["accepted candidate 1", "accepted candidate 2"]}'''
        prompt = builder.build_hybrid("Synthetic input.", ["synthetic candidate"])
        self.assertEqual(prompt.messages[1].content, expected)
        self.assertEqual(prompt.metadata.condition, PromptCondition.HYBRID)

    def test_candidates_use_json_without_filtering_or_deduplication(self):
        candidates = ['quoted "candidate"', "line\nbreak", "\\path", "{json} € 漢字", "duplicate", "duplicate"]
        original = candidates.copy()
        prompt = PromptBuilder(default_prompt_config("finance")).build_hybrid("Synthetic.", candidates)
        block = prompt.messages[1].content.split("### Candidates\n", 1)[1].split("\n\n### Output", 1)[0]
        self.assertEqual(json.loads(block), candidates)
        self.assertEqual(candidates, original)

    def test_empty_candidates_remain_an_empty_json_list(self):
        user = PromptBuilder(default_prompt_config("finance")).build_hybrid("Synthetic.", []).messages[1].content
        self.assertIn("### Candidates\n[]\n\n### Output", user)

    def test_invalid_candidates_are_rejected_without_generating_replacements(self):
        builder = PromptBuilder(default_prompt_config("finance"))
        with self.assertRaises(TypeError):
            builder.build_hybrid("Synthetic.", "not a candidate list")
        for value in ([""], [23]):
            with self.subTest(candidates=value), self.assertRaises(ValueError):
                builder.build_hybrid("Synthetic.", value)

    def test_nonfinancial_hybrid_does_not_introduce_financial_boundary_examples(self):
        user = PromptBuilder(default_prompt_config("wind_energy")).build_hybrid("Synthetic.", ["candidate"]).messages[1].content
        self.assertNotIn("interest rate", user)
        self.assertIn("- Reject candidates with incorrect boundaries.", user)


class AnnotationPolicyTests(unittest.TestCase):
    def prompts_for(self, config):
        builder = PromptBuilder(config)
        return (
            builder.build_basic("Synthetic."), builder.build_patterns("Synthetic."),
            builder.build_few_shot("Synthetic.", demonstrations()),
            builder.build_hybrid("Synthetic.", ["candidate"]),
        )

    def test_named_entity_policy_is_consistent_in_all_four_conditions(self):
        base = default_prompt_config("finance")
        for exclude in (True, False):
            config = replace(base, policy=replace(base.policy, exclude_named_entities=exclude))
            for prompt in self.prompts_for(config):
                with self.subTest(exclude=exclude, condition=prompt.metadata.condition):
                    self.assertEqual("named entities" in prompt.messages[1].content, exclude)
                    self.assertEqual(prompt.metadata.config.policy.exclude_named_entities, exclude)

    def test_acronym_exclusion_affects_all_conditions_and_removes_the_pattern_row(self):
        base = default_prompt_config("finance")
        config = replace(base, policy=replace(base.policy, include_acronyms=False))
        for prompt in self.prompts_for(config):
            content = prompt.messages[1].content
            self.assertIn("acronyms", content)
            self.assertNotIn("Acronyms and abbreviations: EBITDA, NAV, LTV", content)
            self.assertFalse(prompt.metadata.config.policy.include_acronyms)

    def test_longest_policy_is_used_in_all_comparable_conditions(self):
        for prompt in self.prompts_for(default_prompt_config("heart_failure")):
            self.assertIn("only the longest term span", prompt.messages[1].content)
            self.assertEqual(prompt.metadata.config.policy.nested_terms, NestedTerms.LONGEST_ONLY)

    def test_all_nested_terms_require_explicit_opt_in(self):
        base = default_prompt_config("heart_failure")
        config = replace(base, policy=replace(base.policy, nested_terms=NestedTerms.ALL))
        for prompt in self.prompts_for(config):
            self.assertNotIn("only the longest term span", prompt.messages[1].content)
            self.assertEqual(prompt.metadata.config.policy.nested_terms, NestedTerms.ALL)
        self.assertIn("extract both.", self.prompts_for(config)[0].messages[1].content)
        self.assertIn("- Accept both", self.prompts_for(config)[3].messages[1].content)

    def test_invalid_policy_and_example_count_are_rejected(self):
        with self.assertRaises(ValueError):
            AnnotationPolicy(nested_terms="guess_from_silver")
        with self.assertRaises(TypeError):
            AnnotationPolicy(include_acronyms="false")
        for count in (0, -1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                replace(default_prompt_config("finance"), few_shot_examples=count)


if __name__ == "__main__":
    unittest.main()
