"""the original annotation guidelines’ wording and the explicitly requested grammatical corrections."""

from .models import StructureGuide


DEFINITION = '''### Definition
A term is "the designation of a defined concept in a special language by a linguistic expression" (ISO 1087). A term is a word or multiword expression with a specific meaning in a specialised domain.'''

BASIC_OUTPUT = '''### Output
Return only a JSON object with this format, with no additional text:
{"terms": ["term 1", "term 2"]}'''

HYBRID_TASK = '''### Task
Below is a text and a list of candidate expressions automatically extracted from it using part-of-speech patterns. For each candidate, decide whether it is a term relevant to the specified domain as used in this text.'''

HYBRID_OUTPUT = '''### Output
Return only a JSON object with this format, with no additional text:
{"terms": ["accepted candidate 1", "accepted candidate 2"]}'''

CONDITIONAL_NESTING = (
    "- If a multiword term contains shorter terms that are also terms on their own, "
    "extract both, when permitted by the annotation policy."
)
ALL_NESTING = (
    "- If a multiword term contains shorter terms that are also terms on their own, "
    "extract both."
)
LONGEST_NESTING = (
    "- For nested terms, extract only the longest term span. Extract a shorter term "
    "only if it also occurs independently in the text."
)

# These are illustrative English descriptions, never token patterns or matchers.
FINANCIAL_STRUCTURES = StructureGuide(
    title="Typical structures of financial terms",
    introduction=(
        "Financial terms in English frequently (but not exclusively) follow these patterns:"
    ),
    structures=(
        "N: dividend, liquidity, collateral",
        "N + N: interest rate, asset management, credit risk",
        "A + N: fixed income, deferred tax, sovereign debt",
        "N + N + N: credit default swap, balance sheet item",
        "A + N + N: net asset value",
        "A + A + N: gross domestic product",
        "N + of + N: cost of capital",
        "N + of + N + N: statement of cash flows",
        "N + P + N: return on equity",
        "Other specialised constructions: loss given default",
    ),
    acronym_structure="Acronyms and abbreviations: EBITDA, NAV, LTV",
    conclusion=(
        "Use these patterns to help identify term boundaries, but extract only "
        "expressions that designate a financial concept in this context."
    ),
    domain_key="finance",
)

GENERAL_STRUCTURES = StructureGuide(
    title="Typical structures of terms",
    introduction=(
        "Terms in English frequently (but not exclusively) follow these patterns:"
    ),
    structures=(
        "N", "N + N", "A + N", "N + N + N", "A + N + N", "A + A + N",
        "N + of + N", "N + of + N + N", "N + P + N", "Other specialised constructions",
    ),
    acronym_structure="Acronyms and abbreviations",
    conclusion=(
        "Use these patterns to help identify term boundaries, but extract only "
        "expressions that designate a specialised concept in this context."
    ),
)
