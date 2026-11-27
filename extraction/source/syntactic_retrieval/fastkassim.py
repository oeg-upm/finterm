"""Real upstream FastKASSIM LTK, without constructing its bundled parser.

Sources pinned in pyproject.toml and documented in 01_implementacion_fastkassim.md.
The tree kernel is imported from fkassim, not reimplemented or approximated here.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from .models import ParsedText


class FastKassimUnavailableError(RuntimeError):
    """The optional upstream implementation or a numerical dependency is absent."""


class SimilarityComputationError(RuntimeError):
    """The real kernel could not return a usable score."""


@dataclass(frozen=True, slots=True)
class FastKassimConfig:
    metric: str = "LTK"
    sigma: int = 1
    lmbda: float = 0.4
    average: bool = False

    def __post_init__(self) -> None:
        if self.metric != "LTK":
            raise ValueError("This component implements FastKASSIM LTK only")
        if type(self.sigma) is not int or self.sigma not in (0, 1):
            raise ValueError("sigma must be 0 or 1")
        if (
            isinstance(self.lmbda, bool)
            or not isinstance(self.lmbda, (int, float))
            or not math.isfinite(self.lmbda)
            or not 0 < self.lmbda <= 1
        ):
            raise ValueError("lmbda must be finite and in (0, 1]")
        if type(self.average) is not bool:
            raise TypeError("average must be a boolean")


def checked_score(value: float) -> float:
    if isinstance(value, bool):
        raise SimilarityComputationError("A boolean is not a similarity score")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimilarityComputationError("Similarity must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise SimilarityComputationError("Similarity must be finite and nonnegative")
    # Preserve upstream values; do not clamp, symmetrize, or invent scores.
    return result


@lru_cache(maxsize=1)
def _dependencies():
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from fkassim.ltk.ltk import LabelTreeKernel
    except ImportError as exc:
        raise FastKassimUnavailableError(
            "FastKASSIM LTK is unavailable. Prepare the optional 'syntax' "
            "dependencies in the authorized execution environment. "
            "There is no lexical or embedding fallback."
        ) from exc
    return LabelTreeKernel, np, linear_sum_assignment


class FastKassimSimilarity:
    """Score query/example forests using the original LabelTreeKernel.

    The official ATE code calls compute_similarity_preparsed(reference, query).
    We retain that example-first orientation, including for multi-sentence input.
    Default document aggregation is maximum-weight assignment / max(tree counts),
    exactly as in upstream Kassim.syntax_similarity_two_parsed_documents.
    """

    def __init__(self, config: FastKassimConfig | None = None) -> None:
        self.config = config if config is not None else FastKassimConfig()

    def score(self, query: ParsedText, example: ParsedText) -> float:
        if query.parser_id != example.parser_id:
            raise ValueError("Query and example must use the same parser identity")
        kernel, np, assign = _dependencies()
        if kernel.include_leaves is not False:
            raise SimilarityComputationError("Upstream LTK must exclude lexical leaves")

        reference_trees = [tree.to_nltk() for tree in example.trees]
        query_trees = [tree.to_nltk() for tree in query.trees]
        try:
            matrix = np.array(
                [
                    [
                        checked_score(
                            kernel.kernel(
                                reference,
                                candidate,
                                sigma=self.config.sigma,
                                lmbda=self.config.lmbda,
                                use_new_delta=True,
                            )
                        )
                        for candidate in query_trees
                    ]
                    for reference in reference_trees
                ],
                dtype=float,
            )
            if self.config.average:
                result = matrix.mean()
            else:
                rows, columns = assign(matrix, maximize=True)
                result = matrix[rows, columns].sum() / max(matrix.shape)
        except (ArithmeticError, RecursionError, ValueError) as exc:
            raise SimilarityComputationError("FastKASSIM LTK computation failed") from exc
        return checked_score(result)
