"""
Verifier: checks predicted output tokens against labels computed
from the composition function. Labels are generated programmatically —
no human annotation required.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .entity import Entity
from .composition import compose, validate_binding, InvalidBinding
from .generator import serialize_compound_output, Example
from .vocab import Token


@dataclass
class VerificationResult:
    is_valid_correct: bool          # model's is_valid prediction matched ground truth
    output_correct: Optional[bool]  # token sequence correct (None if invalid example)
    predicted_tokens: List[int]
    expected_tokens: List[int]
    details: str


def verify(example: Example, predicted_tokens: List[int], predicted_is_valid: bool) -> VerificationResult:
    """
    Verify a model's predictions against ground truth.

    Args:
        example: the original example (contains entities and ground truth)
        predicted_tokens: the model's predicted target token sequence
        predicted_is_valid: the model's is_valid prediction
    """
    is_valid_correct = (predicted_is_valid == example.is_valid)

    if not example.is_valid:
        return VerificationResult(
            is_valid_correct=is_valid_correct,
            output_correct=None,
            predicted_tokens=predicted_tokens,
            expected_tokens=[Token.END],
            details="invalid example — only is_valid checked",
        )

    expected = example.target_tokens
    output_correct = (predicted_tokens == expected)

    details = "correct" if output_correct else _diff_details(predicted_tokens, expected)

    return VerificationResult(
        is_valid_correct=is_valid_correct,
        output_correct=output_correct,
        predicted_tokens=predicted_tokens,
        expected_tokens=expected,
        details=details,
    )


def _diff_details(predicted: List[int], expected: List[int]) -> str:
    if len(predicted) != len(expected):
        return f"length mismatch: predicted {len(predicted)}, expected {len(expected)}"
    diffs = [i for i, (p, e) in enumerate(zip(predicted, expected)) if p != e]
    return f"{len(diffs)} token(s) differ at positions {diffs[:8]}"


def batch_accuracy(examples: List[Example], predictions: List[Tuple[List[int], bool]]) -> dict:
    """
    Compute accuracy metrics over a batch.

    predictions: list of (predicted_tokens, predicted_is_valid)
    """
    results = [verify(ex, tok, iv) for ex, (tok, iv) in zip(examples, predictions)]

    valid_examples = [r for r, ex in zip(results, examples) if ex.is_valid]
    invalid_examples = [r for r, ex in zip(results, examples) if not ex.is_valid]

    return {
        "is_valid_acc": sum(r.is_valid_correct for r in results) / len(results),
        "output_acc": (
            sum(r.output_correct for r in valid_examples) / len(valid_examples)
            if valid_examples else None
        ),
        "valid_count": len(valid_examples),
        "invalid_count": len(invalid_examples),
    }
