"""Outcome normalization helpers for reliability metrics."""

from __future__ import annotations

from typing import Any


def outcome_binary_from_outcome(outcome: dict[str, Any]) -> int | None:
    """Infer normalized binary outcome (1 pass / 0 fail) from sidecar payload."""
    passed = outcome.get("pass")
    if isinstance(passed, bool):
        return 1 if passed else 0

    # Prefer well-known label-style scorer output (e.g. GAIA C/I).
    for value in outcome.values():
        mapped = outcome_binary_from_value(value)
        if mapped is not None:
            return mapped
    return None


def outcome_label_from_outcome(outcome: dict[str, Any]) -> str | None:
    """Extract canonical textual outcome labels when available."""
    value = outcome.get("gaia_scorer")
    if isinstance(value, str):
        label = value.strip().upper()
        if label in {"C", "I"}:
            return label
    return None


def outcome_binary_from_value(value: Any) -> int | None:
    """Map a single scorer value to binary pass/fail when semantics are clear."""
    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
        return None

    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"C", "CORRECT", "PASS", "TRUE", "YES"}:
            return 1
        if token in {"I", "INCORRECT", "FAIL", "FALSE", "NO"}:
            return 0
        try:
            numeric = float(token)
        except ValueError:
            return None
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    return None
