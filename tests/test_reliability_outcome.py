from __future__ import annotations

from inspect_swe.reliability.outcome import (
    outcome_binary_from_outcome,
    outcome_binary_from_value,
    outcome_label_from_outcome,
)


def test_outcome_binary_from_value_for_common_patterns() -> None:
    assert outcome_binary_from_value(1) == 1
    assert outcome_binary_from_value(0) == 0
    assert outcome_binary_from_value("C") == 1
    assert outcome_binary_from_value("I") == 0
    assert outcome_binary_from_value("1") == 1
    assert outcome_binary_from_value("0") == 0
    assert outcome_binary_from_value("unknown") is None


def test_outcome_binary_from_outcome_prefers_explicit_pass() -> None:
    assert outcome_binary_from_outcome({"pass": True, "gaia_scorer": "I"}) == 1
    assert outcome_binary_from_outcome({"pass": False, "gaia_scorer": "C"}) == 0


def test_outcome_label_from_outcome_extracts_gaia_labels() -> None:
    assert outcome_label_from_outcome({"gaia_scorer": "C"}) == "C"
    assert outcome_label_from_outcome({"gaia_scorer": "i"}) == "I"
    assert outcome_label_from_outcome({"gaia_scorer": "x"}) is None
