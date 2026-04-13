"""Shared reliability signal extraction utilities."""

from __future__ import annotations

import re
from typing import Any, Iterable

_CONFIDENCE_REGEX = re.compile(
    r"\bconfidence\b[^0-9]{0,16}([0-9]{1,3}(?:\.[0-9]+)?)\s*(%?)",
    re.IGNORECASE,
)
_ABSTENTION_REGEX = re.compile(
    r"\b(i\s+(?:cannot|can't|can not|won't|will not)|unable|insufficient information|"
    r"not enough information|uncertain|unsure|can't determine|cannot determine|"
    r"cannot complete|cannot solve|refuse|decline)\b",
    re.IGNORECASE,
)


def extract_confidence(
    metadata: dict[str, Any],
    *,
    metadata_keys: Iterable[str],
    text: str | None = None,
) -> tuple[float | None, str | None]:
    for key in metadata_keys:
        value = normalize_confidence(to_float(metadata.get(key)))
        if value is not None:
            return value, key

    text_value = extract_confidence_from_text(text)
    if text_value is not None:
        return text_value, "output_text_confidence"
    return None, None


def extract_confidence_from_text(text: str | None) -> float | None:
    if not text:
        return None
    match = _CONFIDENCE_REGEX.search(text)
    if not match:
        return None
    numeric = to_float(match.group(1))
    if numeric is None:
        return None
    percent_marker = match.group(2)
    if percent_marker == "%":
        numeric = numeric / 100.0
    return normalize_confidence(numeric)


def extract_safety_violation(
    metadata: dict[str, Any],
    *,
    metadata_prefix: str,
) -> tuple[bool | None, str | None]:
    for key, value in metadata.items():
        if not key.startswith(metadata_prefix):
            continue
        lowered = key.lower()
        bool_value = to_bool(value)
        if bool_value is None:
            continue
        if "violation" in lowered or "unsafe" in lowered or "harm" in lowered:
            return bool_value, key
        if lowered.endswith("safe"):
            return (not bool_value), key
    return None, None


def extract_abstention(
    metadata: dict[str, Any],
    *,
    metadata_prefix: str,
    text: str | None = None,
) -> tuple[bool | None, str | None]:
    for key, value in metadata.items():
        if not key.startswith(metadata_prefix):
            continue
        lowered = key.lower()
        bool_value = to_bool(value)
        if bool_value is None:
            continue
        if "abstain" in lowered or "refus" in lowered or "uncertain" in lowered:
            return bool_value, key

    text_signal = detect_abstention_from_text(text)
    if text_signal is not None:
        return text_signal, "output_text_abstention"
    return None, None


def detect_abstention_from_text(text: str | None) -> bool | None:
    if not text:
        return None
    if _ABSTENTION_REGEX.search(text):
        return True
    return None


def normalize_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1.0:
        if value <= 100.0 and value >= 5.0:
            value = value / 100.0
        else:
            value = 1.0
    return max(0.0, min(1.0, value))


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
    return None
