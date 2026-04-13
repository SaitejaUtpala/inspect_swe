"""Inspect-native reliability scorers."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, mean, scorer

from .signals import extract_abstention, extract_confidence, extract_safety_violation


@scorer(metrics=[mean()], name="reliability_confidence_signal")
def reliability_confidence_signal(
    *,
    metadata_keys: tuple[str, ...] = (
        "reliability_confidence",
        "confidence",
        "self_confidence",
    ),
) -> Scorer:
    """Emit normalized confidence from sample metadata."""

    async def score(state, target) -> Score | None:
        del target
        confidence, source_key = extract_confidence(
            state.metadata, metadata_keys=metadata_keys
        )
        if confidence is None:
            return None
        return Score(
            value=confidence,
            metadata={
                "confidence": confidence,
                "source_key": source_key,
            },
        )

    return score


@scorer(metrics=[mean()], name="reliability_safety_violation")
def reliability_safety_violation(
    *,
    metadata_prefix: str = "reliability_safety_",
) -> Scorer:
    """Emit binary safety-violation signal from sample metadata."""

    async def score(state, target) -> Score | None:
        del target
        violation, source_key = extract_safety_violation(
            state.metadata, metadata_prefix=metadata_prefix
        )
        if violation is None:
            return None
        return Score(
            value=1 if violation else 0,
            metadata={
                "violation": violation,
                "source_key": source_key,
            },
        )

    return score


@scorer(metrics=[mean()], name="reliability_abstention_signal")
def reliability_abstention_signal(
    *,
    metadata_prefix: str = "reliability_abstention_",
) -> Scorer:
    """Emit binary abstention signal from sample metadata."""

    async def score(state, target) -> Score | None:
        del target
        abstained, source_key = extract_abstention(
            state.metadata, metadata_prefix=metadata_prefix
        )
        if abstained is None:
            return None
        return Score(
            value=1 if abstained else 0,
            metadata={
                "abstained": abstained,
                "source_key": source_key,
            },
        )

    return score
