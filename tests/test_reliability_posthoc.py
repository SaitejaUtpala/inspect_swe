from __future__ import annotations

import asyncio
from types import SimpleNamespace

from inspect_ai.model import ModelOutput
from inspect_swe.reliability.posthoc import (
    GAIA_ANSWER_FORMAT_INSTRUCTIONS,
    GAIA_ANSWER_REPAIR,
    apply_posthoc_repair,
    detect_dataset,
    repair_gaia_answer,
    resolve_posthoc_repair,
)
from inspect_swe.reliability.posthoc import gaia as gaia_module


def test_detect_dataset_maps_known_benchmarks() -> None:
    assert detect_dataset("/x/inspect_evals/gaia/gaia.py@gaia_level1") == "gaia"
    assert detect_dataset("/x/inspect_evals/tau2/tau2.py@tau2_airline") == "taubench"
    assert detect_dataset("inspect_evals/taubench") == "taubench"
    assert detect_dataset("inspect_evals/swe_bench") is None
    assert detect_dataset(None) is None


def test_resolve_posthoc_repair_scopes_to_gaia_and_claude_code() -> None:
    assert resolve_posthoc_repair("claude_code", "gaia.py@gaia_level1") is GAIA_ANSWER_REPAIR
    # GAIA repair is agent-scoped: Codex is a no-op.
    assert resolve_posthoc_repair("codex_cli", "gaia.py@gaia_level1") is None
    # No repair registered for taubench yet.
    assert resolve_posthoc_repair("claude_code", "tau2.py@tau2_airline") is None
    # Unknown dataset.
    assert resolve_posthoc_repair("claude_code", "swe_bench") is None


def test_apply_posthoc_repair_wraps_only_when_applicable() -> None:
    base = object()

    wrapped = apply_posthoc_repair(
        base, agent="claude_code", benchmark="gaia.py@gaia_level1", enabled=True
    )
    assert wrapped is not base

    # Disabled -> no-op.
    assert (
        apply_posthoc_repair(
            base, agent="claude_code", benchmark="gaia.py@gaia_level1", enabled=False
        )
        is base
    )
    # Codex on GAIA -> no-op.
    assert (
        apply_posthoc_repair(
            base, agent="codex_cli", benchmark="gaia.py@gaia_level1", enabled=True
        )
        is base
    )
    # Claude on Tau2 -> no-op.
    assert (
        apply_posthoc_repair(
            base, agent="claude_code", benchmark="tau2.py@tau2_airline", enabled=True
        )
        is base
    )
    # None base solver stays None.
    assert (
        apply_posthoc_repair(
            None, agent="claude_code", benchmark="gaia.py@gaia_level1", enabled=True
        )
        is None
    )


class _RepairModel:
    name = "mock/model"

    def __init__(self) -> None:
        self.last_input = None

    async def generate(self, input):  # noqa: A002 - match inspect Model API
        self.last_input = input
        return ModelOutput.from_content(self.name, "Paris")


def _fake_state(completion: str):
    return SimpleNamespace(
        output=ModelOutput.from_content("mock/model", completion),
        messages=[],
        metadata={},
    )


def test_repair_gaia_answer_reformats_and_preserves_original(monkeypatch) -> None:
    model = _RepairModel()
    monkeypatch.setattr(gaia_module, "get_model", lambda: model)

    original = "The answer is Paris, since it is the capital of France."
    state = _fake_state(original)

    repaired = asyncio.run(repair_gaia_answer(state))

    assert repaired.output.completion == "Paris"
    repair_meta = repaired.metadata["reliability_posthoc_repair"]
    assert repair_meta["applied"] is True
    assert repair_meta["original_completion"] == original
    assert repair_meta["repaired_completion"] == "Paris"

    # The reformatting turn carries the strict GAIA instructions and is logged.
    prompt_message = model.last_input[-1]
    assert GAIA_ANSWER_FORMAT_INSTRUCTIONS in prompt_message.text
    assert original in prompt_message.text
    assert prompt_message in repaired.messages


def test_repair_gaia_answer_noop_on_empty_output(monkeypatch) -> None:
    called = False

    def _boom():
        nonlocal called
        called = True
        raise AssertionError("model should not be called on empty output")

    monkeypatch.setattr(gaia_module, "get_model", _boom)

    state = _fake_state("   ")
    repaired = asyncio.run(repair_gaia_answer(state))

    assert repaired.output.completion == "   "
    assert "reliability_posthoc_repair" not in repaired.metadata
    assert called is False
