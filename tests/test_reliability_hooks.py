from __future__ import annotations

from pathlib import Path

import pytest
from inspect_swe.reliability.hooks import (
    ReliabilityHookConfig,
    ReliabilityHooks,
    assert_reliability_hooks_active,
    configure_reliability_hooks,
    disable_reliability_hooks,
)


def test_reliability_hooks_registered() -> None:
    assert_reliability_hooks_active(require_enabled=False)


def test_reliability_hooks_activation_requires_config(tmp_path: Path) -> None:
    disable_reliability_hooks()
    with pytest.raises(RuntimeError):
        assert_reliability_hooks_active(require_enabled=True)

    configure_reliability_hooks(
        ReliabilityHookConfig(
            enabled=True,
            sidecar_path=str(tmp_path / "records.jsonl"),
        )
    )
    assert_reliability_hooks_active(require_enabled=True)
    disable_reliability_hooks()


def test_identity_agent_prefers_reliability_metadata() -> None:
    hooks = ReliabilityHooks()
    hooks._agent_by_eval_id["eval-1"] = "openai/gpt-5.4-2026-03-05"

    agent = hooks._identity_agent(
        {"reliability_agent": "codex_cli"},
        "eval-1",
    )

    assert agent == "codex_cli"


def test_outcome_payload_maps_gaia_labels_to_pass_boolean() -> None:
    hooks = ReliabilityHooks()
    correct = hooks._outcome_payload({"gaia_scorer": "C"})
    incorrect = hooks._outcome_payload({"gaia_scorer": "I"})

    assert correct["pass"] is True
    assert incorrect["pass"] is False


def test_outcome_payload_keeps_unknown_strings_unmapped() -> None:
    hooks = ReliabilityHooks()
    payload = hooks._outcome_payload({"gaia_scorer": "UNKNOWN"})

    assert payload["pass"] is None
