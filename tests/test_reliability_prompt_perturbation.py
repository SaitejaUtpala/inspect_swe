from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from inspect_ai.model import GenerateInput
from inspect_swe.reliability.perturbations.prompt import PromptPerturbationSpec


@pytest.mark.anyio
async def test_prompt_rewrite_filter_rewrites_first_user_message() -> None:
    spec = PromptPerturbationSpec(
        enabled=True,
        mode="rewrite_v1",
        variant_count=3,
        seed=9,
    )
    generate_filter = spec.build_generate_filter(default_seed=0)
    assert generate_filter is not None

    model = SimpleNamespace(name="test-model")
    messages = [
        SimpleNamespace(role="system", content="be helpful"),
        SimpleNamespace(role="user", content="solve this task"),
    ]

    output = await generate_filter(model, messages, [], None, None)
    assert isinstance(output, GenerateInput)
    assert output.input[1].content != "solve this task"
    assert output.input[1].content.startswith("[INSPECT_SWE_PROMPT_VARIANT_")


@pytest.mark.anyio
async def test_prompt_rewrite_noop_returns_none() -> None:
    spec = PromptPerturbationSpec(enabled=True, mode="noop", variant_count=2)
    generate_filter = spec.build_generate_filter(default_seed=3)
    assert generate_filter is None


@pytest.mark.anyio
async def test_prompt_rewrite_llm_mode_uses_inspect_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRewriter:
        async def generate(self, input: Any, config: Any) -> Any:
            return SimpleNamespace(completion="Please solve this task carefully.")

    monkeypatch.setattr(
        "inspect_swe.reliability.perturbations.prompt.get_model",
        lambda model_name: _FakeRewriter(),
    )

    spec = PromptPerturbationSpec(
        enabled=True,
        mode="rewrite_llm_v1",
        variant_count=2,
        seed=7,
        rewrite_model="openai/gpt-4o-mini",
    )
    generate_filter = spec.build_generate_filter(default_seed=0)
    assert generate_filter is not None

    messages = [
        SimpleNamespace(role="system", content="be helpful"),
        SimpleNamespace(role="user", content="solve this task"),
    ]
    output = await generate_filter("openai/gpt-5.4", messages, [], None, None)
    assert isinstance(output, GenerateInput)
    assert output.input[1].content == "Please solve this task carefully."


@pytest.mark.anyio
async def test_prompt_rewrite_llm_mode_falls_back_on_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRewriter:
        async def generate(self, input: Any, config: Any) -> Any:
            return SimpleNamespace(completion="ok")

    monkeypatch.setattr(
        "inspect_swe.reliability.perturbations.prompt.get_model",
        lambda model_name: _FakeRewriter(),
    )

    spec = PromptPerturbationSpec(
        enabled=True,
        mode="rewrite_llm_v1",
        seed=7,
        rewrite_model="openai/gpt-4o-mini",
    )
    generate_filter = spec.build_generate_filter(default_seed=0)
    assert generate_filter is not None

    original = "solve this task with exactly one number as the answer"
    messages = [SimpleNamespace(role="user", content=original)]
    output = await generate_filter("openai/gpt-5.4", messages, [], None, None)
    assert output is None
