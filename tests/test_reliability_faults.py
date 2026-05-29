import pytest
from inspect_ai.model import (
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_swe.reliability.baseline import (
    _benchmark_log_slug,
    _default_solver_for_agent,
)
from inspect_swe.reliability.fault import _codex_cli_fault_kwargs
from inspect_swe.reliability.faults import (
    FaultContext,
    FaultEnvironment,
    FaultPhaseConfig,
    FaultSpec,
)
from inspect_swe.reliability.metrics import compute_fault_metrics


def test_fault_decisions_are_deterministic() -> None:
    spec = FaultSpec(surface="message", mode="append_instruction", probability=0.5)
    env_a = FaultEnvironment([spec], _fault_context())
    env_b = FaultEnvironment([spec], _fault_context())

    assert env_a._maybe_apply(spec, "message:0") == env_b._maybe_apply(spec, "message:0")


def test_fault_probability_zero_and_one_are_exact() -> None:
    context = _fault_context()
    zero = FaultSpec(surface="message", mode="append_instruction", probability=0.0)
    one = FaultSpec(surface="message", mode="append_instruction", probability=1.0)

    assert FaultEnvironment([zero], context)._maybe_apply(zero, "site") is None
    assert FaultEnvironment([one], context)._maybe_apply(one, "site") is not None


@pytest.mark.anyio
async def test_exec_observation_error_calls_model_with_faulted_tool_message() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context())
    model = _CapturingModel()
    observation = ChatMessageTool(
        content=(
            "Command: /bin/bash -lc ls\n"
            "Wall time: 0.0000 seconds\n"
            "Process exited with code 0\n"
            "Original token count: 20\n"
            "Output:\n"
            "bin\netc\n"
        ),
        function="exec_command",
        tool_call_id="call_exec",
    )

    result = await env.model_filter()(
        model,
        [ChatMessageUser(content="List files."), observation],
        [],
        None,
        GenerateConfig(),
    )

    assert isinstance(result, ModelOutput)
    assert model.last_input is not None
    faulted_observation = model.last_input[-1]
    assert isinstance(faulted_observation, ChatMessageTool)
    assert faulted_observation.tool_call_id == "call_exec"
    assert "Command: /bin/bash -lc ls" in faulted_observation.text
    assert "Process exited with code 254" in faulted_observation.text
    assert "bash: fork: Resource temporarily unavailable" in faulted_observation.text
    assert "bin\netc" not in faulted_observation.text
    assert "Process exited with code 0" in observation.text


@pytest.mark.anyio
async def test_exec_observation_error_ignores_non_exec_tool_observation() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context())
    model = _CapturingModel()

    result = await env.model_filter()(
        model,
        [
            ChatMessageUser(content="Search."),
            ChatMessageTool(
                content="Search result text",
                function="web_search",
                tool_call_id="call_web",
            ),
        ],
        [],
        None,
        GenerateConfig(),
    )

    assert result is None
    assert model.last_input is None
    assert env.applied_records == []


@pytest.mark.anyio
async def test_exec_observation_error_probability_zero_never_applies() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=0.0)
    env = FaultEnvironment([spec], _fault_context())
    model = _CapturingModel()

    result = await env.model_filter()(
        model,
        [
            ChatMessageUser(content="List files."),
            ChatMessageTool(
                content="Command: /bin/bash -lc ls\nProcess exited with code 0\nOutput:\nok\n",
                function="exec_command",
                tool_call_id="call_exec",
            ),
        ],
        [],
        None,
        GenerateConfig(),
    )

    assert result is None
    assert model.last_input is None
    assert env.applied_records == []


@pytest.mark.anyio
async def test_exec_observation_error_does_not_resample_seen_tool_call() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context())
    model = _CapturingModel()
    messages = [
        ChatMessageUser(content="List files."),
        ChatMessageTool(
            content="Command: /bin/bash -lc ls\nProcess exited with code 0\nOutput:\nok\n",
            function="exec_command",
            tool_call_id="call_exec",
        ),
    ]

    first = await env.model_filter()(model, messages, [], None, GenerateConfig())
    second = await env.model_filter()(model, messages, [], None, GenerateConfig())

    assert isinstance(first, ModelOutput)
    assert second is None
    assert len(env.applied_records) == 1
    assert model.generate_calls == 1


def test_codex_fault_kwargs_do_not_replace_web_search_by_default() -> None:
    kwargs = _codex_cli_fault_kwargs(
        FaultEnvironment([], _fault_context()),
        FaultPhaseConfig(
            faults=[
                FaultSpec(
                    surface="message",
                    mode="exec_observation_error",
                    probability=1.0,
                )
            ]
        ),
    )

    assert kwargs["retry_refusals"] == 3
    assert "version" not in kwargs
    assert "disallowed_tools" not in kwargs
    assert "bridged_tools" not in kwargs


def test_codex_fault_kwargs_replaces_native_web_search_only_when_enabled() -> None:
    kwargs = _codex_cli_fault_kwargs(
        FaultEnvironment([], _fault_context()),
        FaultPhaseConfig(replace_native_web_search=True),
    )

    assert "version" not in kwargs
    assert kwargs["disallowed_tools"] == ["web_search"]
    assert kwargs["bridged_tools"][0].name == "reliability_search"


def test_baseline_codex_default_uses_fresh_default_constructor(monkeypatch) -> None:
    calls = {}

    def fake_codex_cli(**kwargs):
        calls.update(kwargs)
        return object()

    import inspect_swe

    monkeypatch.setattr(inspect_swe, "codex_cli", fake_codex_cli)

    assert _default_solver_for_agent("codex_cli") is not None
    assert "version" not in calls


def test_benchmark_log_slug_keeps_file_task_refs_under_log_root() -> None:
    slug = _benchmark_log_slug("/tmp/site-packages/inspect_evals/gaia/gaia.py@gaia_level1")

    assert slug == "gaia@gaia_level1"
    assert not slug.startswith("/")


def test_compute_fault_metrics_sets_robustness_ratio() -> None:
    baseline = [_view("a", 1.0), _view("b", 1.0)]
    fault = [_view("a", 1.0), _view("b", 0.0)]

    metrics = compute_fault_metrics(
        baseline_views=baseline,
        baseline_eval_paths=[],
        fault_views=fault,
        fault_eval_paths=[],
    )

    assert metrics["baseline_acc"] == 1.0
    assert metrics["fault_acc"] == 0.5
    assert metrics["robustness_fault_injection"] == 0.5


class _CapturingModel:
    name = "mock/model"

    def __init__(self) -> None:
        self.last_input = None
        self.generate_calls = 0

    async def generate(self, *, input, tools, tool_choice, config):
        del tools, tool_choice, config
        self.generate_calls += 1
        self.last_input = input
        return ModelOutput.from_content(self.name, "generated after exec fault")


def _fault_context() -> FaultContext:
    return FaultContext(
        campaign_id="campaign",
        phase="fault",
        agent="codex_cli",
        repeat_id=0,
        sample_id="sample",
        seed=0,
    )


def _view(sample_id: str, success: float):
    from inspect_swe.reliability.eval_view import BaselineSampleView

    return BaselineSampleView(
        eval_set_id="eval",
        run_id="run",
        task="task",
        sample_id=sample_id,
        sample_uuid=sample_id,
        phase="fault",
        agent="agent",
        scores={"score": success},
    )
