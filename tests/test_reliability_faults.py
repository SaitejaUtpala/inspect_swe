import pytest
from inspect_ai.model import (
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_swe.reliability.baseline import (
    RELIABILITY_CLAUDE_CODE_VERSION,
    RELIABILITY_CODEX_CLI_VERSION,
    _benchmark_log_slug,
    _default_solver_for_agent,
)
from inspect_swe.reliability.fault import (
    _codex_cli_fault_kwargs,
    _default_fault_solver_for_agent,
)
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
async def test_exec_observation_error_faults_claude_bash_tool_message() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context(agent="claude_code"))
    model = _CapturingModel()
    observation = ChatMessageTool(
        content="17054.888019907572 17.054888019907573 17000\n",
        function="Bash",
        tool_call_id="call_bash",
    )

    result = await env.model_filter()(
        model,
        [ChatMessageUser(content="Run the calculation."), observation],
        [],
        None,
        GenerateConfig(),
    )

    assert isinstance(result, ModelOutput)
    assert model.last_input is not None
    faulted_observation = model.last_input[-1]
    assert isinstance(faulted_observation, ChatMessageTool)
    assert faulted_observation.function == "Bash"
    assert faulted_observation.tool_call_id == "call_bash"
    assert "Command: /bin/bash -lc <redacted>" in faulted_observation.text
    assert "Process exited with code 254" in faulted_observation.text
    assert "bash: fork: Resource temporarily unavailable" in faulted_observation.text
    assert "17054.888019907572" not in faulted_observation.text


@pytest.mark.anyio
async def test_exec_observation_error_refaults_replayed_claude_bash_history() -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context(agent="claude_code"))
    model = _CapturingModel()
    first_bash = ChatMessageTool(
        content="attachment://bash-output",
        function="Bash",
        tool_call_id="toolu_existing",
    )

    first_result = await env.model_filter()(
        model,
        [ChatMessageUser(content="Calculate."), first_bash],
        [],
        None,
        GenerateConfig(),
    )

    assert isinstance(first_result, ModelOutput)
    replayed_bash = ChatMessageTool(
        content="17054.888019907572 17.054888019907573 17",
        function="Bash",
        tool_call_id="toolu_existing",
    )
    next_bash = ChatMessageTool(
        content="attachment://second-bash-output",
        function="Bash",
        tool_call_id="toolu_next",
    )

    second_result = await env.model_filter()(
        model,
        [
            ChatMessageUser(content="Calculate."),
            replayed_bash,
            ChatMessageUser(content="Try again."),
            next_bash,
        ],
        [],
        None,
        GenerateConfig(),
    )

    assert isinstance(second_result, ModelOutput)
    assert model.last_input is not None
    faulted_bash_observations = [
        message
        for message in model.last_input
        if isinstance(message, ChatMessageTool) and message.function == "Bash"
    ]
    assert len(faulted_bash_observations) == 2
    assert all(
        "bash: fork: Resource temporarily unavailable" in message.text
        for message in faulted_bash_observations
    )
    assert all(
        "17054.888019907572" not in message.text
        for message in faulted_bash_observations
    )
    assert len(env.applied_records) == 2


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
@pytest.mark.parametrize("function", ["WebSearch", "WebFetch", "web_search"])
async def test_exec_observation_error_ignores_non_shell_tools(function: str) -> None:
    spec = FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)
    env = FaultEnvironment([spec], _fault_context(agent="claude_code"))
    model = _CapturingModel()

    result = await env.model_filter()(
        model,
        [
            ChatMessageUser(content="Search."),
            ChatMessageTool(
                content="Search or fetch result text",
                function=function,
                tool_call_id=f"call_{function}",
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
async def test_exec_observation_error_refaults_without_resampling_seen_tool_call() -> None:
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
    assert isinstance(second, ModelOutput)
    assert len(env.applied_records) == 1
    assert model.generate_calls == 2
    assert model.last_input is not None
    faulted_observation = model.last_input[-1]
    assert isinstance(faulted_observation, ChatMessageTool)
    assert "bash: fork: Resource temporarily unavailable" in faulted_observation.text
    assert "\nok\n" not in faulted_observation.text


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
    assert kwargs["version"] == RELIABILITY_CODEX_CLI_VERSION
    assert "disallowed_tools" not in kwargs
    assert "bridged_tools" not in kwargs


def test_baseline_codex_default_uses_fresh_default_constructor(monkeypatch) -> None:
    calls = {}

    def fake_codex_cli(**kwargs):
        calls.update(kwargs)
        return object()

    import inspect_swe

    monkeypatch.setattr(inspect_swe, "codex_cli", fake_codex_cli)

    assert _default_solver_for_agent("codex_cli") is not None
    assert calls["version"] == RELIABILITY_CODEX_CLI_VERSION


def test_baseline_claude_default_uses_fresh_default_constructor(monkeypatch) -> None:
    calls = {}

    def fake_claude_code(**kwargs):
        calls.update(kwargs)
        return object()

    import inspect_swe

    monkeypatch.setattr(inspect_swe, "claude_code", fake_claude_code)

    assert _default_solver_for_agent("claude_code") is not None
    assert calls == {"version": RELIABILITY_CLAUDE_CODE_VERSION}


def test_fault_claude_default_passes_filter_to_bridge_agent(monkeypatch) -> None:
    calls = {}

    def fake_claude_code(**kwargs):
        calls.update(kwargs)
        return object()

    import inspect_swe

    monkeypatch.setattr(inspect_swe, "claude_code", fake_claude_code)
    env = FaultEnvironment(
        [FaultSpec(surface="message", mode="exec_observation_error", probability=1.0)],
        _fault_context(agent="claude_code"),
    )

    assert _default_fault_solver_for_agent("claude_code", env, FaultPhaseConfig()) is not None
    assert callable(calls["filter"])
    assert calls["version"] == RELIABILITY_CLAUDE_CODE_VERSION


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


def _fault_context(agent: str = "codex_cli") -> FaultContext:
    return FaultContext(
        campaign_id="campaign",
        phase="fault",
        agent=agent,
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
