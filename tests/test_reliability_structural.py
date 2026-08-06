from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from inspect_ai.tool import Tool, ToolDef, ToolParams, tool
from inspect_swe.reliability.baseline import RELIABILITY_CODEX_CLI_VERSION
from inspect_swe.reliability.concurrency import OrchestratorConcurrency
from inspect_swe.reliability.spec import ReliabilitySpec
from inspect_swe.reliability.structural import (
    StructuralPhaseConfig,
    _default_structural_solver_for_agent,
    run_structural_phase,
)
from inspect_swe.reliability.structural_perturbations import (
    StructuralContext,
    StructuralEnvironment,
    perturb_gaia_prompt,
    perturb_taubench_tool_params,
    perturb_taubench_tool_result,
    reverse_tool_args,
)
from inspect_swe.reliability.taubench import tau2_airline_bridged_tools


def test_gaia_prompt_perturbation_is_deterministic() -> None:
    prompt = "Return only your answer. What is 1000 plus 2?"

    first, first_records = perturb_gaia_prompt(
        prompt,
        strength="medium",
        seed_key="seed:repeat0:sample0",
    )
    second, second_records = perturb_gaia_prompt(
        prompt,
        strength="medium",
        seed_key="seed:repeat0:sample0",
    )

    assert first == second
    assert first_records == second_records
    assert first != prompt


def test_gaia_severe_prompt_uses_seed_deterministically() -> None:
    prompt = (
        "Return only your answer. Use the short phrase with as few words as "
        "possible. What happened on 2024-05-01 after 2 attempts?"
    )

    first, first_records = perturb_gaia_prompt(
        prompt,
        strength="severe",
        seed_key="seed:repeat0:sample0:severe",
    )
    second, second_records = perturb_gaia_prompt(
        prompt,
        strength="severe",
        seed_key="seed:repeat0:sample0:severe",
    )
    different_seed, _ = perturb_gaia_prompt(
        prompt,
        strength="severe",
        seed_key="seed:repeat0:sample1:severe",
    )

    assert first == second
    assert first_records == second_records
    assert first != different_seed


def test_gaia_strength_presets_apply_expected_transformations() -> None:
    prompt = "Return only your answer.\n\nWhat happened on 2024-05-01?"

    mild, _ = perturb_gaia_prompt(prompt, strength="mild", seed_key="seed")
    medium, _ = perturb_gaia_prompt(prompt, strength="medium", seed_key="seed")
    severe, _ = perturb_gaia_prompt("What is 2?", strength="severe", seed_key="seed")

    assert mild.startswith("return only your answer.")
    assert "\n\n" not in mild
    assert "May 1, 2024" in medium
    assert "weather has been quite variable lately" in severe.lower()


def test_taubench_schema_renames_params_and_preserves_required_fields() -> None:
    params = ToolParams(
        properties={
            "flight_number": {"type": "string"},
            "reservation_id": {"type": "string"},
            "cabin": {"type": "string"},
        },
        required=["flight_number", "reservation_id"],
    )

    updated, mapping, records = perturb_taubench_tool_params(
        params,
        strength="severe",
    )

    assert "fltNo" in updated.properties
    assert "resId" in updated.properties
    assert updated.required == ["fltNo", "resId"]
    assert mapping == {"fltNo": "flight_number", "resId": "reservation_id", "cls": "cabin"}
    assert len(records) == 3


def test_tau2_airline_bridged_tools_exposes_real_airline_tools() -> None:
    bridged_tools = tau2_airline_bridged_tools()
    tool_names = {ToolDef(tool).name for tool in bridged_tools[0].tools}

    assert bridged_tools[0].name == "tau2_airline"
    assert len(tool_names) == 14
    assert {
        "book_reservation",
        "get_user_details",
        "search_direct_flight",
        "get_flight_status",
    } <= tool_names


def test_tau2_airline_bridged_tools_can_wrap_schema_for_structural_env() -> None:
    env = StructuralEnvironment(
        kind="taubench",
        strength="severe",
        context=StructuralContext(campaign_id="campaign", agent="codex_cli", repeat_id=0),
    )

    bridged_tools = tau2_airline_bridged_tools(env)
    status_tool = next(
        tool
        for tool in bridged_tools[0].tools
        if ToolDef(tool).name == "get_flight_status"
    )
    params = ToolDef(status_tool).parameters

    assert set(params.properties) == {"fltNo", "dt"}
    assert params.required == ["fltNo", "dt"]
    assert env.param_mapping["get_flight_status"] == {
        "fltNo": "flight_number",
        "dt": "date",
    }


def test_taubench_arg_reversal_restores_original_names() -> None:
    kwargs = reverse_tool_args(
        "lookup_flight",
        {"fltNo": "HAT001", "dt": "2024-05-01"},
        {"lookup_flight": {"fltNo": "flight_number", "dt": "date"}},
    )

    assert kwargs == {"flight_number": "HAT001", "date": "2024-05-01"}


def test_taubench_response_perturbation_changes_json_surface() -> None:
    result, records = perturb_taubench_tool_result(
        json.dumps(
            {
                "flight_number": "HAT001",
                "date": "2024-05-01",
                "status": "confirmed",
                "cabin": "basic_economy",
            }
        ),
        strength="medium",
        target="lookup_flight",
    )

    parsed = json.loads(result)
    assert parsed["data"]["flightNumber"] == "HAT001"
    assert parsed["data"]["date"] == "05/01/2024"
    assert parsed["data"]["status"] == "CONFIRMED"
    assert records[0].target == "lookup_flight"


@pytest.mark.anyio
async def test_taubench_tool_wrapper_reverses_args_and_perturbs_result() -> None:
    calls = {}
    env = StructuralEnvironment(
        kind="taubench",
        strength="severe",
        context=StructuralContext(campaign_id="campaign", agent="codex_cli", repeat_id=0),
    )
    wrapped = env._wrap_tool("airline", _capture_flight_tool(calls))

    result = await wrapped(fltNo="HAT001", dt="2024-05-01")
    parsed = json.loads(result)

    assert calls == {"flight_number": "HAT001", "date": "2024-05-01"}
    assert parsed["data"]["fltNo"] == "HAT001"
    assert parsed["data"]["dt"] == "20240501"


def test_structural_runner_emits_paired_metadata(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_eval(**kwargs):
        captured.append(kwargs)
        phase = kwargs["metadata"]["reliability_phase"]
        value = 1.0 if phase == "structural_baseline" else 0.0
        return [_fake_log(value)]

    monkeypatch.setattr("inspect_swe.reliability.structural.eval", fake_eval)

    result = run_structural_phase(
        spec=ReliabilitySpec(
            benchmark="toy_benchmark",
            agents=["mock_agent"],
            phases=["structural"],
            concurrency=OrchestratorConcurrency(max_samples=2),
        ),
        tasks="toy_task",
        config=StructuralPhaseConfig(
            repeats=1,
            campaign_id="campaign",
            compute_confidence=False,
            kind="gaia",
            strength="medium",
        ),
    )

    assert [call["metadata"]["reliability_phase"] for call in captured] == [
        "structural_baseline",
        "structural_perturbed",
    ]
    assert captured[0]["max_samples"] == 2
    assert result.results[0].baseline_accuracy == 1.0
    assert result.results[0].perturbed_accuracy == 0.0
    assert result.results[0].r_struct == 0.0
    assert result.results[0].baseline_log_paths[0].endswith(".eval")


def test_structural_runner_uses_taubench_codex_adapter(monkeypatch) -> None:
    captured: list[dict] = []
    adapter_calls: list[dict] = []

    def fake_adapter(**kwargs):
        adapter_calls.append(kwargs)
        return "tau2_codex_solver"

    def fake_eval(**kwargs):
        captured.append(kwargs)
        return [_fake_log(1.0)]

    monkeypatch.setattr(
        "inspect_swe.reliability.taubench.tau2_airline_codex_solver",
        fake_adapter,
    )
    monkeypatch.setattr("inspect_swe.reliability.structural.eval", fake_eval)

    run_structural_phase(
        spec=ReliabilitySpec(
            benchmark="tau2_airline",
            agents=["codex_cli"],
            phases=["structural"],
        ),
        tasks="tau2_task",
        config=StructuralPhaseConfig(
            repeats=1,
            campaign_id="campaign",
            compute_confidence=False,
            include_baseline=False,
            kind="taubench",
            strength="mild",
            taubench_codex_adapter=True,
            taubench_message_limit=30,
        ),
    )

    assert captured[0]["solver"] is not None
    assert adapter_calls[0]["message_limit"] == 30
    assert adapter_calls[0]["structural_env"].kind == "taubench"
    assert adapter_calls[0]["structural_env"].strength == "mild"


def test_structural_codex_default_kwargs_follow_current_agent_api(monkeypatch) -> None:
    calls = {}

    def fake_codex_cli(**kwargs):
        calls.update(kwargs)
        return "codex-agent"

    monkeypatch.setattr("inspect_swe.codex_cli", fake_codex_cli)
    monkeypatch.setattr(
        "inspect_swe.reliability.structural.as_solver",
        lambda solver_value: ("solver", solver_value),
    )
    env = StructuralEnvironment(
        kind="gaia",
        strength="medium",
        context=StructuralContext(campaign_id="campaign", agent="codex_cli", repeat_id=0),
    )

    solver = _default_structural_solver_for_agent(
        "codex_cli",
        env,
        StructuralPhaseConfig(kind="gaia", compute_confidence=False),
        perturbed=True,
    )

    assert solver == ("solver", "codex-agent")
    assert callable(calls["filter"])
    assert calls["bridged_tools"] is None
    assert calls["retry_refusals"] == 3
    assert calls["version"] == RELIABILITY_CODEX_CLI_VERSION


def _capture_flight_tool(calls: dict[str, str]) -> Tool:
    @tool
    def lookup_flight() -> Tool:
        async def execute(flight_number: str, date: str) -> str:
            """Look up a flight.

            Args:
                flight_number: Flight number.
                date: Flight date.
            """
            calls.update({"flight_number": flight_number, "date": date})
            return json.dumps(
                {
                    "flight_number": flight_number,
                    "date": date,
                    "status": "confirmed",
                }
            )

        return execute

    return lookup_flight()


def _fake_log(score_value: float):
    return SimpleNamespace(
        eval=SimpleNamespace(run_id="run"),
        location="logs/reliability/toy.eval",
        samples=[SimpleNamespace(scores={"score": SimpleNamespace(value=score_value)})],
    )
