# Inspect-SWE Reliability Evaluation

This package adds a reliability evaluation framework on top of Inspect and Inspect-SWE agents (for example `codex_cli`, `claude_code`, and `gemini_cli`).

It is intentionally designed in two layers:

- **Canonical execution truth** comes from Inspect `.eval` logs.
- **Reliability sidecars** (JSONL/JSON) are derived projections used for campaign analysis and reporting.

---

## Part 1: Beginner-Friendly Overview

This section is for readers who are new to Inspect / Inspect-SWE.

### What problem does this solve?

Normal benchmark runs often answer one question: "Did the agent solve the task?"

Reliability evaluation asks more:

- Is performance stable across repeated runs?
- What happens when we inject faults?
- What happens if we rewrite prompts or perturb tool response structure?
- Can we detect safety violations and abstention behavior?

### Reliability phases

This framework supports six phases:

1. `baseline` - repeated clean runs.
2. `fault` - controlled failures (model/tool).
3. `prompt` - instruction rewrites / prompt variants.
4. `structural` - tool-response shape perturbations.
5. `safety` - posthoc campaign analysis.
6. `abstention` - posthoc campaign analysis.

The first four phases execute eval runs. The last two are computed from analysis artifacts.

### How to run it

Use the reliability CLI:

- Phase-level commands: `baseline`, `fault`, `prompt`, `structural`
- End-to-end campaign command: `campaign`
- Analysis command: `analyze`

Example campaign run:

```bash
python -m inspect_swe.reliability.cli campaign \
  --benchmark gaia_level1 \
  --tasks inspect_evals/gaia_level1 \
  --agent codex_cli \
  --phase baseline --phase fault --phase prompt --phase structural --phase safety --phase abstention \
  --model openai/gpt-5.4-2026-03-05 \
  --repeats 3 \
  --campaign-id my_campaign_001
```

### What files are produced?

By default under `logs/reliability/<benchmark>/`:

- Inspect `.eval` logs per repeat (canonical truth).
- Campaign sidecar files (for analysis convenience).
- Campaign manifest JSON (resume state).
- Campaign analysis JSON + markdown report.

### What are hooks in this system?

Hooks are Inspect lifecycle listeners that observe task/sample events while eval is running.

In this package, hooks are used to:

- capture sample-level telemetry (outcomes, behavior, resources, metadata),
- enforce reliability identity tagging rules,
- write sidecar records.

Important: hooks are **not** the source of truth for execution. `.eval` logs are.

### Full flow diagram

```mermaid
flowchart TD
    %% Entry
    A["CLI (`reliability.cli`)\nphase command or campaign command"] --> B["Build `ReliabilitySpec` + config\n(agents, phases, repeats, perturbations, concurrency)"]
    B --> C{"command = campaign?"}

    %% Campaign path
    C -- "yes" --> D["`run_reliability_campaign` (`orchestrator.py`)\npreflight + manifest/resume"]
    D --> E{"phase type"}
    E -- "baseline/fault/prompt/structural" --> F["Phase wrapper\n(`baseline.py` / `fault.py` / `prompt.py` / `structural.py`)"]
    E -- "safety/abstention" --> G["Posthoc marker in manifest\n(no eval run)"]

    %% Single phase path
    C -- "no (single phase)" --> F

    %% Phase execution core
    F --> H["Configure reliability hooks\n(`hooks.py`)"]
    H --> I["`execute_phase_repeats` (`phase_core.py`)\nloop agents x repeats"]
    I --> J["`run_eval_for_repeat` (`phase_core.py`)\ncall Inspect `eval(...)`"]
    J --> K["Inspect runtime\nTask + solver + scorer pipeline"]

    %% Solver/scorer internals
    K --> L["Agent solver from `solver_factory.py`\nwrapped by `reliability_instrumented_solver`"]
    L --> M["`reliability_signal_collector`\nadds confidence/safety/abstention scores"]
    K --> N["Perturbation adapters\nfault / prompt / structural"]

    %% Hook sidecar projection
    K --> O["Inspect sample lifecycle events"]
    O --> P["`ReliabilityHooks.on_sample_end`\nproject to `ReliabilityRecord`"]
    P --> Q["Sidecar JSONL (`records.jsonl`)"]

    %% Canonical logs + coverage
    J --> R["Canonical `.eval` logs (source of truth)"]
    R --> S["Coverage check (`telemetry.py`)\nexpected vs observed sidecar records"]
    S --> T{"telemetry complete?"}
    T -- "no + strict" --> U["Raise phase error"]
    T -- "yes" --> V["Phase result summary\n(run_ids, log paths, coverage)"]

    %% Campaign artifacts
    V --> W["Write campaign JSON projection\n(`artifacts.py`)"]
    W --> X["Update manifest\n(status per phase)"]
    G --> X

    %% Analysis/reporting
    X --> Y{"run analysis?"}
    Y -- "yes" --> Z["`analyze_reliability_campaign` (`analysis.py`)\nsource=`eval_preferred` (fallback sidecar)"]
    Z --> AA["Compute predictability/robustness/\nsafety/abstention metrics"]
    AA --> AB["Write analysis JSON + markdown report\n(`reporting.py`)"]
    Y -- "no" --> AC["Return campaign result"]

    AB --> AC["Final campaign result\n(paths + phase statuses + reports)"]
```

---

## Part 2: Advanced Design (Inspect / Inspect-SWE)

This section is for readers already familiar with Inspect internals.

### Design principles

1. **Inspect-native first**
   - Use Inspect solvers, scorers, reducers, and `.eval` logs as the primary model.
2. **Thin hooks**
   - Hooks are compatibility/normalization adapters, not orchestration owners.
3. **Explicit identity semantics**
   - Preserve `repeat_id`, `sample_retry_id`, and `agent_attempt_id`.
4. **Deterministic campaign layout**
   - Stable path structure and campaign ids for resume + analysis.
5. **Minimal duplication**
   - Shared phase execution loop in `phase_core`.

### Module map

- `spec.py`
  - `ReliabilitySpec`, phase list, perturbation and concurrency specs.
- `orchestrator.py`
  - Campaign lifecycle, resume manifest, phase dispatch, analysis/report wiring.
- `baseline.py`, `fault.py`, `prompt.py`, `structural.py`
  - Thin phase wrappers with phase-specific perturbation/solver wiring.
- `phase_core.py`
  - Shared repeat execution, metadata construction, eval invocation, telemetry checks.
- `solver_factory.py`
  - Agent-specific default solver resolution.
- `solver_signals.py`
  - Solver-level reliability signal collection.
- `scorers.py`, `reducers.py`
  - Inspect-native reliability scorers and reducers.
- `hooks.py`
  - Sidecar emission and metadata normalization from Inspect lifecycle events.
- `analysis.py`
  - Campaign metrics (`eval_preferred` source with sidecar fallback).
- `reporting.py`
  - Analysis JSON and markdown report output.
- `paths.py`
  - Canonical run/campaign path builders.

### Execution flow

For each execution phase:

1. Phase wrapper validates `ReliabilitySpec` and phase config.
2. Hooks are configured (if enabled).
3. `phase_core.execute_phase_repeats(...)` runs all `(agent, repeat)` pairs.
4. Each repeat calls `phase_core.run_eval_for_repeat(...)` with Inspect `eval(...)`.
5. Coverage is checked against sidecar records.
6. Phase results are normalized and returned.

Campaign orchestrator then:

- persists/updates a campaign manifest,
- marks safety/abstention as posthoc phases,
- runs campaign analysis and report generation.

### Solver-centric reliability signals

`solver_signals.py` adds `reliability_signal_collector` and `reliability_instrumented_solver`.

Default agent solvers are wrapped in `solver_factory.py` so reliability signals are collected into `state.scores` consistently. Current built-in signals:

- `reliability_confidence_signal`
- `reliability_safety_violation`
- `reliability_abstention_signal`

This keeps reliability semantics closer to the solver/scorer pipeline and reduces hook-specific behavior.

### Hook contract

`hooks.py` subscribes to task/sample lifecycle and writes `ReliabilityRecord` rows.

Hook responsibilities:

- build `ReliabilityRunIdentity`,
- project outcomes/behavior/resources,
- read reliability signals from scorer outputs when available,
- attach metadata and perturbation tags.

Hook non-responsibilities:

- deciding campaign orchestration logic,
- replacing canonical `.eval` truth,
- owning metric computation.

### Analysis source model

`analyze_reliability_campaign(...)` supports:

- `source="eval_preferred"` (default): derive from eval logs when possible.
- `source="sidecar_only"`: force sidecar-based analysis.

This supports strict Inspect-native analysis while preserving backward compatibility.

### Extending the framework

Typical extension points:

- New perturbation mode:
  - add policy logic under `perturbations/`,
  - expose config in phase + CLI,
  - emit perturbation metadata tags.
- New reliability signal:
  - implement extractor in `signals.py`,
  - expose scorer in `scorers.py`,
  - optionally add reducer in `reducers.py`.
- New agent integration:
  - add mapping in `solver_factory.py`.

Keep extensions aligned with the same rules:

- avoid moving core semantics into hooks,
- prefer solver/scorer/reducer pipelines,
- preserve canonical `.eval` + identity fidelity.

---

## Practical Notes

- Campaign ids isolate run artifacts and make resume deterministic.
- Use `--analysis-source eval_preferred` to bias toward Inspect-native derivation.
- Use strict telemetry checks in CI (`--verify-telemetry --fail-on-incomplete-telemetry`).

