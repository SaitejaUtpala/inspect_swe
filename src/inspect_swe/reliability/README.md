# Inspect-SWE Reliability Evaluation

This package runs reliability evaluations for Inspect-SWE agents (for example `codex_cli`, `claude_code`, and `gemini_cli`) using Inspect-native building blocks.

The short version:

- `.eval` logs are the source of truth.
- Solvers/scorers/reducers do the core reliability work.
- Hooks are kept thin and mainly write sidecar records.
- Campaigns run all phases and produce one analysis + report.

## What this adds beyond normal eval

A normal eval tells you if the agent solved tasks.

Reliability eval also checks:

- consistency across repeats,
- robustness under faults/prompt changes/structural changes,
- safety signals,
- abstention behavior.

Supported phases:

1. `baseline`: repeated clean runs.
2. `fault`: injected failures (model/tool).
3. `prompt`: rewritten prompt variants.
4. `structural`: tool-response structure perturbations.
5. `safety`: posthoc analysis phase.
6. `abstention`: posthoc analysis phase.

The first four run evals. Safety and abstention are computed during analysis.

## How to run

Main commands:

- `campaign`: run multiple phases end-to-end.
- `baseline` / `fault` / `prompt` / `structural`: run one phase directly.
- `analyze`: recompute analysis/report from artifacts.

Example full campaign:

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

## Output files

Outputs are grouped by campaign/benchmark under `logs/reliability/`.

You will typically see:

- Inspect `.eval` logs for each repeat (canonical run data),
- sidecar records (`records.jsonl`) used for compatibility and projections,
- campaign manifest JSON (resume/status),
- campaign analysis JSON,
- markdown report.

## What hooks mean here

Hooks are Inspect lifecycle listeners.

In this package they are intentionally limited to:

- collecting sample telemetry,
- projecting data into reliability sidecar records,
- preserving run identity tags.

Hooks do **not** define execution truth and do **not** own metric logic. `.eval` logs and Inspect scorer outputs do.

## Full flow

```mermaid
flowchart TD
    A["CLI (`reliability.cli`)"] --> B["Build spec + phase configs"]
    B --> C{"campaign?"}

    C -- "yes" --> D["`run_reliability_campaign` (`orchestrator.py`)"]
    C -- "no" --> E["Single phase runner"]

    D --> E
    E --> F["`phase_core.execute_phase_repeats(...)`"]
    F --> G["`phase_core.run_eval_for_repeat(...)`"]
    G --> H["Inspect `eval(...)` runtime"]

    H --> I["Agent solver from `solver_factory.py`"]
    I --> J["`reliability_instrumented_solver` + signal collector"]
    H --> K["Perturbation adapters\nfault / prompt / structural"]

    H --> L["Inspect lifecycle events"]
    L --> M["`hooks.py` writes sidecar records"]

    G --> N["Canonical `.eval` logs"]
    N --> O["`analysis.py` (`eval_preferred`, sidecar fallback)"]
    O --> P["`reporting.py` writes markdown + json"]
```

## Code map

- `spec.py`: reliability configs and phase specs.
- `orchestrator.py`: campaign lifecycle, resume, phase dispatch.
- `phase_core.py`: shared repeat/eval execution logic.
- `baseline.py`, `fault.py`, `prompt.py`, `structural.py`: thin phase wrappers.
- `solver_factory.py`: default solver wiring per agent.
- `solver_signals.py`, `signals.py`, `scorers.py`, `reducers.py`: Inspect-native reliability signals and aggregation.
- `hooks.py`: sidecar projection from lifecycle events.
- `analysis.py`: campaign metrics and summaries.
- `reporting.py`: markdown and JSON output.
- `paths.py`: path helpers for campaign artifacts.

## Design rules (so this stays clean)

- Keep `.eval` logs as canonical truth.
- Prefer solver/scorer/reducer logic over hook-heavy logic.
- Keep hooks minimal and mechanical.
- Preserve run identities (`repeat_id`, `sample_retry_id`, `agent_attempt_id`).
- Reuse `phase_core` instead of duplicating phase runner code.
