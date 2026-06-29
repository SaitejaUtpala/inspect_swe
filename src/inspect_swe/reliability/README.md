# Reliability

This package contains reliability runners for `inspect_swe`. The current tested
path is baseline/fault evals for Codex CLI, with emphasis on `exec_command`
observation faulting.

## Exec Command Faulting

The main fault tested so far is:

```text
--fault-mode exec_observation_error
```

It does not modify the command the agent requested. Codex is still allowed to
call `exec_command` normally. After the tool returns, the model-facing tool
observation can be replaced with a realistic shell failure:

```text
Command: /bin/bash -lc <redacted>
Wall time: 0.0000 seconds
Process exited with code 254
Original token count: 8
Output:
bash: fork: Resource temporarily unavailable
```

This keeps the protocol valid. The model sees a normal tool result, but the
result represents a failed command. Codex can then retry, reason around the
failure, or give a final answer.

## Run GAIA Exec Fault Evals

From this checkout:

```bash
cd /Users/saitejautpala/work/hal_explore/inspect_swe_new/inspect_swe
```

GAIA Level 1 is easiest to reference by file task:

```bash
TASK="/Users/saitejautpala/miniconda3/envs/inspect_swe_new/lib/python3.12/site-packages/inspect_evals/gaia/gaia.py@gaia_level1"
```

Example fault run:

```bash
conda run -n inspect_swe_new python -m inspect_swe.reliability.cli campaign \
  --benchmark "$TASK" \
  --phase fault \
  --agent codex_cli \
  --model openai/gpt-5.5 \
  --limit 12 \
  --max-samples 6 \
  --repeats 1 \
  --fault-mode exec_observation_error \
  --fault-probability 0.5 \
  --fault-target message.exec_command \
  --no-compute-confidence \
  --log-root logs/reliability_migration_fault_tolerant
```

View logs:

```bash
conda run -n inspect_swe_new inspect view --log-dir logs/reliability_migration_fault_tolerant
```

## Structural Robustness

Structural robustness measures whether Codex handles harmless changes to the
surface of the task. For GAIA, that means prompt formatting, instruction wording,
number/date formatting, and later free-text observation wrapping. For TauBench,
it means the typed tool/API surface: parameter names, required fields, and JSON
result shape.

The clean workflow is to run `k` baselines first as their own phase, then run
structural perturbations separately. That keeps the clean anchor reusable across
`mild`, `medium`, and `severe` runs instead of paying for a new baseline inside
each structural command.

Run the GAIA baseline anchor:

```bash
cd /Users/saitejautpala/work/hal_explore/inspect_swe_new/inspect_swe
TASK="/Users/saitejautpala/miniconda3/envs/inspect_swe_new/lib/python3.12/site-packages/inspect_evals/gaia/gaia.py@gaia_level1"

conda run -n inspect_swe_new python -m inspect_swe.reliability.cli campaign \
  --benchmark "$TASK" \
  --phase baseline \
  --agent codex_cli \
  --model openai/gpt-5.5 \
  --repeats 3 \
  --limit 10 \
  --max-samples 4 \
  --sandbox docker \
  --no-compute-confidence \
  --log-root logs/reliability/gaia_structural_10
```

Then run the perturbed structural passes:

```bash
for strength in mild medium severe; do
  conda run -n inspect_swe_new python -m inspect_swe.reliability.cli campaign \
    --benchmark "$TASK" \
    --phase structural \
    --agent codex_cli \
    --model openai/gpt-5.5 \
    --repeats 1 \
    --limit 10 \
    --max-samples 4 \
    --sandbox docker \
    --structural-kind gaia \
    --structural-strength "$strength" \
    --no-structural-baseline \
    --no-compute-confidence \
    --log-root logs/reliability/gaia_structural_10 \
    2>&1 | tee "logs/reliability/gaia_structural_10/gaia_struct_${strength}_10.out"
done
```

`--no-structural-baseline` is intentional here. It says: only run the perturbed
structural pass, because the clean baselines already exist. If you omit it, the
structural runner will create a baseline and perturbed pair inside each
structural campaign, which is easier to inspect but more expensive.

Dump one `.eval` to JSON:

```bash
conda run -n inspect_swe_new python -c "from inspect_ai.log import read_eval_log; from pathlib import Path; src=Path('<path-to-log.eval>'); src.with_suffix('.json').write_text(read_eval_log(str(src)).model_dump_json(indent=2), encoding='utf-8')"
```

## What Has Been Tested

- Reliability branch assessed against upstream `inspect_swe` `0.2.63-11-gf86c551`
  (`upstream/main` commit `f86c551`, merge base `bc37fce`). The reliability
  package is branch-local custom code and must be preserved when merging upstream.
- Focused reliability tests pass: `24 passed`.
- Focused reliability tests cover Codex and Claude default solver construction,
  Codex structural wrapper kwargs, TauBench tool schema perturbation/reversal,
  and exec-observation fault injection.
- Ruff checks pass for `src/inspect_swe/reliability` and the reliability tests.
- GAIA Level 1, `codex_cli`, GPT-5.4, `limit=8`, `max_samples=4`, `p=0.8` completed successfully.
- GAIA Level 1, `codex_cli`, GPT-5.5, `limit=12`, `max_samples=6`, `p=0.5` completed successfully.
- A `p=1.0` smoke run confirmed the injected observation appears in the model-facing log view.
- The fault is scoped to `ChatMessageTool(function="exec_command")`; it does not change web search behavior.
- GAIA structural runs with `codex_cli` and GPT-5.5 have completed for
  `mild`, `medium`, and `severe` perturbation strengths on 10 examples.

## Probability Semantics

`FaultSpec.probability` is per eligible fault site, not per sample and not an
exact percentage of the run. For `exec_observation_error`, each new
`exec_command` observation gets one deterministic decision. Already-seen tool
call IDs are not resampled.
