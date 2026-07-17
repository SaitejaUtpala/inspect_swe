#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# OpenCode reliability sweep for GAIA: baseline, structural, and fault phases.
#
# It sweeps a set of bridge models (MODELS). For each model, opencode_model (the
# provider client opencode formats requests for) is matched to the model's
# provider automatically by the reliability solver, so mixing anthropic/openai/
# google models needs no extra flags. Logs are written per-model.
#
# The fault phase injects exec-observation faults into opencode's shell tool
# (opencode's shell tool is `bash`, so the fault target is message.bash).

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-anthropic/claude-opus-4-8 openai/gpt-5.5 google/gemini-3.5-flash}"
GAIA_LEVEL="${GAIA_LEVEL:-gaia_level1}"
LIMIT="${LIMIT:-15}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
REPEATS="${REPEATS:-1}"
SANDBOX="${SANDBOX:-docker}"
STRENGTHS="${STRENGTHS:-mild medium severe}"
FAULT_PROBABILITY="${FAULT_PROBABILITY:-0.4}"
FAULT_MODE="${FAULT_MODE:-exec_observation_error}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/opencode_gaia_${GAIA_LEVEL}_${LIMIT}_ms${MAX_SAMPLES}}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_STRUCTURAL="${RUN_STRUCTURAL:-1}"
RUN_FAULT="${RUN_FAULT:-1}"

mkdir -p "$LOG_ROOT"

DEFAULT_GAIA_BENCHMARK="$(
  "$PYTHON_BIN" - "$GAIA_LEVEL" <<'PY'
from pathlib import Path
import importlib
import sys

gaia_module = importlib.import_module("inspect_evals.gaia.gaia")

print(f"{Path(gaia_module.__file__)}@{sys.argv[1]}")
PY
)"
GAIA_BENCHMARK="${GAIA_BENCHMARK:-$DEFAULT_GAIA_BENCHMARK}"

echo "MODELS=$MODELS"
echo "GAIA_LEVEL=$GAIA_LEVEL"
echo "GAIA_BENCHMARK=$GAIA_BENCHMARK"
echo "LIMIT=$LIMIT"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "SANDBOX=$SANDBOX"
echo "LOG_ROOT=$LOG_ROOT"
echo "STRENGTHS=$STRENGTHS"
echo "FAULT_MODE=$FAULT_MODE"
echo "FAULT_PROBABILITY=$FAULT_PROBABILITY"

if [[ -z "$GAIA_BENCHMARK" ]]; then
  echo "Failed to resolve GAIA benchmark path. Is inspect_evals installed in this active environment?" >&2
  exit 1
fi
if [[ ! -f "${GAIA_BENCHMARK%@*}" ]]; then
  echo "Resolved GAIA module file does not exist: ${GAIA_BENCHMARK%@*}" >&2
  exit 1
fi

model_slug() {
  # Turn a model ref like "google/gemini-3.5-flash" into "google_gemini-3-5-flash".
  echo "$1" | tr '/.' '__'
}

run_model() {
  local model="$1"
  local slug
  slug="$(model_slug "$model")"
  local root="$LOG_ROOT/$slug"
  mkdir -p "$root"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "Running OpenCode GAIA baseline: model=$model samples=$MAX_SAMPLES"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$GAIA_BENCHMARK" \
      --phase baseline \
      --agent opencode \
      --model "$model" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --no-compute-confidence \
      --log-root "$root/baseline" \
      2>&1 | tee "$root/baseline.out"
  fi

  if [[ "$RUN_STRUCTURAL" == "1" ]]; then
    for strength in $STRENGTHS; do
      echo "Running OpenCode GAIA structural: model=$model strength=$strength samples=$MAX_SAMPLES"
      "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
        --benchmark "$GAIA_BENCHMARK" \
        --phase structural \
        --agent opencode \
        --model "$model" \
        --repeats "$REPEATS" \
        --limit "$LIMIT" \
        --max-samples "$MAX_SAMPLES" \
        --sandbox "$SANDBOX" \
        --structural-kind gaia \
        --structural-strength "$strength" \
        --no-structural-baseline \
        --no-compute-confidence \
        --log-root "$root/structural_${strength}" \
        2>&1 | tee "$root/structural_${strength}.out"
    done
  fi

  if [[ "$RUN_FAULT" == "1" ]]; then
    echo "Running OpenCode GAIA fault: model=$model mode=$FAULT_MODE p=$FAULT_PROBABILITY samples=$MAX_SAMPLES"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$GAIA_BENCHMARK" \
      --phase fault \
      --agent opencode \
      --model "$model" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --fault-surface message \
      --fault-mode "$FAULT_MODE" \
      --fault-probability "$FAULT_PROBABILITY" \
      --fault-target message.bash \
      --no-compute-confidence \
      --log-root "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}" \
      2>&1 | tee "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}.out"
  fi
}

for model in $MODELS; do
  run_model "$model"
done
