#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-openai/gpt-5.5}"
GAIA_LEVEL="${GAIA_LEVEL:-gaia_level1}"
LIMIT="${LIMIT:-15}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
REPEATS="${REPEATS:-1}"
SANDBOX="${SANDBOX:-docker}"
STRENGTHS="${STRENGTHS:-mild medium severe}"
FAULT_PROBABILITY="${FAULT_PROBABILITY:-0.4}"
FAULT_MODE="${FAULT_MODE:-exec_observation_error}"
TAUBENCH_MESSAGE_LIMIT="${TAUBENCH_MESSAGE_LIMIT:-30}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/codex_gpt55_gaia_tau2_${LIMIT}_ms${MAX_SAMPLES}}"

RUN_GAIA="${RUN_GAIA:-1}"
RUN_TAU2="${RUN_TAU2:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_STRUCTURAL="${RUN_STRUCTURAL:-1}"
RUN_FAULT="${RUN_FAULT:-1}"
RUN_TAU2_FAULT="${RUN_TAU2_FAULT:-0}"

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

TAU2_BENCHMARK="$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import inspect_evals.tau2.tau2 as tau2

print(f"{Path(tau2.__file__)}@tau2_airline")
PY
)"

echo "MODEL=$MODEL"
echo "GAIA_BENCHMARK=$GAIA_BENCHMARK"
echo "TAU2_BENCHMARK=$TAU2_BENCHMARK"
echo "LIMIT=$LIMIT"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "SANDBOX=$SANDBOX"
echo "LOG_ROOT=$LOG_ROOT"
echo "STRENGTHS=$STRENGTHS"
echo "FAULT_MODE=$FAULT_MODE"
echo "FAULT_PROBABILITY=$FAULT_PROBABILITY"

if [[ -z "$GAIA_BENCHMARK" || ! -f "${GAIA_BENCHMARK%@*}" ]]; then
  echo "Failed to resolve GAIA benchmark path: $GAIA_BENCHMARK" >&2
  exit 1
fi
if [[ -z "$TAU2_BENCHMARK" || ! -f "${TAU2_BENCHMARK%@*}" ]]; then
  echo "Failed to resolve Tau2 Airline benchmark path: $TAU2_BENCHMARK" >&2
  exit 1
fi

run_gaia() {
  local root="$LOG_ROOT/gaia_${GAIA_LEVEL}"
  mkdir -p "$root"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "Running Codex GAIA baseline"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$GAIA_BENCHMARK" \
      --phase baseline \
      --agent codex_cli \
      --model "$MODEL" \
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
      echo "Running Codex GAIA structural: strength=$strength"
      "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
        --benchmark "$GAIA_BENCHMARK" \
        --phase structural \
        --agent codex_cli \
        --model "$MODEL" \
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
    echo "Running Codex GAIA fault/env-style exec observation test"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$GAIA_BENCHMARK" \
      --phase fault \
      --agent codex_cli \
      --model "$MODEL" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --fault-surface message \
      --fault-mode "$FAULT_MODE" \
      --fault-probability "$FAULT_PROBABILITY" \
      --fault-target message.exec_command \
      --no-compute-confidence \
      --log-root "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}" \
      2>&1 | tee "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}.out"
  fi
}

run_tau2() {
  local root="$LOG_ROOT/tau2_airline"
  mkdir -p "$root"

  if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "Running Codex Tau2 baseline"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$TAU2_BENCHMARK" \
      --phase baseline \
      --agent codex_cli \
      --model "$MODEL" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --task-arg shuffle=false \
      --taubench-codex-adapter \
      --taubench-message-limit "$TAUBENCH_MESSAGE_LIMIT" \
      --no-compute-confidence \
      --log-root "$root/baseline" \
      2>&1 | tee "$root/baseline.out"
  fi

  if [[ "$RUN_STRUCTURAL" == "1" ]]; then
    for strength in $STRENGTHS; do
      echo "Running Codex Tau2 structural/environmental: strength=$strength"
      "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
        --benchmark "$TAU2_BENCHMARK" \
        --phase structural \
        --agent codex_cli \
        --model "$MODEL" \
        --repeats "$REPEATS" \
        --limit "$LIMIT" \
        --max-samples "$MAX_SAMPLES" \
        --sandbox "$SANDBOX" \
        --task-arg shuffle=false \
        --structural-kind taubench \
        --structural-strength "$strength" \
        --taubench-codex-adapter \
        --taubench-message-limit "$TAUBENCH_MESSAGE_LIMIT" \
        --no-structural-baseline \
        --no-compute-confidence \
        --log-root "$root/structural_${strength}" \
        2>&1 | tee "$root/structural_${strength}.out"
    done
  fi

  if [[ "$RUN_TAU2_FAULT" == "1" ]]; then
    echo "Running Codex Tau2 fault exec observation test"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$TAU2_BENCHMARK" \
      --phase fault \
      --agent codex_cli \
      --model "$MODEL" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --task-arg shuffle=false \
      --fault-surface message \
      --fault-mode "$FAULT_MODE" \
      --fault-probability "$FAULT_PROBABILITY" \
      --fault-target message.exec_command \
      --taubench-codex-adapter \
      --taubench-message-limit "$TAUBENCH_MESSAGE_LIMIT" \
      --no-compute-confidence \
      --log-root "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}" \
      2>&1 | tee "$root/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}.out"
  else
    echo "Skipping Tau2 fault by default: Tau2 environmental robustness is covered by structural tool/API perturbations."
  fi
}

if [[ "$RUN_GAIA" == "1" ]]; then
  run_gaia
fi

if [[ "$RUN_TAU2" == "1" ]]; then
  run_tau2
fi
