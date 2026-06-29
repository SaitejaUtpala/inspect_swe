#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-anthropic/claude-opus-4-8}"
SAMPLES="${SAMPLES:-10}"
STRENGTH="${STRENGTH:-medium}"
GAIA_BENCHMARK="${GAIA_BENCHMARK:-inspect_evals/gaia}"

GAIA_LOG_ROOT="${GAIA_LOG_ROOT:-logs/reliability/claude_code_gaia_structural_${STRENGTH}_${SAMPLES}}"
TAU2_LOG_ROOT="${TAU2_LOG_ROOT:-logs/reliability/claude_code_tau2_airline_structural_${STRENGTH}_${SAMPLES}}"

mkdir -p "$GAIA_LOG_ROOT" "$TAU2_LOG_ROOT"

TAU2="$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import inspect_evals.tau2.tau2 as tau2

print(f"{Path(tau2.__file__)}@tau2_airline")
PY
)"

echo "MODEL=$MODEL"
echo "TAU2=$TAU2"
if [[ -z "$TAU2" ]]; then
  echo "Failed to resolve Tau2 Airline benchmark path. Is inspect_evals installed in this active environment?" >&2
  exit 1
fi
if [[ ! -f "${TAU2%@*}" ]]; then
  echo "Resolved Tau2 module file does not exist: ${TAU2%@*}" >&2
  exit 1
fi

echo "Running Claude Code GAIA structural: strength=$STRENGTH samples=$SAMPLES"
"$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
  --benchmark "$GAIA_BENCHMARK" \
  --phase structural \
  --agent claude_code \
  --model "$MODEL" \
  --repeats 1 \
  --limit "$SAMPLES" \
  --max-samples "$SAMPLES" \
  --sandbox docker \
  --structural-kind gaia \
  --structural-strength "$STRENGTH" \
  --no-structural-baseline \
  --no-compute-confidence \
  --log-root "$GAIA_LOG_ROOT" \
  2>&1 | tee "$GAIA_LOG_ROOT/gaia_struct_${STRENGTH}_${SAMPLES}.out"

echo "Running Claude Code Tau2 structural: strength=$STRENGTH samples=$SAMPLES"
echo "Note: Tau2 tool-schema perturbations are currently wired through the Codex TauBench adapter; this command smoke-tests the Claude Code structural path."
"$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
  --benchmark "$TAU2" \
  --phase structural \
  --agent claude_code \
  --model "$MODEL" \
  --repeats 1 \
  --limit "$SAMPLES" \
  --max-samples "$SAMPLES" \
  --sandbox docker \
  --task-arg shuffle=false \
  --structural-kind taubench \
  --structural-strength "$STRENGTH" \
  --no-structural-baseline \
  --no-compute-confidence \
  --log-root "$TAU2_LOG_ROOT" \
  2>&1 | tee "$TAU2_LOG_ROOT/tau2_airline_struct_${STRENGTH}_${SAMPLES}.out"
