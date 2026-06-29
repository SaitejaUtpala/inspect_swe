#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-anthropic/claude-opus-4-8}"
GAIA_BENCHMARK="${GAIA_BENCHMARK:-inspect_evals/gaia}"
LIMIT="${LIMIT:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
REPEATS="${REPEATS:-1}"
SANDBOX="${SANDBOX:-docker}"
STRENGTHS="${STRENGTHS:-mild medium severe}"
FAULT_PROBABILITY="${FAULT_PROBABILITY:-1.0}"
FAULT_MODE="${FAULT_MODE:-exec_observation_error}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/claude_code_gaia_opus_4_8_reliability_${MAX_SAMPLES}}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_FAULT="${RUN_FAULT:-1}"
RUN_STRUCTURAL="${RUN_STRUCTURAL:-1}"

mkdir -p "$LOG_ROOT"

echo "MODEL=$MODEL"
echo "GAIA_BENCHMARK=$GAIA_BENCHMARK"
echo "LIMIT=$LIMIT"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "SANDBOX=$SANDBOX"
echo "LOG_ROOT=$LOG_ROOT"

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "Running Claude Code GAIA baseline: samples=$MAX_SAMPLES"
  "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
    --benchmark "$GAIA_BENCHMARK" \
    --phase baseline \
    --agent claude_code \
    --model "$MODEL" \
    --repeats "$REPEATS" \
    --limit "$LIMIT" \
    --max-samples "$MAX_SAMPLES" \
    --sandbox "$SANDBOX" \
    --no-compute-confidence \
    --log-root "$LOG_ROOT/baseline" \
    2>&1 | tee "$LOG_ROOT/baseline.out"
fi

if [[ "$RUN_FAULT" == "1" ]]; then
  echo "Running Claude Code GAIA fault: mode=$FAULT_MODE p=$FAULT_PROBABILITY samples=$MAX_SAMPLES"
  "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
    --benchmark "$GAIA_BENCHMARK" \
    --phase fault \
    --agent claude_code \
    --model "$MODEL" \
    --repeats "$REPEATS" \
    --limit "$LIMIT" \
    --max-samples "$MAX_SAMPLES" \
    --sandbox "$SANDBOX" \
    --fault-surface message \
    --fault-mode "$FAULT_MODE" \
    --fault-probability "$FAULT_PROBABILITY" \
    --no-compute-confidence \
    --log-root "$LOG_ROOT/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}" \
    2>&1 | tee "$LOG_ROOT/fault_${FAULT_MODE}_p${FAULT_PROBABILITY}.out"
fi

if [[ "$RUN_STRUCTURAL" == "1" ]]; then
  for strength in $STRENGTHS; do
    echo "Running Claude Code GAIA structural: strength=$strength samples=$MAX_SAMPLES"
    "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
      --benchmark "$GAIA_BENCHMARK" \
      --phase structural \
      --agent claude_code \
      --model "$MODEL" \
      --repeats "$REPEATS" \
      --limit "$LIMIT" \
      --max-samples "$MAX_SAMPLES" \
      --sandbox "$SANDBOX" \
      --structural-kind gaia \
      --structural-strength "$strength" \
      --no-structural-baseline \
      --no-compute-confidence \
      --log-root "$LOG_ROOT/structural_${strength}" \
      2>&1 | tee "$LOG_ROOT/structural_${strength}.out"
  done
fi
