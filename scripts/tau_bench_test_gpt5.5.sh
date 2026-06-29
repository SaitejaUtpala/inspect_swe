#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/tau2_airline_structural_10}"
MODEL="${MODEL:-openai/gpt-5.5}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
LIMIT="${LIMIT:-10}"
REPEATS="${REPEATS:-1}"
MESSAGE_LIMIT="${MESSAGE_LIMIT:-30}"

mkdir -p "$LOG_ROOT"

TAU2="$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import inspect_evals.tau2.tau2 as tau2

print(f"{Path(tau2.__file__)}@tau2_airline")
PY
)"

echo "TAU2=$TAU2"
if [[ -z "$TAU2" ]]; then
  echo "Failed to resolve Tau2 Airline benchmark path. Is inspect_evals installed in this active environment?" >&2
  exit 1
fi
if [[ ! -f "${TAU2%@*}" ]]; then
  echo "Resolved Tau2 module file does not exist: ${TAU2%@*}" >&2
  exit 1
fi

for strength in mild medium severe; do
  echo "Running strength: $strength"

  "$PYTHON_BIN" -m inspect_swe.reliability.cli campaign \
    --benchmark "$TAU2" \
    --phase structural \
    --agent codex_cli \
    --model "$MODEL" \
    --repeats "$REPEATS" \
    --limit "$LIMIT" \
    --max-samples "$MAX_SAMPLES" \
    --sandbox docker \
    --task-arg shuffle=false \
    --structural-kind taubench \
    --structural-strength "$strength" \
    --taubench-codex-adapter \
    --taubench-message-limit "$MESSAGE_LIMIT" \
    --no-structural-baseline \
    --no-compute-confidence \
    --log-root "$LOG_ROOT" \
    2>&1 | tee "$LOG_ROOT/tau2_airline_struct_${strength}_${MAX_SAMPLES}.out"
done
