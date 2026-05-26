# Inspect-SWE Bridge Slice Tests

These are small, synthetic Inspect tasks for debugging `codex_cli` bridge behavior
without GAIA, reliability campaigns, or large datasets.

They are intended for fast experiments before/after talking with the Inspect team.
Each file isolates one surface.

## Recommended Current Slices

- `bridge_filter_smoke.py`: proves `sandbox_agent_bridge(filter=...)` sees the
  model, messages, tools, tool choice, and config.
- `model_fault_filter_slice.py`: injects a retryable model/API-style fault via
  `GenerateFilter`.
- `bridged_tool_fault_slice.py`: demonstrates the clean Inspect-owned tool
  surface using `BridgedToolsSpec`.
- `inspect_owned_search_fault_slice.py`: recommended replacement path for
  model-native/server-side search. It is self-contained: disables native
  `web_search`, defines a local Inspect-owned search wrapper, and faults it.
- `generate_input_context_fault_slice.py`: recommended context perturbation path
  for Codex-native tool/environment observations. It is self-contained: defines
  a local bridge filter that returns `GenerateInput`.

Run the two post-JJ direction slices from the repo root:

```bash
inspect eval slice_tests/inspect_owned_search_fault_slice.py@inspect_owned_search_fault_slice \
  --model openai/gpt-5.4-2026-03-05 \
  --max-samples 1
```

```bash
inspect eval slice_tests/generate_input_context_fault_slice.py@generate_input_context_fault_slice \
  --model openai/gpt-5.4-2026-03-05 \
  --max-samples 1
```

## Historical Debug Slices

These explain why native `web_search` mutation was rejected as a production
fault surface. Keep them for debugging and discussion, not as recommended
reliability code.

- `native_web_search_observe.py`: observes native Codex `web_search` behavior
  without mutating it.
- `native_web_search_filter_convert_slice.py`: reproduces the filter-based
  native `web_search` mutation attempt. It calls the model from the bridge
  filter and rewrites web-search outputs into failed observations.
- `native_web_search_filter_result_slice.py`: variant of the filter-based
  mutation that writes `web_search failed` into the result content instead of
  setting the tool error field.
- `native_web_search_proxy_patch_slice.py`: reproduces the raw Responses proxy
  patch attempt by marking `web_search_call` output items as failed.

## Common Command Shape

Run from the repo root:

```bash
inspect eval slice_tests/bridge_filter_smoke.py@bridge_filter_smoke \
  --model openai/gpt-5.5-2026-04-23 \
  --max-samples 1
```

All slices are intentionally one sample. If a slice hangs or fails, it is much
easier to inspect than a full GAIA run.

The tasks declare `sandbox="docker"` because `codex_cli` always runs inside an
Inspect sandbox and uses `sandbox_agent_bridge` from there.

