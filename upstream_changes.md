# Upstream Changes Since Reliability Branch Base

Compared branch: `reliability-eval-v1` at `7e3efc0`

Upstream reference: `upstream/main` at `f86c551` (`0.2.63-11-gf86c551`)

Merge base: `bc37fce`

## Summary

Upstream has moved mainly in the agent runtime layer since this reliability
branch split off. The reliability package is branch-local custom code, so
upstream does not contain `src/inspect_swe/reliability/*` or the reliability
docs/tests. The relevant action is to keep the reliability layer compatible with
the updated `claude_code`, `codex_cli`, Gemini, sandbox, and Inspect dependency
APIs.

## Relevant Upstream Commits

- `f86c551` / `7ee1511`: Fix Gemini CLI ACP bridged tools.
- `364504a`: Merge checkpoint-agent-bridge support.
- `4439db5`: Pin `inspect_ai` to GitHub main for checkpointer bridge support.
- `b0879c2`: Resolve consumer outer span lazily at emission time.
- `e56efcf`: Add Claude Code checkpoint support.
- `cc34bcc`: All agents fall back to the home directory when the sandbox default working directory is `/`.
- `83a9bd0`: Codex CLI uses the `releases/latest` endpoint to avoid failures from full release listings.
- `36d583e`: Use checkpointer support in Codex.
- `03c44dc`: Only trace full session output when `debug` is enabled.
- `289973e`: Claude Code reports the real served model instead of the `inspect` sentinel.
- `71733d9`: Codex model name improvements.
- `8f874c0`: Fix Claude Code system prompt duplication on resumed turns.
- `6fe6fa1`, `5a9e15e`, `66b9245`: Inspect dependency updates.
- `8a1e4fa`, `ac66a0e`, `678138c`, `a7fd689`: Codex CLI tool/model/event stream alignment and configuration improvements.
- `7f83c5f`: Gemini CLI explicitly sets auth type to `gemini-api-key`.

## Reliability Impact

- **Claude Code**: The upstream fix for duplicated system prompts on resumed turns is relevant to reliability baseline and structural runs that use `--agent claude_code`. Our reliability code should continue to call `claude_code()` without pinning old versions or rewriting its system prompt behavior locally.
- **Codex CLI**: Upstream changed model naming, event streaming, release discovery, and checkpoint behavior. Reliability tests should catch accidental use of stale `version`/constructor kwargs and ensure fault/structural wrappers still pass only the intended `filter`, bridged tools, and retry settings.
- **Gemini CLI**: Reliability currently supports `gemini_cli` through the generic default solver path. The ACP bridged-tool fix is upstream runtime behavior; no reliability-specific code change is needed unless we add Gemini-specific reliability coverage.
- **Inspect dependency**: Upstream is tested against newer Inspect bridge/checkpoint support than this branch's `pyproject.toml` lower bound. Reliability README should record the exact upstream commit/version used for this assessment.

## Local Follow-Up

- Added mocked regression tests around reliability's Claude and Codex solver construction, so upstream agent API drift is caught without launching live agents.
- Updated reliability README with the upstream commit/version this custom reliability branch has been assessed against.
- No direct merge of upstream into the reliability branch has been performed here; merge/rebase should preserve the branch-local reliability package and resolve upstream runtime conflicts carefully.
