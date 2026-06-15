# Structural Robustness (`R_struct`)

Structural robustness — sometimes called **environment robustness**, since for
TauBench we reformat the entire tool/data environment the agent operates in —
measures whether an agent survives **cosmetic, semantics-preserving changes** to
its inputs and tool interface. The correct answer never changes; only *how* the
task, inputs, and tool I/O are presented does. An agent that genuinely understood
the task should be unaffected; one that pattern-matched on exact formatting
degrades.

```
R_struct = perturbed_accuracy / baseline_accuracy
```

Each run pairs an unperturbed baseline with a perturbed run on the *same* tasks
(`reliability_eval/phases/structural.py`), so the ratio is clean. Three strength
presets — `mild` / `medium` / `severe` — escalate how aggressively we deform the
surface.

The two benchmarks deform very different things, because their "interface" is
different:

- **GAIA** — interface is a **natural-language prompt + free-text tool outputs**
  (`hal/utils/gaia_perturbations.py`).
- **TauBench** — interface is a **typed tool / JSON API**
  (`hal/utils/taubench_perturbations.py`).

---

## GAIA — perturbing the question, instructions, and tool text

| Category | Concrete before → after | mild | med | sev |
|----------|------------------------|:---:|:---:|:---:|
| **Question case** | `"What is the population of Paris?"` → lowercase / `MiXeD cAsE` | ✓ (lower) | ✓ | ✓ (mixed) |
| **Whitespace** | collapse runs of spaces to a single space | ✓ | ✓ | ✓ |
| **Noise words** | prepend `"Please"`/`"Kindly"`, append `"Thank you."` | | ✓ | ✓ |
| **Instruction rephrasing** | `"Return only your answer"` → formal `"Please provide exclusively your final answer"` → terse `"Answer only"` | | ✓ (formal) | ✓ (terse) |
| **Reorder instructions** | shuffle the answer-format bullet points | | ✓ | ✓ |
| **Number format** | `10000` → `10,000` (commas) → `three` (words) | | ✓ (commas) | ✓ (words) |
| **Date format** | `2024-01-15` → `January 15, 2024` (verbose) / `20240115` (compact) | | ✓ | ✓ |
| **Tool-output noise** | search results get `[Result 1.]` markers; webpages get `[Navigation: Home > …]` headers + copyright footers | | | ✓ |
| **Response wrapping** | tool output wrapped in `[Response Status: OK]\n[Data Begin]…[Data End]` | | | ✓ |
| **Irrelevant context** | inject filler sentences like `"The weather has been quite variable lately."` | | | ✓ |

Semantics are protected: 4-digit years are excluded from comma-insertion, and
dates are reformatted *before* numbers so date digits aren't word-converted
(`gaia_perturbations.py`).

**Preset summary**

- **mild** — lowercase + whitespace normalization only.
- **medium** — adds noise words, formal instruction rephrasing, bullet reorder,
  comma numbers, verbose dates.
- **severe** — adds mixed case, terse instructions, number-to-words, tool-output
  noise, response wrapping, irrelevant context.

---

## TauBench — perturbing the tool API contract and data formats

This is the more demanding axis. We rename the **parameters the agent must call
tools with**, then transparently reverse the names before hitting the real
environment, so the underlying task is identical but the agent must adapt to a
"different API." The perturbed knowledge base/wiki and every tool response the
agent reads pass through the same engine, so the agent sees a fully consistent
but reformatted world.

| Category | Concrete before → after | mild | med | sev |
|----------|------------------------|:---:|:---:|:---:|
| **Response key case** | `flight_number` → `flightNumber` (camel) / `FlightNumber` (Pascal) | ✓ (camel) | ✓ | ✓ |
| **Tool param names** | agent must call with `fltNo` instead of `flight_number`, `cls` instead of `cabin`, `resId` instead of `reservation_id` | ✓ (camel) | ✓ (camel) | ✓ (abbrev) |
| **Key abbreviations** | `reservation_id`→`res_id`, `first_name`→`fname`, `passengers`→`pax` | | | ✓ |
| **Time format** | `14:00:00` → `2:00 PM` (12h) / `1400` (compact) | | ✓ (12h) | ✓ (compact) |
| **Date format** | `2024-01-15` → `01/15/2024` (US) / `20240115` (compact) | | ✓ (US) | ✓ (compact) |
| **Status values** | `confirmed` → `CONFIRMED` (upper) / `CNF` (abbrev) / `2` (numeric) | | ✓ (upper) | ✓ (abbrev) |
| **Cabin classes** | `basic_economy` → `Basic Economy` (title) / `Y` (code) | | ✓ (title) | ✓ (abbrev) |
| **Response wrapping** | payloads wrapped in `{"status":"success","data":{…}}` | | ✓ | ✓ |
| **Nest/flatten structure** | `first_name`,`last_name` → `name:{first,last}`; or the reverse | | | ✓ (nest) |

**Preset summary**

- **mild** — camelCase response keys + camelCase tool parameter names only.
- **medium** — adds response wrapping, 12h time, US dates, uppercase status,
  title-case cabins.
- **severe** — adds nesting, compact time/date, abbreviated status/cabins, key
  abbreviations, abbreviated tool parameter names.
