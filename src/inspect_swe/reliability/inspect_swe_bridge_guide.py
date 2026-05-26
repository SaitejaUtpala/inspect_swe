"""Meeting guide: how Inspect-SWE agents use Inspect's bridge.

This file is intentionally educational. It is not imported by the reliability
package. It explains the moving parts that matter for a reliability/fault
tolerance discussion with the Inspect team.

High-level mental model
=======================

Inspect-SWE agents such as ``codex_cli`` are not normal in-process Python
agents. They are command-line agents running inside an Inspect sandbox. Inspect
connects them to the selected model through ``sandbox_agent_bridge``.

The important flow is:

    Inspect eval sample
      -> Inspect-SWE task solver/agent
      -> ``codex_cli`` agent wrapper
      -> ``sandbox_agent_bridge(...)``
      -> model proxy inside the sandbox
      -> OpenAI Responses / Anthropic / Google-compatible request
      -> Inspect model provider
      -> response converted back into the agent's protocol
      -> Codex continues or exits

That bridge is the reason Inspect can run external agents while still logging
messages, tool schemas, model usage, scores, and transcript events in ``.eval``.


Key files
=========

Inspect-SWE side:

* ``inspect_swe/_codex_cli/codex_cli.py``
  Defines the ``codex_cli`` agent. It starts Codex in the sandbox and wires it
  to Inspect through ``sandbox_agent_bridge``.

* ``inspect_swe/reliability/fault.py``
  Runs the fault phase. It creates ``FaultEnvironment`` and passes its
  ``model_filter()`` into agents like ``codex_cli``.

* ``inspect_swe/reliability/faults.py``
  Defines the current fault primitives. After cleanup, it only uses
  Inspect-controlled surfaces:

  * model/message perturbation through ``GenerateFilter``
  * explicit Inspect tools exposed through ``BridgedToolsSpec``

Inspect-AI side:

* ``inspect_ai/agent/_bridge/sandbox/bridge.py``
  Defines ``sandbox_agent_bridge``. It starts the sandbox model proxy, registers
  bridged MCP tools, and yields a ``SandboxAgentBridge`` object.

* ``inspect_ai/agent/_bridge/util.py``
  Defines ``bridge_generate``. This is where JJ's suggested ``filter`` runs.
  The filter can:

  * return ``None`` to let normal generation continue
  * return ``ModelOutput`` to replace the model result
  * return ``GenerateInput`` to replace the messages/tools/config sent onward

* ``inspect_ai/agent/_bridge/responses_impl.py``
  Converts OpenAI Responses requests into Inspect messages/tools and converts
  Inspect model outputs back into Responses objects.

* ``inspect_ai/agent/_bridge/sandbox/service.py``
  Exposes sandbox service methods such as ``generate_responses``,
  ``list_tools``, and ``call_tool``.


How ``codex_cli`` uses the bridge
=================================

In ``inspect_swe/_codex_cli/codex_cli.py``, the agent does roughly:

1. Pick a per-sample proxy port.
2. Start ``sandbox_agent_bridge`` with:

   * current ``AgentState``
   * model / model aliases
   * optional ``filter``
   * optional ``bridged_tools``
   * retry/refusal settings

3. Install or locate the Codex binary.
4. Write Codex config and optional MCP server config.
5. Set ``OPENAI_BASE_URL`` to the bridge proxy:

   ``http://localhost:{bridge.port}/v1``

6. Run ``codex exec ...`` inside the sandbox.

Codex thinks it is talking to an OpenAI-compatible endpoint. In reality, the
request goes through Inspect's sandbox model proxy, then back to the active
Inspect model provider.


What the bridge filter can and cannot do
========================================

The bridge filter is a ``GenerateFilter`` passed into ``sandbox_agent_bridge``.
It runs inside ``bridge_generate`` before the model generation is performed.

Signature, conceptually:

    async def filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        ...

Good uses:

* simulate retryable model/API instability by returning a controlled
  ``ModelOutput``
* delay model calls
* perturb prompts/messages before generation
* replace generation inputs in a typed way with ``GenerateInput``

Hard uses:

* fault a tool after it executes
* change a Codex-native tool result
* make native ``web_search`` fail while preserving protocol correctness

The filter sees the model generation boundary. It does not automatically own
the agent's native tool execution boundary.


Three tool categories
=====================

1. Inspect-owned tools
----------------------

These are explicit Inspect tools exposed through ``BridgedToolsSpec``. Inspect
owns the Python function and the result path. These are the easiest and cleanest
fault surface.

Flow:

    Codex calls MCP tool
      -> bridge service ``call_tool(server, tool, args)``
      -> Python ``Tool`` function runs
      -> result returned to Codex

This is what ``FaultEnvironment.wrap_bridged_tools`` can safely wrap. It can
inject:

* ``ToolError``
* delay
* malformed output
* empty output

This is also closest to how HAL-style harnesses work. In ``hal-harness`` /
``hal-generalist``, agents are relatively primitive: the harness owns the
action/tool/observation loop. Since the harness executes the tool and returns
the observation, fault injection is straightforward.


2. Codex-native tools
---------------------

Examples seen in Codex model events:

* ``exec_command``
* ``apply_patch``
* ``update_plan``
* ``write_stdin``
* ``view_image``

These are visible to Inspect as model tools, but they are declared and executed
by Codex's own runtime. They are not Python ``Tool`` objects owned by
``inspect_swe``.

Why this matters:

* Inspect can log the schema and messages.
* ``inspect_swe`` does not currently have a clean wrapper around the actual
  execution/result path.
* Faulting these from ``inspect_swe`` would require mutating model outputs,
  Codex runtime behavior, or bridge/protocol artifacts.


3. Model/provider-native tools
------------------------------

Example:
* ``web_search``

In Codex GAIA runs, ``web_search`` is enabled with Codex's
``--enable web_search_request`` and flows through OpenAI Responses server-tool
semantics. It may appear internally as things like ``web_search_call`` or
``ContentToolUse(tool_type='web_search')``.

This is not a normal Python function in ``inspect_swe``. It is provider/server
tool behavior surfaced through the Inspect bridge.


What was tried for native ``web_search``
========================================

Several prototypes attempted to fault native ``web_search`` from
``inspect_swe``:

1. Mutate raw Responses / bridge payloads.
   This was faithful in spirit, but schema-sensitive. Small changes produced
   invalid Responses payloads.

2. Mark ``web_search_call`` as failed.
   This got closer, but could surface as ``unsupported call: web_search`` rather
   than a realistic tool failure.

3. Convert Codex function-style ``web_search`` calls into failed native-looking
   web-search results.
   This required Codex/Responses-specific manipulation, including valid
   ``ws_...`` IDs. It produced injected fault metadata in small runs, but the
   behavior was often terminal: Codex did not continue, retry, or recover after
   the injected failure.

Those prototypes proved the bridge is conceptually the right layer, but also
showed that doing it from ``inspect_swe`` by reshaping protocol artifacts is
fragile.


Current reliability fault design
================================

The current cleaned-up ``FaultEnvironment`` intentionally keeps only
Inspect-controlled surfaces:

* model faults through ``GenerateFilter``
* message/prompt faults through the solver wrapper
* Inspect-owned bridged tool faults through ``BridgedToolsSpec`` wrappers

It no longer tries to fault native ``web_search`` or Codex-native tools from
inside ``inspect_swe``.


Questions for the Inspect meeting
=================================

1. Is ``GenerateFilter`` intended to be sufficient for this reliability use
   case, or is it mainly a model-generation boundary?

2. Is there, or should there be, a typed hook around native tool
   execution/results in the sandbox agent bridge?

3. For Codex-native tools like ``exec_command`` and ``apply_patch``, where is
   the cleanest boundary to inject a recoverable failure?

4. For provider-native tools like ``web_search``, can Inspect expose a stable
   representation before serializing back to Responses/Codex?

5. What is the recommended way to ensure an injected fault is:

   * protocol-valid
   * logged in ``.eval``
   * visible to the agent as a recoverable observation
   * not a harness crash or unsupported-tool artifact


One-sentence summary
====================

Faulting Inspect-owned tools is clean because Inspect owns the function and
result path; faulting Codex-native or provider-native tools is hard because
``inspect_swe`` currently sees bridge artifacts, not a stable typed native-tool
execution boundary.
"""

