# inspect_swe_docs

Source: [https://meridianlabs-ai.github.io/inspect_swe/](https://meridianlabs-ai.github.io/inspect_swe/)

Inspect SWE

## Overview

The `inspect_swe` package makes software engineering agents like [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview), [Codex CLI](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli), [OpenCode](https://github.com/anomalyco/opencode), and [Mini SWE Agent](https://github.com/SWE-agent/mini-swe-agent) available as standard Inspect agents. For example, here we use the `claude_code()` agent as the solver in an Inspect task:

```
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import claude_code

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=claude_code(),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

Inspect SWE agents are implemented using the Inspect [sandbox_agent_bridge()](https://inspect.aisi.org.uk/agent-bridge.html#sandbox-bridge).

Agents run inside the sample sandbox and their model API calls are proxied back to Inspect. This means that you can use any model with Inspect SWE agents, and that features like token or time limits and log transcripts work as normal with the agents.

## Getting Started

Install Inspect SWE from PyPI with:

```
pip install inspect-swe
```

Then, try out one or more of the available agents:

| Agent | Description |
| --- | --- |
| [claude_code()](https://meridianlabs-ai.github.io/inspect_swe/claude_code.html) | Anthropic's agentic coding tool [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) |
| [codex_cli()](https://meridianlabs-ai.github.io/inspect_swe/codex_cli.html) | OpenAI's terminal-based coding agent [Codex CLI](https://github.com/openai/codex) |
| [gemini_cli()](https://meridianlabs-ai.github.io/inspect_swe/gemini_cli.html) | Google's open-source AI agent [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
| [opencode()](https://meridianlabs-ai.github.io/inspect_swe/opencode.html) | Provider-independent terminal-based coding agent. |
| [mini_swe_agent()](https://meridianlabs-ai.github.io/inspect_swe/mini_swe_agent.html) | SWE-agent's minimal 100-line agent. |
