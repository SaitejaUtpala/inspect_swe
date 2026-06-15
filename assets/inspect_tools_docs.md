# inspect_tools_docs

Source: [https://inspect.aisi.org.uk/tools.html](https://inspect.aisi.org.uk/tools.html)

Tool Basics - Inspect

## Overview

Many models now have the ability to interact with client-side Python functions in order to expand their capabilities. This enables you to equip models with your own set of custom tools so they can perform a wider variety of tasks.

Inspect natively supports registering Python functions as tools and providing these tools to models that support them. Inspect also includes several standard tools for code execution, text editing, computer use, web search, and web browsing.

Note Tools and Agents

One application of tools is to run them within an agent scaffold that pursues an objective over multiple interactions with a model. The scaffold uses the model to help make decisions about which tools to use and when, and orchestrates calls to the model to use the tools. This is covered in more depth in the [Agents](https://inspect.aisi.org.uk/agents.html) section.

## Standard Tools

Inspect has built-in tools for computing and agentic planning. Computing tools include:

- [Web Search](https://inspect.aisi.org.uk/tools-standard.html#sec-web-search), which uses a search provider (either built in to the model or external) to execute and summarize web searches.
- [Bash and Python](https://inspect.aisi.org.uk/tools-standard.html#sec-bash-and-python) for executing arbitrary shell and Python code.
- [Bash Session](https://inspect.aisi.org.uk/tools-standard.html#sec-bash-session) for creating a stateful bash shell that retains its state across calls from the model.
- [Text Editor](https://inspect.aisi.org.uk/tools-standard.html#sec-text-editor) which enables viewing, creating and editing text files.
- [Computer](https://inspect.aisi.org.uk/tools-standard.html#sec-computer), which provides the model with a desktop computer viewed through screenshots that supports mouse and keyboard interaction.
- [Code Execution](https://inspect.aisi.org.uk/tools-standard.html#sec-code-execution), which gives models a sandboxed Python code execution environment running within the model provider's infrastructure.
- [Web Browser](https://inspect.aisi.org.uk/tools-standard.html#sec-web-browser), which provides the model with a headless Chromium web browser that supports navigation, history, and mouse/keyboard interactions.

Agentic tools include:

- [Skill](https://inspect.aisi.org.uk/tools-standard.html#sec-skill), which provides agent skill specifications to the model with specialized knowledge and expertise for specific tasks.
- [Update Plan](https://inspect.aisi.org.uk/tools-standard.html#sec-update-plan), which helps the model track steps and progress across longer horizon tasks.
- [Memory](https://inspect.aisi.org.uk/tools-standard.html#sec-memory), which enables storing and retrieving information through a memory file directory.
- [Think](https://inspect.aisi.org.uk/tools-standard.html#sec-think), which provides models the ability to include an additional thinking step as part of getting to its final answer.

If you are only interested in using the standard tools, check out their respective documentation links above. To learn more about creating your own tools read on below.

## MCP Tools

The [Model Context Protocol](https://modelcontextprotocol.io/introduction) is a standard way to provide capabilities to LLMs. There are hundreds of [MCP Servers](https://github.com/modelcontextprotocol/servers) that provide tools for a myriad of purposes including web search and browsing, filesystem interaction, database access, git, and more.

Tools exposed by MCP servers can be easily integrated into Inspect. Learn more in the article on [MCP Tools](https://inspect.aisi.org.uk/tools-mcp.html).

## Custom Tools

Here's a simple tool that adds two numbers. The `@tool` decorator is used to register it with the system:

```python
from inspect_ai.tool import tool

@tool
def add():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y

    return execute
```

### Annotations

Type annotations and descriptions are required for tool declarations so that the model can be informed which types to pass back to the tool function and what the purpose of each parameter is.

## Using Tools

We can use the `addition()` tool in an evaluation by passing it to the [use_tools()](https://inspect.aisi.org.uk/reference/inspect_ai.solver.html#use_tools) solver:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, use_tools
from inspect_ai.scorer import match

@task
def addition_problem():
    return Task(
        dataset=[Sample(input="What is 1 + 1?", target=["2"])],
        solver=[
            use_tools(add()),
            generate()
        ],
        scorer=match(numeric=True),
    )
```

When using tools with models, the models do not call the Python function directly. Rather, the model generates a structured request which includes function parameters, and then Inspect calls the function and returns the result to the model.

See the Custom Tools article for details on more advanced custom tool features including sandboxing, error handling, and dynamic tool definitions.
