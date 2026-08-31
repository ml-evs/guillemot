"""Running the agent with something to watch while it works.

`agent.run` returns only once the whole run is over, which on a local model can be
several minutes of silence — indistinguishable from a hang. Iterating the run instead
gives access to the parts as the model streams them, so its reasoning can be shown as
it arrives, along with how full the context window is getting.

Only `run_and_show` is meant to be called from outside; everything else here is about
how the terminal looks.
"""

import json
import os
import sys
from typing import Any

from pydantic_ai import Agent

from guillemot.tracing import record_context_usage


def dim(text: str) -> str:
    """Style text as subordinate to the answer.

    The escapes are dropped when stdout is redirected, where they would be noise in
    the file rather than shading in a terminal.
    """
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


# How much of a tool's arguments and result to show before cutting them off. A .inp file
# passed to `save_topas_inp` would otherwise bury the reasoning it came from.
ARGUMENT_WIDTH = int(os.getenv("GUILLEMOT_TOOL_ARG_WIDTH", 160))
RESULT_WIDTH = int(os.getenv("GUILLEMOT_TOOL_RESULT_WIDTH", 200))


def _shorten(text: str, width: int) -> str:
    """One line, at most `width` characters, with the truncation made obvious."""
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_arguments(part) -> str:
    """The tool's arguments as `name=value` pairs, short enough to sit on one line.

    Arguments arrive either already parsed or as the JSON the model streamed; a partly
    streamed call can leave that JSON unparseable, which is not worth an exception in
    what is only a progress message.
    """
    arguments = part.args
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return _shorten(arguments, ARGUMENT_WIDTH)
    if not isinstance(arguments, dict):
        return _shorten(arguments, ARGUMENT_WIDTH)
    if not arguments:
        return ""

    budget = max(ARGUMENT_WIDTH // len(arguments), 24)
    return ", ".join(f"{k}={_shorten(v, budget)}" for k, v in arguments.items())


def _show_tool_call(part) -> None:
    """Announce a tool call as it is made, so a long run is legible as it happens."""
    print(dim(f"🔧 {part.tool_name}({_format_arguments(part)})"), flush=True)


def _show_tool_result(event) -> None:
    """Say how a tool call turned out. Retries are the interesting case, so say so."""
    from pydantic_ai.messages import RetryPromptPart

    result = event.result
    if isinstance(result, RetryPromptPart):
        complaint = _shorten(result.model_response(), RESULT_WIDTH)
        print(dim(f"   ↩ retrying: {complaint}"), flush=True)
        return
    print(dim(f"   ↳ {_shorten(result.content, RESULT_WIDTH)}"), flush=True)


def _report_context_usage(used: int, limit: int | None) -> None:
    """Show how full the context window is, in the terminal and in the trace."""
    if not used:
        return

    record_context_usage(used, limit)

    if limit:
        message = f"{used:,} / {limit:,} tokens ({used / limit:.0%} of the window)"
    else:
        message = f"{used:,} tokens"

    print(dim(f"📊 Context: {message}"), flush=True)


async def run_and_show(
    agent: Agent, message: Any, context_limit: int | None = None
) -> str:
    """Run the agent, printing its reasoning and tool calls as they arrive, and return
    the answer.

    `context_limit` is how many tokens the model can hold; without it the context
    meter reports tokens but no percentage.
    """
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartDeltaEvent,
        PartStartEvent,
        ThinkingPart,
        ThinkingPartDelta,
    )

    counted = 0  # input tokens already accounted for, to get each request's own total
    thinking = False  # whether a block of reasoning is currently open

    def show_thinking(text: str) -> None:
        """Print reasoning as it streams, opening the block on the first word of it."""
        nonlocal thinking
        if not text:
            return
        if not thinking:
            print("\n💭 Thinking:", flush=True)
            thinking = True
        print(dim(text), end="", flush=True)

    def end_thinking() -> None:
        """Close an open block of reasoning, so what follows starts on a clean line."""
        nonlocal thinking
        if thinking:
            print("\n", flush=True)
            thinking = False

    async with agent.iter(message) as run:
        async for node in run:
            # Tool calls are handled in their own node, after the request that asked for
            # them: this is where the arguments are complete and the results arrive.
            if Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FunctionToolCallEvent):
                            _show_tool_call(event.part)
                        elif isinstance(event, FunctionToolResultEvent):
                            _show_tool_result(event)
                continue

            if not Agent.is_model_request_node(node):
                continue

            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, ThinkingPart
                    ):
                        show_thinking(event.part.content)
                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, ThinkingPartDelta
                    ):
                        show_thinking(event.delta.content_delta or "")
            end_thinking()

            # `run.usage()` accumulates over the whole run, but what fills the window is
            # a single request's prompt: the difference since the last one.
            total = run.usage().input_tokens
            _report_context_usage(total - counted, context_limit)
            counted = total

    return run.result.output
