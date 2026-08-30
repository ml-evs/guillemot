"""Running the agent with something to watch while it works.

`agent.run` returns only once the whole run is over, which on a local model can be
several minutes of silence — indistinguishable from a hang. Iterating the run instead
gives access to the parts as the model streams them, so its reasoning can be shown as
it arrives, along with how full the context window is getting.

Only `run_and_show` is meant to be called from outside; everything else here is about
how the terminal looks.
"""

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
    """Run the agent, printing its reasoning as it arrives, and return the answer.

    `context_limit` is how many tokens the model can hold; without it the context
    meter reports tokens but no percentage.
    """
    from pydantic_ai.messages import (
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

    async with agent.iter(message) as run:
        async for node in run:
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
            if thinking:
                print("\n", flush=True)  # leave the answer a clean line to start on
                thinking = False

            # `run.usage()` accumulates over the whole run, but what fills the window is
            # a single request's prompt: the difference since the last one.
            total = run.usage().input_tokens
            _report_context_usage(total - counted, context_limit)
            counted = total

    return run.result.output
