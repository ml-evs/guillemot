"""How much time a run has left, for tools to pass on to the agent.

A local model spends nearly all of a run generating text — in the first benchmark runs
the tools took seconds out of an hour — so an agent that does not know the clock is
running keeps polishing until it is killed, never giving a final answer. Whoever starts
the run (the benchmark harness) sets the deadline in the environment; the tools report
it in their results, and say so plainly once it is close.

* ``GUILLEMOT_DEADLINE`` — when the run will be stopped, as a Unix timestamp.
* ``GUILLEMOT_TIME_BUDGET_S`` — how long the run was given in total, in seconds.
"""

import os
import time

# Tell the agent to wrap up once less than this share of the budget remains, but never
# with less warning than WRAP_UP_MIN_S: a single reasoning turn can take minutes.
WRAP_UP_FRACTION = 0.25
WRAP_UP_MIN_S = 600


def minutes_remaining() -> float | None:
    """Minutes until the deadline, or None if no deadline was set."""
    deadline = os.getenv("GUILLEMOT_DEADLINE")
    if not deadline:
        return None
    return round(max(0.0, float(deadline) - time.time()) / 60, 1)


def wrap_up_note() -> str | None:
    """A plain instruction to finish, once the deadline is close; otherwise None."""
    remaining = minutes_remaining()
    if remaining is None:
        return None
    budget_s = float(os.getenv("GUILLEMOT_TIME_BUDGET_S") or 0)
    threshold_s = max(WRAP_UP_MIN_S, WRAP_UP_FRACTION * budget_s)
    if remaining * 60 > threshold_s:
        return None
    return (
        f"Only {remaining:.0f} minutes remain before this run is stopped. Do not start "
        "anything new: write your summary now and end it with the FINAL line naming "
        "your best refinement so far."
    )
