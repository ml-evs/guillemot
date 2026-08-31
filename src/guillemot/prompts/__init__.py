"""The system prompts, kept as text files so they can be edited and swapped.

Prompt engineering means comparing wordings, which is awkward when the prompt is a
triple-quoted string in the middle of the application. Each variant here is one
markdown file in this directory; pick one with ``GUILLEMOT_PROMPT`` or ``--prompt``,
and a path to a file outside the package works too, so a throwaway variant does not
have to be committed to be tried.

Placeholders are ``{{name}}`` rather than `str.format`'s braces, because the prompt
carries an example TOPAS .inp file and TOPAS syntax is full of braces of its own.
Every placeholder must be filled: an unfilled one is a mistake in a prompt that will
otherwise be sent to the model with `{{topas_example}}` in it, which is worse than a
loud failure at start-up.
"""

import os
import pathlib
import re

PROMPT_DIR = pathlib.Path(__file__).parent
FRAGMENT_DIR = PROMPT_DIR / "fragments"

DEFAULT_PROMPT = "default"
SUFFIX = ".md"

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptNotFound(LookupError):
    """The named prompt is neither a variant in this package nor a file on disk."""


def available_prompts() -> list[str]:
    """The names of the prompt variants shipped with guillemot."""
    return sorted(path.stem for path in PROMPT_DIR.glob(f"*{SUFFIX}"))


def selected_prompt_name() -> str:
    """Which prompt to use, unless the caller has been told otherwise."""
    return os.getenv("GUILLEMOT_PROMPT") or DEFAULT_PROMPT


def prompt_path(name: str) -> pathlib.Path:
    """Resolve a prompt name to a file: a shipped variant, or a path to one of your own."""
    packaged = PROMPT_DIR / f"{name}{SUFFIX}"
    if packaged.is_file():
        return packaged

    candidate = pathlib.Path(name).expanduser()
    if candidate.is_file():
        return candidate

    raise PromptNotFound(
        f"No prompt named {name!r}. Built-in prompts: {', '.join(available_prompts())}. "
        "A path to a markdown file also works."
    )


def load_fragment(name: str) -> str:
    """Read a reusable chunk of prompt — the bits that vary with how guillemot is set up."""
    return (FRAGMENT_DIR / f"{name}{SUFFIX}").read_text().strip()


def fill(template: str, **context: str) -> str:
    """Substitute ``{{name}}`` placeholders, refusing to leave any behind."""
    missing: set[str] = set()

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key not in context:
            missing.add(key)
            return match.group(0)
        return context[key]

    filled = _PLACEHOLDER.sub(substitute, template)
    if missing:
        raise KeyError(
            f"Prompt has placeholders nothing was given for: {', '.join(sorted(missing))}"
        )
    return filled


def render_prompt(name: str | None = None, **context: str) -> str:
    """The system prompt to run with, ready to hand to the agent."""
    name = name or selected_prompt_name()
    return fill(prompt_path(name).read_text(), **context)
