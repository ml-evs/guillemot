"""Per-conversation working directories.

Each conversation gets its own directory under `run_dir/`, so the files belonging to
one refinement session stay together and cannot be confused with those of an earlier
one. Everything the agent writes — .inp files, refinement outputs, plots — lands
there, and the data files it decides to work on are copied in alongside them.

The session is started once, by the application, rather than left to the model to
remember. Tools call `session_dir()`, which falls back to `run_dir/` itself when no
session has been started, so the tools remain usable outside the chat application.
"""

import pathlib
import shutil
import uuid
from datetime import datetime
from typing import Optional

RUN_DIR = pathlib.Path("run_dir")

_current: Optional["Session"] = None


class Session:
    """One conversation's working directory."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.directory = RUN_DIR / session_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Session({self.session_id!r})"


def new_session_id() -> str:
    """A sortable, unique name: when the session ran, plus a hash to keep it distinct."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def start_session(session_id: Optional[str] = None) -> Session:
    """Begin a new session and make it current. Called once per conversation."""
    global _current
    _current = Session(session_id or new_session_id())
    return _current


def current_session() -> Optional[Session]:
    return _current


def end_session() -> None:
    """Forget the current session, so later work falls back to `run_dir/`."""
    global _current
    _current = None


def session_dir() -> pathlib.Path:
    """Where the tools should write.

    The current session's directory, or `run_dir/` when no session has been started —
    which keeps the tools working when used directly rather than through the chat
    application.
    """
    if _current is not None:
        return _current.directory
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR


def copy_into_session(paths: list[str]) -> list[str]:
    """Copy data files into the current session directory, and return their new paths.

    Use this to bring the pattern (and any CIFs) you are going to work with into the
    session directory, so everything for this refinement sits in one place. The
    originals are left where they are, so other sessions can still use them. Pass the
    paths exactly as `list_available_data` reported them.
    """
    from pydantic_ai.exceptions import ModelRetry

    destination = session_dir()
    copied: list[str] = []

    for path in paths:
        source = pathlib.Path(path)
        if not source.is_file():
            raise ModelRetry(
                f"No file at {path}. Call `list_available_data` to see what exists and "
                "use the paths exactly as it reports them."
            )
        if not source.resolve().is_relative_to(pathlib.Path.cwd().resolve()):
            raise ModelRetry(
                f"Refusing to copy {path}: only files inside the working directory can "
                "be brought into the session."
            )

        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(str(target))

    return copied
