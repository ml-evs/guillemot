"""Letting the agent find out what data is actually on disk.

Every other tool takes a path, so anything that reports files has to report paths that
can be handed straight back — relative to the working directory the agent runs in.
"""

import os
import pathlib

from pydantic import BaseModel, Field

from guillemot.session import session_dir

# What each kind of file is for, so the agent does not have to infer it from suffixes.
PATTERN_SUFFIXES = {".xy", ".xye", ".xrdml", ".dat", ".raw", ".txt", ".csv"}
STRUCTURE_SUFFIXES = {".cif"}
INPUT_SUFFIXES = {".inp"}
RESULT_SUFFIXES = {".out", ".png"}

# The shared directories the agent may look in: downloaded sample data and structures.
SEARCH_DIRS = ("data", "cifs")


class DataFile(BaseModel):
    path: str
    kind: str
    size_bytes: int
    sample_id: str | None = None
    in_session: bool = False


class DataInventory(BaseModel):
    """Everything on disk the agent can act on, with usable paths."""

    working_directory: str
    session_directory: str
    patterns: list[DataFile] = Field(default_factory=list)
    structures: list[DataFile] = Field(default_factory=list)
    inputs: list[DataFile] = Field(default_factory=list)
    previous_runs: list[DataFile] = Field(default_factory=list)
    other: list[DataFile] = Field(default_factory=list)
    samples: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _classify(path: pathlib.Path, session: pathlib.Path) -> str:
    # Files sitting directly in the current session directory are this conversation's
    # working set. Anything else under run_dir/ belongs to a finished refinement —
    # including the copies it kept of its own inputs, which would otherwise be offered
    # back as stale duplicates of the real pattern.
    in_session = path.parent == session
    if not in_session and path.parts[0] == "run_dir":
        return "previous_run"

    # The worked examples keep their finished refinements under a `Runs/` directory.
    if "Runs" in path.parts:
        return "previous_run"

    suffix = path.suffix.lower()
    if path.name == "output.txt" or path.name.endswith("_output.txt"):
        return "previous_run"  # a TOPAS Out_X_Yobs_Ycalc export, not a raw pattern
    if suffix in PATTERN_SUFFIXES:
        return "pattern"
    if suffix in STRUCTURE_SUFFIXES:
        return "structure"
    if suffix in INPUT_SUFFIXES:
        return "topas_input"
    if suffix in RESULT_SUFFIXES:
        return "result"
    return "other"


def list_available_data() -> DataInventory:
    """List the data files on disk, with the paths needed to use them.

    Call this first, before assuming anything about what is available or where it
    lives. Every path returned is relative to the working directory and can be passed
    straight to `inspect_xrd_pattern`, `save_topas_inp` or the refinement tools.

    Covers `data/<sample_id>/` (downloaded from datalab), `cifs/`, and this session's
    own directory. Other sessions' directories are not visible.

    This is not the only way to reach a file: if the user gives you a path, pass it
    straight to the tool that needs it. Only say a file is missing if that tool reports
    it missing.
    """
    session = session_dir()
    inventory = DataInventory(
        working_directory=os.getcwd(), session_directory=str(session)
    )

    for directory in (*SEARCH_DIRS, session):
        root = pathlib.Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            # data/<sample_id>/<file> — record which sample a file came from
            sample_id = None
            if root.name == "data" and path.parent != root:
                sample_id = path.parent.name
                if sample_id not in inventory.samples:
                    inventory.samples.append(sample_id)

            entry = DataFile(
                path=str(path),
                kind=_classify(path, session),
                size_bytes=path.stat().st_size,
                sample_id=sample_id,
                in_session=path.parent == session,
            )
            if entry.kind == "pattern":
                inventory.patterns.append(entry)
            elif entry.kind == "structure":
                inventory.structures.append(entry)
            elif entry.kind == "topas_input":
                inventory.inputs.append(entry)
            elif entry.kind == "previous_run":
                inventory.previous_runs.append(entry)
            else:
                inventory.other.append(entry)

    if not inventory.patterns:
        inventory.notes.append(
            "No diffraction patterns found on disk. Use `get_samples` to see what is "
            "in datalab, then `get_sample` to download one, then call this tool again."
        )
    data_dir = pathlib.Path("data")
    if data_dir.is_dir() and (data_dir / "{sample_id}").exists():
        inventory.notes.append(
            "A directory literally named '{sample_id}' exists: something called "
            "`get_sample` with an unsubstituted placeholder. Ignore it."
        )

    return inventory
