"""Letting the agent find out what data is actually on disk.

Every other tool takes a path, so anything that reports files has to report paths that
can be handed straight back — relative to the working directory the agent runs in.
"""

import os
import pathlib

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

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
    lives.

    Every path returned is relative to the working directory and can be passed
    straight to the refinement tools.

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


# Where `read_file` may read: this session's own files, the shared sample data and
# structures, and the worked examples. Not other sessions (another benchmark entry's
# answer lives there), and not `datasets/` (the labels).
READABLE_DIRS = (*SEARCH_DIRS, "examples")
READ_DEFAULT_LINES = 200
READ_MAX_LINES = 500
READ_MAX_LINE_WIDTH = 400


class FileExcerpt(BaseModel):
    """Some lines of a text file, numbered as TOPAS numbers them in its errors.

    For diffraction data the text is a summary of the pattern instead of its lines.
    """

    path: str
    kind: str = "text"  # or "pattern_summary"
    total_lines: int
    first_line: int
    last_line: int
    text: str
    truncated: bool = Field(
        description="True if the file has lines after last_line; ask again from there."
    )


def _readable_roots() -> list[pathlib.Path]:
    return [session_dir().resolve()] + [
        pathlib.Path(d).resolve() for d in READABLE_DIRS if pathlib.Path(d).is_dir()
    ]


def read_file(
    path: str, start_line: int = 1, max_lines: int = READ_DEFAULT_LINES
) -> FileExcerpt:
    """Read a text file, with line numbers, so you can check exactly what it says.

    Diffraction data — a measured pattern, or a refinement's `_output.txt` — comes back
    as a summary instead: its range, step, intensities and strongest peaks, and for a
    refinement output the Rwp and where the fit is worst. Reading thousands of numbers
    yourself is slow and error-prone; to study peaks, run a refinement and plot it.

    Use it to look at the .inp you last saved before changing it, at the `.out` file of
    a finished refinement (the .inp with every refined value filled in), at a CIF, or at
    a worked example in `examples/`. Line numbers match the ones in TOPAS error
    messages.

    You can read files in this session's directory (including the run directories
    synced back from TOPAS), `data/`, `cifs/` and `examples/`. Long files are returned
    a page at a time: if `truncated` is true, call again with `start_line` set to one
    past `last_line`. Images cannot be read here; use `plot_refinement_results`.

    Args:
        path: the file to read, as other tools reported it.
        start_line: the first line to return, counting from 1.
        max_lines: how many lines to return, at most 500.
    """
    target = pathlib.Path(path)
    resolved = target.resolve()
    if not any(resolved.is_relative_to(root) for root in _readable_roots()):
        raise ModelRetry(
            f"{path} is outside the files you can read: this session's directory "
            f"({session_dir()}), {', '.join(f'{d}/' for d in READABLE_DIRS)}."
        )
    if not resolved.is_file():
        raise ModelRetry(
            f"No file at {path}. Call `list_available_data` to see what exists, and use "
            "the paths exactly as it reports them."
        )

    data = resolved.read_bytes()
    if b"\0" in data[:8192]:
        raise ModelRetry(
            f"{path} is not a text file. To look at a refinement plot, use "
            "`plot_refinement_results`."
        )

    lines = data.decode("utf-8", errors="replace").splitlines()
    columns = _numeric_columns(lines)
    if columns is not None:
        return FileExcerpt(
            path=path,
            kind="pattern_summary",
            total_lines=len(lines),
            first_line=1,
            last_line=len(lines),
            text=_summarise_pattern(columns),
            truncated=False,
        )

    start = max(1, start_line)
    count = max(1, min(max_lines, READ_MAX_LINES))
    chosen = lines[start - 1 : start - 1 + count]
    width = len(str(start + len(chosen)))

    def clip(line: str) -> str:
        if len(line) <= READ_MAX_LINE_WIDTH:
            return line
        return (
            line[:READ_MAX_LINE_WIDTH]
            + f" …[{len(line) - READ_MAX_LINE_WIDTH} more characters]"
        )

    numbered = "\n".join(
        f"{n:>{width}} | {clip(line)}" for n, line in enumerate(chosen, start)
    )
    last = start + len(chosen) - 1
    return FileExcerpt(
        path=path,
        total_lines=len(lines),
        first_line=start,
        last_line=max(last, start - 1),
        text=numbered,
        truncated=last < len(lines),
    )


# A file is treated as diffraction data when it has at least this many lines and
# nearly all of them are 2-4 columns of numbers.
PATTERN_MIN_LINES = 50
PATTERN_NUMERIC_FRACTION = 0.95
PATTERN_TOP_PEAKS = 12


def _numeric_columns(lines: list[str]):
    """The data as an array of columns, if these lines are a pattern; else None."""
    import numpy as np

    rows, other = [], 0
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        try:
            values = [float(f) for f in fields]
        except ValueError:
            other += 1
            continue
        if 2 <= len(values) <= 4:
            rows.append(values[:3])
        else:
            other += 1
    if len(rows) < PATTERN_MIN_LINES or other > (1 - PATTERN_NUMERIC_FRACTION) * len(
        rows
    ):
        return None
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows]).T


def _summarise_pattern(columns) -> str:
    """What a diffractionist would want to know about a pattern at a glance."""
    import numpy as np
    from scipy.signal import find_peaks

    x, y = columns[0], columns[1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    step = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    lines = [
        f"Diffraction data: {len(x)} points, 2θ {x[0]:.3f}–{x[-1]:.3f}° (step {step:.4f}°).",
        f"Observed intensity: min {y.min():.4g}, median {np.median(y):.4g}, max {y.max():.4g}.",
    ]

    floor = np.median(y)
    found, props = find_peaks(y, prominence=0.03 * max(y.max() - floor, 1e-12))
    strongest = sorted(found[np.argsort(y[found])[::-1][:PATTERN_TOP_PEAKS]])
    if strongest:
        listed = ", ".join(f"{x[i]:.2f}° ({y[i]:.4g})" for i in strongest)
        lines.append(f"Strongest peaks, by 2θ (intensity): {listed}.")

    if columns.shape[0] >= 3:
        ycalc = columns[2][order]
        diff = y - ycalc
        if np.all(y > 0):
            rwp = 100 * np.sqrt(np.sum(diff**2 / y) / np.sum(y))
            lines.append(f"Fit: Rwp {rwp:.2f}% (weights 1/yobs) over the whole range.")
        worst = sorted(np.argsort(np.abs(diff))[::-1][:5])
        listed = ", ".join(
            f"{x[i]:.2f}° (obs {y[i]:.4g}, calc {ycalc[i]:.4g})" for i in worst
        )
        lines.append(f"Largest misfits: {listed}.")

    lines.append(
        "This is a summary, not the data: to examine peaks or the fit, run a refinement "
        "and look at it with `plot_refinement_results`."
    )
    return "\n".join(lines)
