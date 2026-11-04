import fcntl
import hashlib
import os
import pathlib
import re
import select
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from os.path import join
from typing import Optional

from guillemot.deadline import minutes_remaining, wrap_up_note
from guillemot.session import session_dir
from guillemot.tools.plotting import render_refinement_plot
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_ai.exceptions import ModelRetry

RUN_DIR = "run_dir"  # fallback root when no session has been started

# Default location of the TOPAS console executable on the remote Windows machine.
DEFAULT_TOPAS_EXE = r"C:\Science\Topas-7\tc.exe"
# Where run directories get created on the remote machine.
DEFAULT_REMOTE_BASE_DIR = r"C:\guillemot_runs"
# Process names that indicate TOPAS is already busy on the remote machine: the console
# binary, and the GUI, which is `ta.exe` (versioned as e.g. `ta61.exe` in TOPAS 6).
TOPAS_PROCESS_NAMES = ("tc.exe", "ta.exe", "ta61.exe", "topas.exe")

# Filenames are interpolated into a remote `cmd /c` string, where a quote or an
# ampersand would end the command and start another one. Keep them boring.
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_safe_filename(name: str, what: str = "filename") -> str:
    """Reject anything that is not a plain, single-component filename."""
    if name != os.path.basename(name) or not SAFE_FILENAME_RE.fullmatch(name):
        raise ModelRetry(
            f"{what} {name!r} is not usable: it must be a plain filename made of "
            "letters, digits, dots, underscores or hyphens, with no directory "
            "component. Choose a simple name like 'KD1_refine' and try again."
        )
    return name


# TOPAS reports a parse failure as a line number plus the token it choked on, e.g.
#
#     *** Error loading sstring_in
#         at LINE 42
#     *** Error at: 1.0E-4
#
# Neither half is reliable on its own. The line number is where the parser gave up,
# which may be the line *after* the mistake: `scale @ 1.0E-4` written on line 41 was
# reported as line 42, while `Phase_Density_g_on_cm3( @ ... )` on line 43 was reported
# exactly. So the agent gets a window around the reported line with the token named,
# rather than a single line asserted to be the culprit — it has no other way to see
# the file it wrote, and counting lines from memory is how the last session misread
# one of these and wasted a whole remote run on the wrong fix.
TOPAS_ERROR_LINE_RE = re.compile(r"^\s*at LINE\s+(\d+)", re.MULTILINE)
TOPAS_ERROR_TOKEN_RE = re.compile(r"^\s*\*\*\* Error at:\s*(\S.*?)\s*$", re.MULTILINE)


def _read_text(path) -> str:
    """The .inp as text, or "" if it cannot be read — never a reason to fail harder."""
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _inp_excerpt(inp_text: str, topas_output: str, name: str, context: int = 5) -> str:
    """Quote the lines of an .inp around the parse error TOPAS reported, if it did.

    Returns "" when the output carries no line number — a refinement can fail for
    reasons that have nothing to do with the file's syntax, and a misleading excerpt
    would be worse than none.
    """
    match = TOPAS_ERROR_LINE_RE.search(topas_output)
    if not match:
        return ""

    lines = inp_text.splitlines()
    reported = int(match.group(1))
    if not 1 <= reported <= len(lines):
        return ""

    token_match = TOPAS_ERROR_TOKEN_RE.search(topas_output)
    token = token_match.group(1) if token_match else None
    # A one-character token like `@` occurs on half the lines in a typical .inp, so it
    # is worth naming but not worth marking.
    markable = token if token and len(token) > 1 else None

    first = max(1, reported - context)
    last = min(len(lines), reported + context)
    width = len(str(last))

    quoted = []
    for number in range(first, last + 1):
        line = lines[number - 1]
        if number == reported:
            marker = ">"
        elif markable and markable in line:
            marker = "*"
        else:
            marker = " "
        quoted.append(f" {marker} {number:>{width}} | {line}")

    header = f"TOPAS stopped parsing {name} at line {reported}"
    if token:
        header += f", on the token `{token}`"
    header += (
        ". That is where its parser gave up, so the mistake is on that line or just "
        "before it"
    )
    header += " (marked *):" if markable else ":"

    return f"\n\n{header}\n\n" + "\n".join(quoted)


class SaveInpResult(BaseModel):
    inp_path: str
    line_count: int


def save_topas_inp(filename: str, inp_text: str) -> SaveInpResult:
    """
    A tool that writes a topas .inp file to ./run_dir/ and does some basic checks.
    The AI model must make sure that the .inp file includes the export lines:
    Out_X_Yobs_Ycalc("<filename>_output.txt")

    Each str block should also have:
    Out_CIF_STR("<filename>_<phase_name>.cif"). Note that the input filename (without ".inp")

    should prefix the phase name.

    Blocks are only defined by whitespace and do not need closing tags.

    The input pattern can be assumed to be in the same directory that TOPAS runs from and in the same dir as input file is.
    """
    basename = _check_safe_filename(os.path.splitext(filename)[0], "input filename")

    inp_path = join(str(session_dir()), f"{basename}.inp")

    # Build the expected macro string, then check presence
    output_macro_text = f'Out_X_Yobs_Ycalc("{basename}_output.txt")'
    if output_macro_text not in inp_text:
        raise ModelRetry(
            message=f"input file doesn't contain the correct output macro: {output_macro_text}. Please try again."
        )

    with open(inp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(inp_text)
    lines = inp_text.splitlines()

    return SaveInpResult(
        inp_path=inp_path,
        line_count=len(lines),
    )


class RunRefinementResult(BaseModel):
    status: str  # "success" | "failure" | "timeout"
    stdout: str
    stderr: str
    outfile_path: Optional[str]
    outfile_contents: Optional[str]
    refinement_result_path: Optional[str]
    logs_tail: Optional[str]
    plot_path: Optional[str]
    # aux_files: list[str] = Field(default_factory=list)  # any xy/csv exports found
    rwp: Optional[float] = Field(
        default=None,
        description="Rwp in percent over the whole pattern, computed from the output file.",
    )
    refined_values: list[str] = Field(
        default_factory=list,
        description="Each changed line of the input, as `start  ->  refined`.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Refined values that look wrong. Deal with these before anything else.",
    )
    time_remaining_minutes: Optional[float] = Field(
        default=None,
        description="Minutes left before this run is stopped, when there is a deadline.",
    )


def _collect_refinement_outputs(inp_path: str) -> dict:
    """Gather the output files that TOPAS leaves next to ``inp_path`` and plot them.

    The plot is written to disk but not returned as an image: see the note at
    ``plot_path`` below.

    ``inp_path`` is a *local* path: for remote runs this points at the copy of the
    input inside the synced-back run directory.
    """
    outfile_path = str(pathlib.Path(inp_path.replace(".inp", ".out")))

    if os.path.isfile(outfile_path):
        with open(outfile_path) as f:
            outfile_contents = f.read()
    else:
        outfile_contents = None
        outfile_path = None

    refinement_result_path = inp_path.replace(".inp", "_output.txt")
    # Check if the refinement result file exists
    if not os.path.isfile(refinement_result_path):
        refinement_result_path = None

    hkl_file = inp_path.replace(".inp", "_hkl.txt")
    if not os.path.isfile(hkl_file):
        hkl_file = None

    # Draw the fit and leave the PNG next to the rest of the run, but hand back only
    # its path: the image goes to the model when it asks for it with
    # `plot_refinement_results`, not unbidden on the back of every refinement.
    plot_path = None
    if refinement_result_path is not None:
        plot_path = render_refinement_plot(
            output_file=refinement_result_path,
            save_path=refinement_result_path.replace("_output.txt", "_plot.png"),
            hkl_file=hkl_file,
        )

    refined_values, warnings = (
        refinement_report(_read_text(inp_path), outfile_contents)
        if outfile_contents
        else ([], [])
    )

    if note := wrap_up_note():
        warnings = [note] + warnings

    return {
        "outfile_path": outfile_path,
        "outfile_contents": outfile_contents,
        "refinement_result_path": refinement_result_path,
        "plot_path": plot_path,
        "rwp": _rwp_from_output(refinement_result_path),
        "refined_values": refined_values,
        "warnings": warnings,
        "time_remaining_minutes": minutes_remaining(),
    }

# A number as TOPAS writes it in a .out: the value, then a backtick if it was refined,
# then optionally `_esd` and/or a `_LIMIT_MIN_#`/`_LIMIT_MAX_#` flag.
_OUT_NUMBER_RE = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(?:`(?:_[-+0-9.eE]+)?)?")
_LIMIT_RE = re.compile(r"_LIMIT_(MIN|MAX)_(-?[0-9.eE+-]+)")
_LATTICE_RE = re.compile(r"^\s*(a|b|c)\s+(?:!?[A-Za-z_]\w*\s+|@\s*)?(-?[0-9.]+)")
_BEQ_RE = re.compile(r"\bbeq\s+(?:!?[A-Za-z_]\w*\s+|@\s*)?(-?[0-9.]+)")
_SIZE_RE = re.compile(r"\b(CS_L|CS_G|CS|Crystallite_Size)\(\s*[^,]*,\s*(-?[0-9.]+)")
_SPACE_GROUP_RE = re.compile(r"^\s*space_group\s+(\S+)")

# Relative change in a lattice parameter beyond which a refinement has probably gone
# astray rather than converged: a wrong peak assignment, or a zero/displacement error
# absorbed into the cell.
LATTICE_JUMP = 0.02
# Crystallite sizes are in nm with a floor of 0.3; anything this small means peaks so
# broad the refinement cannot see them — usually a starting value in the wrong units.
SMALL_CRYSTALLITE_NM = 2.0
MAX_REPORTED_CHANGES = 40
# Isotropic displacement parameters above this (Å²) usually mean something else is wrong.
LARGE_BEQ = 4.0


def _line_key(line: str) -> str:
    """A line with its numbers masked, for lining up an .inp with its .out."""
    line = _LIMIT_RE.sub("", line.split("'")[0])
    return "".join(_OUT_NUMBER_RE.sub("#", line).split())


def _plain(line: str) -> str:
    """A .out line without esds or refinement marks, and with tidy whitespace."""
    line = re.sub(r"`(_[-+0-9.eE]+)?", "", _LIMIT_RE.sub("", line))
    return " ".join(line.split())


def _crystal_family(space_group: str) -> str | None:
    """'cubic', 'hexagonal-like' or None, from a space group number or symbol."""
    try:
        number = int(space_group)
    except ValueError:
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            number = SpaceGroup(space_group.replace("_", "")).int_number
        except Exception:
            return None
    if 195 <= number <= 230:
        return "cubic"
    if 75 <= number <= 194:
        return "a=b"  # tetragonal, trigonal and hexagonal all have a = b
    return None


def refinement_report(inp_text: str, out_text: str) -> tuple[list[str], list[str]]:
    """(changes, warnings) between the .inp that was run and the .out TOPAS wrote.

    The .out is the .inp with refined values filled in, so lining the two up line by
    line shows what moved; the warnings are the checks a crystallographer makes before
    looking at the plot.
    """
    import difflib

    inp_lines, out_lines = inp_text.splitlines(), out_text.splitlines()
    matcher = difflib.SequenceMatcher(
        a=[_line_key(line) for line in inp_lines],
        b=[_line_key(line) for line in out_lines],
        autojunk=False,
    )
    pairs = [
        (inp_lines[block.a + k], out_lines[block.b + k])
        for block in matcher.get_matching_blocks()
        for k in range(block.size)
    ]

    changes: list[str] = []
    warnings: list[str] = []
    family = None
    cell: dict[str, float] = {}
    jumps: list[str] = []

    def check_cell():
        if jumps:
            warnings.append(
                f"The cell moved a long way ({', '.join(jumps)}). That is fine if the "
                "starting cell was simply off and the peaks now line up in the plot; "
                "otherwise it usually means a wrong peak assignment, a zero or "
                "displacement error absorbed into the cell, or peaks too broad to fit."
            )
        if (
            family == "cubic"
            and len(cell) == 3
            and max(cell.values()) - min(cell.values()) > 1e-4
        ):
            warnings.append(
                f"The space group is cubic but a, b, c refined to different values "
                f"({cell['a']:.5f}, {cell['b']:.5f}, {cell['c']:.5f}): give all three the "
                "same parameter name, or use Cubic(@ value)."
            )
        if (
            family == "a=b"
            and {"a", "b"} <= cell.keys()
            and abs(cell["a"] - cell["b"]) > 1e-4
        ):
            warnings.append(
                f"The space group requires a = b, but they refined to {cell['a']:.5f} and "
                f"{cell['b']:.5f}: give a and b the same parameter name."
            )

    for before, after in pairs:
        if before.strip().startswith("str") and before.split()[:1] == ["str"]:
            check_cell()
            family, cell, jumps = None, {}, []
        if match := _SPACE_GROUP_RE.match(before):
            family = _crystal_family(match.group(1))

        refined = _plain(after)
        if "`" in after and _plain(before) != refined:
            changes.append(f"{_plain(before)}  ->  {refined}")

        for limit in _LIMIT_RE.finditer(after):
            warnings.append(
                f"TOPAS flagged this as refined to its {limit.group(1).lower()}imum limit "
                f"({limit.group(2)}): `{refined}`. Check whether the parameter is needed at "
                "all, and its starting value."
            )

        if (start := _LATTICE_RE.match(before)) and (end := _LATTICE_RE.match(after)):
            axis, a0, a1 = start.group(1), float(start.group(2)), float(end.group(2))
            cell[axis] = a1
            if a0 and abs(a1 - a0) / a0 > LATTICE_JUMP:
                jumps.append(f"{axis} {a0} -> {a1}, {100 * (a1 - a0) / a0:+.1f}%")

        for beq in _BEQ_RE.finditer(after):
            if float(beq.group(1)) < 0:
                warnings.append(f"Negative beq is unphysical: `{refined}`.")
            elif float(beq.group(1)) > LARGE_BEQ:
                warnings.append(
                    f"beq of {float(beq.group(1)):.2f} Å² is far larger than an ordered "
                    f"solid has (about 0.3-2): `{refined}`. The usual causes are a missing "
                    "LP_Factor, unmodelled absorption, or a wrong occupancy."
                )

        for size in _SIZE_RE.finditer(after):
            if float(size.group(2)) < SMALL_CRYSTALLITE_NM:
                warnings.append(
                    f"{size.group(1)} crystallite size is {float(size.group(2)):.3g} nm, at "
                    "or near its 0.3 nm minimum: the peaks are modelled as enormously broad. "
                    "Sizes are in nm; start at 100-300."
                )
    check_cell()
    if re.search(r"^\s*xdd", inp_text, re.MULTILINE) and "LP_Factor" not in inp_text:
        warnings.append(
            "There is no LP_Factor: the Lorentz factor applies to every powder pattern, "
            "synchrotron included. Add LP_Factor(90) (or the example's form for lab data)."
        )

    if len(changes) > MAX_REPORTED_CHANGES:
        changes = changes[:MAX_REPORTED_CHANGES] + [
            f"... and {len(changes) - MAX_REPORTED_CHANGES} more; read the .out for the rest."
        ]
    return changes, list(dict.fromkeys(warnings))


def _time_suffix() -> str:
    """The deadline, for appending to a failure the agent has to act on."""
    if note := wrap_up_note():
        return f"\n\n{note}"
    remaining = minutes_remaining()
    return f"\n\n({remaining:.0f} minutes remain.)" if remaining is not None else ""


def _rwp_from_output(path: Optional[str]) -> Optional[float]:
    """Rwp (%) with 1/yobs weights from an `Out_X_Yobs_Ycalc` file, or None."""
    if not path:
        return None
    try:
        import numpy as np

        data = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    except Exception:
        return None
    yobs, ycalc = data[:, 1], data[:, 2]
    if len(yobs) == 0 or np.any(yobs <= 0):
        return None
    return round(
        float(100 * np.sqrt(np.sum((yobs - ycalc) ** 2 / yobs) / np.sum(yobs))), 3
    )


def run_topas_refinement(inp_path: str, timeout_s: int = 60) -> RunRefinementResult:
    """Run a TOPAS refinement with a locally-installed TOPAS executable."""
    stdout = stderr = ""
    returncode = None
    try:
        result = subprocess.run(
            [os.getenv("GUILLEMOT_TOPAS_EXE", DEFAULT_TOPAS_EXE), inp_path],
            timeout=timeout_s,
            text=True,
            capture_output=True,
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        status = "success" if returncode == 0 else "failure"
    except subprocess.TimeoutExpired as timeout:
        stdout = timeout.stdout or ""
        stderr = timeout.stderr or ""
        status = "timeout"

    outputs = _collect_refinement_outputs(inp_path)

    # A zero exit code is not proof of success: TOPAS 6 exits 0 after "Abnormal program
    # termination". The declared refinement result file is the real evidence.
    if status == "success" and outputs["refinement_result_path"] is None:
        status = "failure"

    if status == "failure":
        excerpt = _inp_excerpt(
            _read_text(inp_path), f"{stdout}\n{stderr}", os.path.basename(inp_path)
        )
        raise ModelRetry(
            f"TOPAS did not produce a refinement result (exit {returncode}). "
            f"Its output was:\n{stdout}\n{stderr}{excerpt}\n\n"
            "Fix the .inp and try again." + _time_suffix()
        )

    # todo: add tail of log file located at \Science\Topas-7\topas.log

    return RunRefinementResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        logs_tail=None,
        **outputs,
    )


# --------------------------------------------------------------------------------------
# Remote (SSH) execution
# --------------------------------------------------------------------------------------


class TopasSSHConfig(BaseModel):
    """Connection settings for the remote Windows machine that runs TOPAS.

    Configured through the environment:

    * ``GUILLEMOT_TOPAS_SSH_HOST`` - ``user@hostname`` (or a ``~/.ssh/config`` alias). Required.
    * ``GUILLEMOT_TOPAS_EXE`` - path to ``tc.exe`` on the remote machine.
    * ``GUILLEMOT_TOPAS_REMOTE_DIR`` - the workspace: the *only* directory on the
      remote machine guillemot writes to. Each run gets its own subdirectory of it,
      and TOPAS is confined to that subdirectory.
    * ``GUILLEMOT_TOPAS_SSH_PORT`` - optional non-standard SSH port.
    * ``GUILLEMOT_TOPAS_SSH_PASSWORD`` - optional password, for a machine that does not
      accept our key. It is handed to ``sshpass`` through its environment, never on a
      command line, so it does not show up in ``ps`` or in error messages. Key
      authentication is preferable wherever it is possible.
    """

    host: str
    topas_exe: str = DEFAULT_TOPAS_EXE
    remote_base_dir: str = DEFAULT_REMOTE_BASE_DIR
    port: Optional[int] = None
    password: Optional[SecretStr] = None

    @field_validator("host", "topas_exe", "remote_base_dir")
    @classmethod
    def _no_control_characters(cls, value: str, info) -> str:
        """Catch Windows paths mangled by .env escape processing.

        `python-dotenv` expands backslash escapes inside double-quoted values, so
        ``"C:\\Users\\x\\Topas_6\\tc.exe"`` arrives with a tab where ``\\t`` was. The
        resulting cmd error is baffling, so name the cause here instead.
        """
        if any(character in value for character in "\t\n\r\v\f\b"):
            raise ValueError(
                f"{info.field_name} contains a control character ({value!r}); a "
                "backslash in the path was probably read as an escape sequence. "
                "Quote Windows paths in .env with single quotes, e.g. "
                r"GUILLEMOT_TOPAS_EXE='C:\Science\Topas-7\tc.exe'."
            )
        return value

    @classmethod
    def from_env(cls) -> "TopasSSHConfig":
        host = os.getenv("GUILLEMOT_TOPAS_SSH_HOST")
        if not host:
            raise ModelRetry(
                "No remote TOPAS machine is configured. Set GUILLEMOT_TOPAS_SSH_HOST "
                "(e.g. 'user@topas-pc') in the environment or .env file, and optionally "
                "GUILLEMOT_TOPAS_EXE and GUILLEMOT_TOPAS_REMOTE_DIR."
            )
        port = os.getenv("GUILLEMOT_TOPAS_SSH_PORT")
        return cls(
            host=host,
            topas_exe=os.getenv("GUILLEMOT_TOPAS_EXE", DEFAULT_TOPAS_EXE),
            remote_base_dir=os.getenv(
                "GUILLEMOT_TOPAS_REMOTE_DIR", DEFAULT_REMOTE_BASE_DIR
            ),
            port=int(port) if port else None,
            password=os.getenv("GUILLEMOT_TOPAS_SSH_PASSWORD") or None,
        )

    def run_dir_for(self, run_id: str) -> str:
        """The remote directory for one refinement: a subdirectory of the workspace."""
        return f"{_win_path(self.remote_base_dir)}\\{_sanitise_run_id(run_id)}"

    def _auth_prefix(self, program: str) -> list[str]:
        """How ssh/scp start: non-interactive with a key, or through sshpass.

        BatchMode forbids password prompts outright, so with a password it is swapped
        for a single prompt for sshpass to answer: a wrong password then fails at once
        rather than leaving the run waiting at a prompt nobody will see.
        """
        if self.password is None:
            return [program, "-o", "BatchMode=yes"]
        if shutil.which("sshpass") is None:
            raise RuntimeError(
                "GUILLEMOT_TOPAS_SSH_PASSWORD is set but `sshpass` is not installed; "
                "install it, or set up key authentication to the TOPAS machine."
            )
        return ["sshpass", "-e", program, "-o", "NumberOfPasswordPrompts=1"]

    @property
    def subprocess_env(self) -> Optional[dict]:
        """The environment for ssh/scp: ours, plus the password for sshpass if any."""
        if self.password is None:
            return None
        return {**os.environ, "SSHPASS": self.password.get_secret_value()}

    @property
    def ssh_cmd(self) -> list[str]:
        cmd = self._auth_prefix("ssh")
        if self.port:
            cmd += ["-p", str(self.port)]
        return cmd + [self.host]

    @property
    def scp_cmd(self) -> list[str]:
        cmd = self._auth_prefix("scp") + ["-r"]
        if self.port:
            # scp spells the port flag with a capital P
            cmd += ["-P", str(self.port)]
        return cmd


def _win_path(path: str) -> str:
    """Normalise a Windows path to backslashes, stripping any trailing separator."""
    return path.replace("/", "\\").rstrip("\\")


def _scp_path(path: str) -> str:
    """Windows paths travel through scp more reliably with forward slashes."""
    return path.replace("\\", "/")


def _run_streaming(
    cmd: list[str], timeout_s: int, echo: bool = True, env: Optional[dict] = None
) -> tuple[int | None, str, str]:
    """Run ``cmd``, echoing output as it arrives, and return (returncode, stdout, stderr).

    A returncode of ``None`` means the command hit ``timeout_s`` and was killed.
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    streams = {proc.stdout: [], proc.stderr: []}
    deadline = time.monotonic() + timeout_s
    timed_out = False

    open_streams = list(streams)
    while open_streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            proc.kill()
            break
        ready, _, _ = select.select(open_streams, [], [], min(remaining, 1.0))
        for stream in ready:
            line = stream.readline()
            if not line:
                open_streams.remove(stream)
                continue
            streams[stream].append(line)
            if echo:
                print(line.rstrip("\n"), flush=True)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Drain anything buffered after a kill so no output is lost.
    for stream in streams:
        try:
            streams[stream].append(stream.read() or "")
        except ValueError:  # stream already closed
            pass
        finally:
            stream.close()

    stdout = "".join(streams[proc.stdout])
    stderr = "".join(streams[proc.stderr])
    return (None if timed_out else proc.returncode), stdout, stderr


def _ssh(
    config: TopasSSHConfig, remote_command: str, timeout_s: int, echo: bool = False
) -> tuple[int | None, str, str]:
    """Run ``remote_command`` on the remote machine via ``cmd.exe``.

    Wrapping in ``cmd /c`` means the call behaves the same whether the remote
    OpenSSH default shell is cmd.exe or PowerShell.
    """
    return _run_streaming(
        config.ssh_cmd + [f'cmd /c "{remote_command}"'],
        timeout_s=timeout_s,
        echo=echo,
        env=config.subprocess_env,
    )


class TopasProcessStatus(BaseModel):
    running: bool
    processes: list[str] = Field(default_factory=list)
    host: str
    detail: str


def check_remote_topas_running(timeout_s: int = 30) -> TopasProcessStatus:
    """Check whether a TOPAS process is already running on the remote machine.

    Use this before starting a remote refinement: TOPAS refinements on a shared
    machine must not overlap, since a second instance will fight the first one for
    the licence and the working directory.
    """
    config = TopasSSHConfig.from_env()

    # tasklist can only apply one IMAGENAME filter at a time, so query each in turn.
    # List every process once and match names here, rather than running a filtered
    # `tasklist` per name. tasklist takes only one IMAGENAME filter at a time, and
    # chaining the calls with `&` proved unreliable on Windows: a query for a process
    # the account cannot enumerate returns nothing, exits 1, and takes the output of
    # the other queries in the chain with it.
    returncode, stdout, stderr = _ssh(config, "tasklist /NH", timeout_s=timeout_s)
    if returncode is None:
        raise ModelRetry(
            f"Timed out after {timeout_s}s checking for running TOPAS processes on {config.host}."
        )
    elif returncode == 255:
        raise RuntimeError(
            f"Could not connect to {config.host} to check for running TOPAS processes. "
            f"stdout: {stdout} stderr: {stderr}"
        )
    if returncode != 0:
        raise ModelRetry(
            f"Could not query processes on {config.host} (exit {returncode}). "
            f"stdout: {stdout} stderr: {stderr}"
        )

    wanted = {name.lower() for name in TOPAS_PROCESS_NAMES}
    processes = [
        line.strip()
        for line in stdout.splitlines()
        if line.split() and line.split()[0].lower() in wanted
    ]

    listed = len([line for line in stdout.splitlines() if line.strip()])
    return TopasProcessStatus(
        running=bool(processes),
        processes=processes,
        host=config.host,
        detail=(
            f"{len(processes)} TOPAS process(es) among {listed} listed on {config.host}"
        ),
    )


def _sanitise_run_id(name: str) -> str:
    """Reduce an agent-chosen name to a single safe directory component."""
    sanitised = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "run"
    return sanitised[:64]


def _find_local_file(
    name: str, search_dirs: list[pathlib.Path]
) -> Optional[pathlib.Path]:
    """Locate a file referenced by an .inp file, by basename, near the run directory."""
    candidate = pathlib.Path(name)
    if candidate.is_file():
        return candidate

    basename = candidate.name
    for directory in search_dirs:
        direct = directory / basename
        if direct.is_file():
            return direct

    # Fall back to a shallow search of the downloaded sample data.
    data_dir = pathlib.Path("data")
    if data_dir.is_dir():
        for match in data_dir.glob(f"*/{basename}"):
            if match.is_file():
                return match
    return None


def _stage_inputs(
    inp_path: pathlib.Path, staging_dir: pathlib.Path
) -> tuple[pathlib.Path, list[pathlib.Path]]:
    """Prepare an .inp and its data files for upload into a flat remote run directory.

    Data files are referenced either unquoted after an ``xdd``-style directive or as a
    quoted name; only references that resolve to a file that exists locally are treated
    as inputs, which conveniently excludes the names used by the ``Out_*`` output macros.
    Since everything is uploaded flat into the run directory, any reference carrying a
    directory prefix is rewritten to its bare filename in the staged copy of the .inp.

    Returns the staged .inp path and the local data files to upload with it.
    """
    text = inp_path.read_text(encoding="utf-8", errors="replace")
    search_dirs = [inp_path.parent, pathlib.Path.cwd()]

    references = re.findall(
        r'^\s*xdd\w*\s+"?([^"\n]+?)"?\s*$', text, flags=re.MULTILINE
    )
    references += re.findall(r'"([^"\n]+)"', text)

    found: dict[str, pathlib.Path] = {}
    for reference in references:
        reference = reference.strip()
        local = _find_local_file(reference, search_dirs) if reference else None
        if local is None or local.resolve() == inp_path.resolve():
            continue
        found[local.name] = local
        if reference != local.name:
            # Point the .inp at the flat copy that will sit next to it remotely.
            text = re.sub(
                rf"(?<![\w./\\]){re.escape(reference)}(?![\w.])", local.name, text
            )

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_inp = staging_dir / inp_path.name
    staged_inp.write_text(text, encoding="utf-8", newline="\n")
    return staged_inp, list(found.values())


class RemoteRefinementResult(RunRefinementResult):
    """A refinement result plus the details of where it ran."""

    host: str
    run_id: str
    remote_run_dir: str
    local_run_dir: str
    files_uploaded: list[str] = Field(default_factory=list)
    files_returned: list[str] = Field(default_factory=list)


def _run_topas_refinement_remote_unlocked(
    inp_path: str,
    timeout_s: int = 600,
    extra_files: Optional[list[str]] = None,
) -> RemoteRefinementResult:
    """Run a TOPAS refinement on the configured remote Windows machine over SSH.

    This checks that TOPAS is not already running remotely, copies the .inp file and
    every data file it references into a fresh timestamped run directory on the remote
    machine, runs TOPAS there while streaming its output, then syncs the whole run
    directory back locally and plots the result.

    On success the result gives the Rwp, every value the refinement changed
    (`refined_values`, as `start  ->  refined`), and `warnings` about refined values that
    look wrong — deal with any warnings before anything else.

    The plot of the fit is saved to `plot_path` in the returned result, but is not shown
    to you here. To look at it, call `plot_refinement_results` with the returned
    `refinement_result_path` and `plot_path`.

    All remote writes are confined to a per-run subdirectory of the configured remote
    workspace, so every filename in the .inp file — the data files it reads and the
    names given to the ``Out_*`` macros alike — must be a bare filename with no drive
    letter, leading slash or directory component.

    Args:
        inp_path: local path of the .inp file, as returned by `save_topas_inp`.
        timeout_s: how long to let the remote refinement run before killing it.
        extra_files: any additional local files to upload alongside the .inp.
    """
    config = TopasSSHConfig.from_env()

    local_inp = pathlib.Path(inp_path)
    if not local_inp.is_file():
        raise ModelRetry(
            f"No .inp file at {inp_path}. Write one with `save_topas_inp` first."
        )
    # Checked again here: the name is interpolated into the remote command string, and
    # this tool can be handed a path that did not come from `save_topas_inp`.
    _check_safe_filename(local_inp.name, "input filename")

    # Tagged with the session, so two agents running `riet_01` in the same second (as
    # concurrent benchmark entries can) never share a remote directory.
    session_tag = hashlib.sha1(str(session_dir().resolve()).encode()).hexdigest()[:6]
    run_id = _sanitise_run_id(
        f"{local_inp.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_tag}"
    )
    remote_run_dir = config.run_dir_for(run_id)

    # Validate and stage everything locally first, so a bad .inp is rejected before the
    # remote machine is touched at all.
    staging_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"guillemot_{run_id}_"))
    try:
        staged_inp, data_files = _stage_inputs(local_inp, staging_dir)
        uploads = [staged_inp] + data_files
        for extra in extra_files or []:
            extra_path = pathlib.Path(extra)
            if not extra_path.is_file():
                raise ModelRetry(
                    f"Requested extra file {extra} does not exist locally."
                )
            uploads.append(extra_path)

        # (a) never start a second refinement on top of a running one. Refinements from
        # this machine are already queued by the lock in `run_topas_refinement_remote`;
        # this waits out anyone else's, and only gives up if one runs for a long time.
        waited_until = time.monotonic() + TOPAS_BUSY_WAIT_S
        while (status := check_remote_topas_running()).running:
            if time.monotonic() > waited_until:
                raise ModelRetry(
                    f"TOPAS is already running on {config.host} ({status.processes}) and "
                    f"has been for over {TOPAS_BUSY_WAIT_S // 60} minutes. Try again later."
                )
            time.sleep(15)

        # (b) create a fresh, run-specific directory and upload the inputs into it
        returncode, stdout, stderr = _ssh(
            config,
            f'if not exist "{remote_run_dir}" mkdir "{remote_run_dir}"',
            timeout_s=60,
        )
        if returncode != 0:
            raise ModelRetry(
                f"Could not create remote run directory {remote_run_dir} on {config.host} "
                f"(exit {returncode}). stdout: {stdout} stderr: {stderr}"
            )

        upload_cmd = (
            config.scp_cmd
            + [str(p) for p in uploads]
            + [f"{config.host}:{_scp_path(remote_run_dir)}/"]
        )
        returncode, stdout, stderr = _run_streaming(
            upload_cmd, timeout_s=300, echo=False, env=config.subprocess_env
        )
        if returncode != 0:
            raise ModelRetry(
                f"Failed to copy input files to {config.host} with "
                f"`{shlex.join(upload_cmd)}` (exit {returncode}). stdout: {stdout} stderr: {stderr}"
            )
        uploaded_names = [p.name for p in uploads]
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    # (c) run TOPAS in that directory, streaming its output to the terminal
    print(f"▶ running TOPAS on {config.host} in {remote_run_dir}", flush=True)
    run_returncode, run_stdout, run_stderr = _ssh(
        config,
        # The .inp is named by its full path: TOPAS 8 resolves a bare filename against
        # its own install directory, not the working directory (TOPAS 6 used the
        # latter). Its data and output files are relative to the .inp, so they still
        # land in the run directory.
        f'cd /d "{remote_run_dir}" && "{config.topas_exe}" "{remote_run_dir}\\{local_inp.name}"',
        timeout_s=timeout_s,
        echo=True,
    )

    if run_returncode is None:
        run_status = "timeout"
    elif run_returncode == 0:
        run_status = "success"
    else:
        run_status = "failure"

    # (d) sync everything back, even on failure — the logs are the useful part
    local_run_dir = session_dir() / run_id
    local_run_dir.parent.mkdir(parents=True, exist_ok=True)
    download_cmd = config.scp_cmd + [
        f"{config.host}:{_scp_path(remote_run_dir)}",
        str(local_run_dir.parent),
    ]
    dl_returncode, dl_stdout, dl_stderr = _run_streaming(
        download_cmd, timeout_s=300, echo=False, env=config.subprocess_env
    )
    if dl_returncode != 0:
        raise ModelRetry(
            f"TOPAS finished with status '{run_status}' but the results could not be copied "
            f"back from {config.host} (exit {dl_returncode}). stdout: {dl_stdout} stderr: {dl_stderr}"
        )

    files_returned = (
        sorted(p.name for p in local_run_dir.iterdir())
        if local_run_dir.is_dir()
        else []
    )

    outputs = _collect_refinement_outputs(str(local_run_dir / local_inp.name))

    # TOPAS 6 exits 0 even when it rejects the input ("Abnormal program termination"),
    # so a zero exit code alone does not mean the refinement ran. The refinement result
    # file that `save_topas_inp` requires the .inp to declare is the real evidence.
    if run_status == "success" and outputs["refinement_result_path"] is None:
        run_status = "failure"

    if run_status == "failure":
        # Quote from the copy that came back, which is byte-for-byte what TOPAS parsed
        # and so is numbered the way its error message is.
        excerpt = _inp_excerpt(
            _read_text(local_run_dir / local_inp.name) or _read_text(local_inp),
            f"{run_stdout}\n{run_stderr}",
            local_inp.name,
        )
        raise ModelRetry(
            f"TOPAS did not produce a refinement result on {config.host} "
            f"(exit {run_returncode}). Its output was:\n{run_stdout}\n{run_stderr}"
            f"{excerpt}\n\n"
            f"Whatever was written is in {local_run_dir}. Fix the .inp and try again."
            + _time_suffix()
        )

    return RemoteRefinementResult(
        status=run_status,
        stdout=run_stdout,
        stderr=run_stderr,
        logs_tail=None,
        host=config.host,
        run_id=run_id,
        remote_run_dir=remote_run_dir,
        local_run_dir=str(local_run_dir),
        files_uploaded=uploaded_names,
        files_returned=files_returned,
        **outputs,
    )


# How long to wait for someone else's TOPAS run on the shared machine to finish.
TOPAS_BUSY_WAIT_S = 600


def run_topas_refinement_remote(
    inp_path: str,
    timeout_s: int = 600,
    extra_files: Optional[list[str]] = None,
) -> RemoteRefinementResult:
    # Several agents on this machine (concurrent benchmark entries) take turns: TOPAS
    # runs take seconds, so queueing costs little, where refusing would cost a turn.
    config = TopasSSHConfig.from_env()
    lock_path = pathlib.Path(RUN_DIR) / f".topas-{_sanitise_run_id(config.host)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _run_topas_refinement_remote_unlocked(
                inp_path, timeout_s, extra_files
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


run_topas_refinement_remote.__doc__ = _run_topas_refinement_remote_unlocked.__doc__
