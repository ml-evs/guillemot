"""Tests for the SSH-based TOPAS runner.

These exercise the real code paths against stub `ssh`/`scp` executables that emulate
a Windows remote: `tasklist` process queries, `cmd /c` quoting, a flat run directory
that TOPAS writes its outputs into, and file transfer in both directions.
"""

import os
import pathlib
import stat
import textwrap

import pytest
from pydantic_ai.exceptions import ModelRetry

from guillemot.tools.topas import (
    save_topas_inp,
    TopasSSHConfig,
    _sanitise_run_id,
    _stage_inputs,
    check_remote_topas_running,
    run_topas_refinement_remote,
)

FAKE_SSH = '''#!/usr/bin/env python3
"""Stub for the Windows OpenSSH server reached over `ssh`."""
import os
import pathlib
import re
import sys
import time

REMOTE_ROOT = os.environ["REMOTE_WS"]


def to_local(remote_path):
    """Map a remote Windows path onto the fake root directory."""
    tail = remote_path.replace("/", "\\\\")
    if not tail.lower().startswith(REMOTE_ROOT.lower()):
        raise SystemExit(f"stub-ssh: write outside workspace: {remote_path}")
    tail = tail[len(REMOTE_ROOT):].strip("\\\\")
    return pathlib.Path(os.environ["FAKE_ROOT"]).joinpath(*filter(None, tail.split("\\\\")))


command = sys.argv[-1]
match = re.fullmatch(r'cmd /c "(.*)"', command, flags=re.DOTALL)
if not match:
    sys.exit("stub-ssh: command was not wrapped in cmd /c")
inner = match.group(1)

if inner.startswith("tasklist"):
    print("System Idle Process              0 Services     0          8 K")
    print("explorer.exe                  5300 Console      1    290,796 K")
    # a name that merely contains a TOPAS process name must not count as a match
    print("nottc.exe                     5301 Console      1      1,024 K")
    if os.environ.get("FAKE_TOPAS_RUNNING"):
        print("tc.exe                        4242 Console      1     70,000 K")
    if os.environ.get("FAKE_TOPAS_GUI_RUNNING"):
        print("ta61.exe                      4243 Console      1     90,000 K")
    sys.exit(0)

if inner.startswith("if not exist"):
    target = re.findall(r'"([^"]+)"', inner)[-1]
    to_local(target).mkdir(parents=True, exist_ok=True)
    sys.exit(0)

if inner.startswith("cd /d"):
    run_dir, exe, inp = re.findall(r'"([^"]+)"', inner)
    assert exe == os.environ["EXPECTED_EXE"], f"unexpected exe {exe}"
    assert "\\\\" not in inp and "/" not in inp, f"inp should be a bare name: {inp}"
    local_dir = to_local(run_dir)
    os.chdir(local_dir)

    time.sleep(float(os.environ.get("FAKE_TOPAS_SLEEP", "0")))
    print("TOPAS stub: refining", inp)
    print("Rwp  4.21")

    exit_code = int(os.environ.get("FAKE_TOPAS_EXIT", "0"))
    if exit_code:
        print("Error in input file", file=sys.stderr)

    if os.environ.get("FAKE_TOPAS_PARSE_ERROR"):
        # A real TOPAS 6 parse failure: a line number, the offending token, exit 0.
        print("*** Error loading sstring_in")
        print("    at LINE", os.environ["FAKE_TOPAS_PARSE_ERROR"])
        print("*** Error at:", os.environ.get("FAKE_TOPAS_PARSE_TOKEN", "@"))
        print("Abnormal program termination.")
        sys.exit(0)

    if os.environ.get("FAKE_TOPAS_NO_OUTPUT"):
        # TOPAS 6 rejects the input but still exits 0
        print(" cell_volume should not have an equation at this stage")
        print("Abnormal program termination.")
        sys.exit(0)

    stem = inp[: -len(".inp")]
    inp_text = pathlib.Path(inp).read_text()
    pathlib.Path(f"{stem}.out").write_text(inp_text)

    pattern_name = re.search(r"^\\s*xdd\\w*\\s+(\\S+)", inp_text, flags=re.M).group(1)
    rows = []
    for line in pathlib.Path(pattern_name).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append(f"{parts[0]} {parts[1]} {float(parts[1]) * 0.98}")
    pathlib.Path(f"{stem}_output.txt").write_text("\\n".join(rows) + "\\n")
    sys.exit(exit_code)

sys.exit(f"stub-ssh: unhandled command {inner!r}")
'''

FAKE_SCP = '''#!/usr/bin/env python3
"""Stub for `scp`, mapping remote Windows paths onto the fake root."""
import os
import pathlib
import shutil
import sys

REMOTE_ROOT = os.environ["REMOTE_WS"].replace("\\\\", "/")

paths = []
args = sys.argv[1:]
while args:
    arg = args.pop(0)
    if arg in ("-o", "-P"):
        args.pop(0)
        continue
    if arg.startswith("-"):
        continue
    if ":" in arg and not arg.startswith("/"):
        remote = arg.split(":", 1)[1]
        assert remote.lower().startswith(REMOTE_ROOT.lower()), f"scp outside workspace: {remote}"
        tail = remote[len(REMOTE_ROOT):].strip("/")
        arg = str(pathlib.Path(os.environ["FAKE_ROOT"]).joinpath(*filter(None, tail.split("/"))))
    paths.append(arg)

*sources, destination = paths
for source in sources:
    source_path = pathlib.Path(source)
    if source_path.is_dir():
        shutil.copytree(source_path, pathlib.Path(destination) / source_path.name)
    else:
        shutil.copy(source_path, destination)
'''

REMOTE_WORKSPACE = r"C:\remote_ws"
TOPAS_EXE = r"C:\Topas_6\tc.exe"

INP_TEMPLATE = textwrap.dedent(
    """\
    xdd {pattern}
        start_X 5
        finish_X 60
    str
        phase_name "NaCoO2"
        Out_CIF_STR("{stem}_NaCoO2.cif")
    Out_X_Yobs_Ycalc("{stem}_output.txt")
    """
)


def _write_executable(path: pathlib.Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A working directory wired up to a stub remote machine."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "ssh", FAKE_SSH)
    _write_executable(bin_dir / "scp", FAKE_SCP)

    fake_root = tmp_path / "remote"
    fake_root.mkdir()

    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_ROOT", str(fake_root))
    monkeypatch.setenv("REMOTE_WS", REMOTE_WORKSPACE)
    monkeypatch.setenv("EXPECTED_EXE", TOPAS_EXE)
    monkeypatch.setenv("GUILLEMOT_TOPAS_SSH_HOST", "group_user@topas-stub")
    monkeypatch.setenv("GUILLEMOT_TOPAS_EXE", TOPAS_EXE)
    monkeypatch.setenv("GUILLEMOT_TOPAS_REMOTE_DIR", REMOTE_WORKSPACE)
    monkeypatch.delenv("GUILLEMOT_TOPAS_SSH_PORT", raising=False)

    # every test starts from a run_dir holding a .inp and the pattern it refers to
    _make_run()

    return fake_root


def _make_run(
    workdir: pathlib.Path | None = None, stem: str = "KD1", pattern: str | None = None
):
    """Write a run_dir containing a .inp and the pattern it refers to."""
    run_dir = (workdir or pathlib.Path.cwd()) / "run_dir"
    run_dir.mkdir(exist_ok=True)
    (run_dir / f"{stem}.xy").write_text(
        "\n".join(f"{5 + 0.01 * i:.5f} {1000 + i}.000" for i in range(200)) + "\n"
    )
    inp = run_dir / f"{stem}.inp"
    inp.write_text(
        INP_TEMPLATE.format(stem=stem, pattern=pattern or f"run_dir/{stem}.xy")
    )
    return inp


def test_run_remote_success(remote):
    result = run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    assert result.status == "success"
    assert result.host == "group_user@topas-stub"
    assert result.remote_run_dir.startswith(REMOTE_WORKSPACE + "\\")
    assert "TOPAS stub: refining KD1.inp" in result.stdout

    # inputs went up, the whole run directory came back
    assert sorted(result.files_uploaded) == ["KD1.inp", "KD1.xy"]
    assert set(result.files_returned) >= {
        "KD1.inp",
        "KD1.out",
        "KD1_output.txt",
        "KD1.xy",
    }

    local_run_dir = pathlib.Path(result.local_run_dir)
    assert local_run_dir.parent.name == "run_dir"
    assert (local_run_dir / "KD1_plot.png").is_file()
    # The plot is on disk and named in the result, but the image itself is not sent to
    # the model here — it only travels with a `plot_refinement_results` call.
    assert result.plot_path == str(local_run_dir / "KD1_plot.png")
    assert result.outfile_contents is not None
    assert result.refinement_result_path == str(local_run_dir / "KD1_output.txt")

    # the run directory is a single component below the workspace
    assert (remote / result.run_id).is_dir()
    assert result.remote_run_dir == f"{REMOTE_WORKSPACE}\\{result.run_id}"


def test_results_land_in_the_session_directory(remote):
    """With a conversation running, its results belong in that conversation's directory."""
    from guillemot.session import start_session

    session = start_session()
    result = run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    local_run_dir = pathlib.Path(result.local_run_dir)
    assert local_run_dir.parent == session.directory
    assert (local_run_dir / "KD1_output.txt").is_file()


def test_pattern_path_is_flattened_for_the_remote(remote):
    """`xdd run_dir/KD1.xy` must become `xdd KD1.xy` next to the uploaded pattern."""
    result = run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    uploaded_inp = (remote / result.run_id / "KD1.inp").read_text()
    assert "xdd KD1.xy" in uploaded_inp
    assert "run_dir/KD1.xy" not in uploaded_inp
    # output macro names are untouched
    assert 'Out_X_Yobs_Ycalc("KD1_output.txt")' in uploaded_inp
    # ...and the local .inp is not modified
    assert "xdd run_dir/KD1.xy" in pathlib.Path("run_dir/KD1.inp").read_text()


def test_data_file_found_outside_run_dir(remote):
    """A pattern downloaded into data/<sample>/ is located and uploaded."""
    data_dir = pathlib.Path("data/SFRORL")
    data_dir.mkdir(parents=True)
    inp = _make_run(pattern="SFRORL.xy")
    (pathlib.Path("run_dir") / "KD1.xy").rename(data_dir / "SFRORL.xy")

    result = run_topas_refinement_remote(str(inp), timeout_s=60)

    assert sorted(result.files_uploaded) == ["KD1.inp", "SFRORL.xy"]
    assert (remote / result.run_id / "SFRORL.xy").is_file()


def test_refuses_to_start_when_topas_is_busy(remote, monkeypatch):
    monkeypatch.setenv("FAKE_TOPAS_RUNNING", "1")

    status = check_remote_topas_running()
    assert status.running
    assert any("tc.exe" in process for process in status.processes)

    with pytest.raises(ModelRetry, match="already running"):
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    # nothing was staged on the remote machine
    assert list(remote.iterdir()) == []


def test_reports_idle_remote(remote):
    """Other processes, including names that merely contain 'tc.exe', are not TOPAS."""
    status = check_remote_topas_running()
    assert not status.running
    assert status.processes == []


def test_detects_the_gui_as_well_as_the_console_binary(remote, monkeypatch):
    """TOPAS 6's GUI is ta.exe / ta61.exe, not topas.exe."""
    monkeypatch.setenv("FAKE_TOPAS_GUI_RUNNING", "1")

    status = check_remote_topas_running()
    assert status.running
    assert any("ta61.exe" in process for process in status.processes)

    with pytest.raises(ModelRetry, match="already running"):
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)


def test_failure_syncs_results_back_before_raising(remote, monkeypatch):
    monkeypatch.setenv("FAKE_TOPAS_EXIT", "1")

    with pytest.raises(ModelRetry) as excinfo:
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    message = str(excinfo.value)
    assert "did not produce a refinement result" in message
    assert "Error in input file" in message  # stderr is handed back to the model

    synced = sorted(p.name for p in pathlib.Path("run_dir").glob("KD1_*/*"))
    assert "KD1.out" in synced


def test_zero_exit_without_results_is_a_failure(remote, monkeypatch):
    """TOPAS 6 exits 0 after 'Abnormal program termination' — don't call that success."""
    monkeypatch.setenv("FAKE_TOPAS_NO_OUTPUT", "1")

    with pytest.raises(ModelRetry) as excinfo:
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    message = str(excinfo.value)
    assert "did not produce a refinement result" in message
    # the diagnostic TOPAS printed is what the model needs to fix the .inp
    assert "cell_volume should not have an equation" in message


def test_timeout_is_reported_not_raised(remote, monkeypatch):
    monkeypatch.setenv("FAKE_TOPAS_SLEEP", "5")

    result = run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=1)

    assert result.status == "timeout"


def test_ssh_and_scp_carry_the_configured_port():
    config = TopasSSHConfig(host="user@box", port=2222)
    assert config.ssh_cmd == ["ssh", "-o", "BatchMode=yes", "-p", "2222", "user@box"]
    assert config.scp_cmd == ["scp", "-o", "BatchMode=yes", "-r", "-P", "2222"]


def test_staging_leaves_a_bare_inp_when_nothing_to_rewrite(tmp_path):
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "KD1.xy").write_text("5.0 1000.0\n")
    inp = run_dir / "KD1.inp"
    inp.write_text(INP_TEMPLATE.format(stem="KD1", pattern="KD1.xy"))

    os.chdir(tmp_path)
    staged, data_files = _stage_inputs(inp, tmp_path / "stage")

    assert [p.name for p in data_files] == ["KD1.xy"]
    assert "xdd KD1.xy" in staged.read_text()


def test_parse_error_quotes_the_offending_lines(remote, monkeypatch):
    """A failed parse must show the agent the lines it wrote, not just a line number.

    The agent cannot read its own .inp back, so a bare "at LINE 6" leaves it counting
    from memory — which is how a previous session blamed the wrong line and burned a
    second remote run on the wrong fix.
    """
    monkeypatch.setenv("FAKE_TOPAS_PARSE_ERROR", "6")
    monkeypatch.setenv("FAKE_TOPAS_PARSE_TOKEN", "Out_CIF_STR")

    with pytest.raises(ModelRetry) as excinfo:
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    message = str(excinfo.value)
    assert "KD1.inp at line 6" in message
    assert "on the token `Out_CIF_STR`" in message
    # the reported line, quoted with its number and marked
    assert "> 6 |" in message
    assert 'Out_CIF_STR("KD1_NaCoO2.cif")' in message
    # and its neighbours, so an off-by-one is still visible
    assert "5 |" in message and "7 |" in message


def test_failure_without_a_line_number_quotes_nothing(remote, monkeypatch):
    """Not every failure is a syntax error; a misleading excerpt is worse than none."""
    monkeypatch.setenv("FAKE_TOPAS_NO_OUTPUT", "1")

    with pytest.raises(ModelRetry) as excinfo:
        run_topas_refinement_remote("run_dir/KD1.inp", timeout_s=60)

    assert "stopped parsing" not in str(excinfo.value)
