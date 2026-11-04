# This file was edited with the assistance of an AI model and requires human review from the contributor.
"""The benchmark harness, on synthetic opXRD entries: no dataset, model or TOPAS needed."""

import json

import numpy as np
import pytest

from guillemot.benchmark import runner, scoring, splits
from guillemot.benchmark.opxrd import (
    TIER_NONE,
    TIER_PARTIAL,
    TIER_STRUCTURE,
    parse_entry,
    reference_structure,
)

NACL_BASIS = [
    {"x": x, "y": y, "z": z, "occupancy": 1, "symbol": symbol}
    for symbol, origin in (("Na1+", 0.0), ("Cl1-", 0.5))
    for x, y, z in (
        (origin, 0.0, 0.0),
        (origin, 0.5, 0.5),
        ((origin + 0.5) % 1, 0.0, 0.5),
        ((origin + 0.5) % 1, 0.5, 0.0),
    )
]


def raw_entry(phases, primary="1.54439", secondary="1.54056"):
    """An entry as opXRD stores it, JSON-in-strings and swapped wavelengths included."""
    two_theta = np.linspace(10, 80, 701)
    label = {
        "phases": [json.dumps(p) for p in phases],
        "xray_info": {"primary_wavelength": primary, "secondary_wavelength": secondary},
        "is_simulated": False,
    }
    return {
        "two_theta_values": two_theta.tolist(),
        "intensities": (100 + 50 * np.exp(-((two_theta - 31.7) ** 2) / 0.05)).tolist(),
        "label": json.dumps(label),
        "metadata": json.dumps({"institution": "Test"}),
    }


NACL = {
    "lattice": "(5.64, 5.64, 5.64, 90.0, 90.0, 90.0)",
    "basis": [json.dumps(atom) for atom in NACL_BASIS],
    "spacegroup": None,
    "chemical_composition": None,
}


def test_labels_are_decoded_and_wavelengths_put_right():
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    assert entry.label_tier == TIER_STRUCTURE
    assert entry.formulas == ["NaCl"]
    assert entry.elements == ["Cl", "Na"]
    assert entry.kalpha1 == pytest.approx(1.54056)
    assert entry.kalpha2 == pytest.approx(1.54439)
    assert entry.group_key == "compound:NaCl"


def test_label_tiers():
    partial = {
        "lattice": None,
        "basis": None,
        "spacegroup": "62",
        "chemical_composition": "PbBr2",
    }
    assert parse_entry(raw_entry([partial]), "T/p").label_tier == TIER_PARTIAL
    assert parse_entry(raw_entry([]), "T/p").label_tier == TIER_NONE


def test_single_wavelength_has_no_kalpha2():
    entry = parse_entry(raw_entry([], primary="1.23984", secondary=None), "T/p")
    assert entry.kalpha1 == pytest.approx(1.23984)
    assert entry.kalpha2 is None


def test_reference_structure_recovers_symmetry_and_merges_duplicates():
    duplicated = dict(NACL, basis=NACL["basis"] + NACL["basis"][:1])
    entry = parse_entry(raw_entry([duplicated]), "T/p")
    structure = reference_structure(entry.phases[0])
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    assert SpacegroupAnalyzer(structure).get_space_group_number() == 225
    assert structure.composition.reduced_formula == "NaCl"
    assert all(site.species.num_atoms <= 1 + 1e-9 for site in structure)


def test_splits_are_deterministic_and_grouped():
    assert splits.split_for("compound:NaCl") == splits.split_for("compound:NaCl")
    counts = {s: 0 for s in splits.SPLITS}
    for n in range(20000):
        counts[splits.split_for(f"compound:X{n}")] += 1
    for name, fraction in splits.SPLIT_FRACTIONS.items():
        assert counts[name] / 20000 == pytest.approx(fraction, abs=0.01)


def test_lbnl_series_share_a_group():
    a = parse_entry(raw_entry([]), "LBNL/LBNL-A/pattern_group_7_0")
    b = parse_entry(raw_entry([]), "LBNL/LBNL-A/pattern_group_7_12")
    assert a.group_key == b.group_key


def test_perfect_fit_scores_zero_and_partial_range_is_invalid(tmp_path):
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    x, y = np.array(entry.two_theta), np.array(entry.intensities)

    perfect = tmp_path / "perfect_output.txt"
    np.savetxt(perfect, np.column_stack([x, y, y]))
    result = scoring.score(entry, perfect)
    assert result.rwp == pytest.approx(0.0, abs=1e-9)
    assert result.coverage == pytest.approx(1.0)
    assert result.valid

    window = (x > 25) & (x < 40)
    cropped = tmp_path / "cropped_output.txt"
    np.savetxt(cropped, np.column_stack([x[window], y[window], y[window]]))
    assert not scoring.score(entry, cropped).valid


def test_rwp_matches_topas_definition():
    yobs = np.array([100.0, 200.0, 400.0])
    ycalc = np.array([110.0, 190.0, 400.0])
    rwp, _, weighting = scoring.r_factors(yobs, ycalc)
    expected = np.sqrt(np.sum((yobs - ycalc) ** 2 / yobs) / np.sum(yobs))
    assert weighting == "poisson"
    assert rwp == pytest.approx(100 * expected)


def test_lattice_error_against_a_refined_cif(tmp_path):
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    cif = tmp_path / "refined.cif"
    cif.write_text(
        "data_x\n_cell_length_a 5.6682(4)\n_cell_length_b 5.6682(4)\n"
        "_cell_length_c 5.6682(4)\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
    )
    (phase,) = scoring.score_phases(entry, [cif])
    assert phase.max_length_rel_error == pytest.approx(0.0282 / 5.64, rel=1e-3)
    assert phase.volume_rel_error == pytest.approx((5.6682 / 5.64) ** 3 - 1, rel=1e-3)


def test_prepare_structure_mode_perturbs_the_starting_cell(tmp_path):
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    task = runner.prepare(entry, tmp_path, "structure", perturb=0.02)
    strain = task["perturbation"]["isotropic_strain"]
    assert 0.01 <= abs(strain) < 0.02
    assert task["starting_cifs"] == ["TEST_pattern_0_start_phase1.cif"]
    a = scoring.read_cif_lattice(tmp_path / task["starting_cifs"][0]).a
    assert a == pytest.approx(5.64 * (1 + strain), rel=1e-6)
    message = runner.task_message(task, tmp_path)
    assert "Kα1 = 1.54056" in message and "FINAL:" in message


def test_blind_mode_gives_nothing_away(tmp_path):
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    task = runner.prepare(entry, tmp_path, "blind")
    message = runner.task_message(task, tmp_path)
    assert task["elements"] is None and not task["starting_cifs"]
    assert "Na" not in message.replace("Nothing", "")


def test_radiation_guess():
    from guillemot.benchmark.opxrd import guess_radiation

    assert guess_radiation(1.54056, 1.54439, 80) == "xray_lab_doublet"
    assert guess_radiation(0.4133, None, 50) == "xray_synchrotron"
    assert guess_radiation(2.0775, None, 166.25) == "neutron_likely"
    assert guess_radiation(1.5482, None, 151.8) == "neutron_likely"
    assert guess_radiation(1.5406, None, 90) == "single_wavelength"


# --------------------------------------------------------------------------------------
# read_file: what the agent can and cannot see
# --------------------------------------------------------------------------------------


@pytest.fixture
def in_repo(tmp_path, monkeypatch):
    """A working directory with a session, shared dirs, and things that must stay hidden."""
    from guillemot.session import start_session

    monkeypatch.chdir(tmp_path)
    session = start_session("bench/run/entry_a")
    (session.directory / "riet_01.inp").write_text(
        "\n".join(f"line {n}" for n in range(1, 31))
    )
    other = tmp_path / "run_dir" / "bench" / "run" / "entry_b"
    other.mkdir(parents=True)
    (other / "secret.cif").write_text("answer")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "label.json").write_text("{}")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "ok.inp").write_text("xdd a.xy\n")
    (session.directory / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
    return session.directory


def test_read_file_numbers_lines_and_pages(in_repo):
    from guillemot.tools.files import read_file

    page = read_file(str(in_repo / "riet_01.inp"), start_line=11, max_lines=5)
    assert page.first_line == 11 and page.last_line == 15
    assert page.text.splitlines()[0] == "11 | line 11"
    assert page.truncated and page.total_lines == 30
    assert not read_file(str(in_repo / "riet_01.inp"), start_line=26).truncated
    assert read_file("examples/ok.inp").text == "1 | xdd a.xy"


@pytest.mark.parametrize(
    "path",
    [
        "run_dir/bench/run/entry_b/secret.cif",  # another entry's session
        "run_dir/bench/run/entry_a/../entry_b/secret.cif",  # ... reached by ..
        "datasets/label.json",  # the labels
        "/etc/passwd",
    ],
)
def test_read_file_refuses_outside_its_directories(in_repo, path):
    from pydantic_ai import ModelRetry

    from guillemot.tools.files import read_file

    with pytest.raises(ModelRetry, match="outside the files you can read"):
        read_file(path)


def test_read_file_refuses_binary(in_repo):
    from pydantic_ai import ModelRetry

    from guillemot.tools.files import read_file

    with pytest.raises(ModelRetry, match="not a text file"):
        read_file(str(in_repo / "plot.png"))


def test_failed_run_still_accounts_for_usage_and_tool_calls():
    """A tool that raises ends the run; what it cost up to then must still be recorded."""
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from guillemot.streaming import run_and_show_result

    def boom() -> str:
        """Fail."""
        raise ValueError("kaboom")

    async def stream(messages, info):
        yield {0: DeltaToolCall(name="boom", json_args="{}")}

    agent = Agent(FunctionModel(stream_function=stream), tools=[boom])
    capture: dict = {}
    with pytest.raises(ValueError, match="kaboom"):
        asyncio.run(run_and_show_result(agent, "go", capture=capture))

    run = capture["run"]
    assert run.usage().requests == 1
    assert runner._tool_counts(list(run.ctx.state.message_history))["tool_calls"] == {
        "boom": 1
    }


def test_deadline_asks_child_to_stop_before_killing(tmp_path):
    """At the deadline the child gets SIGTERM, and time to write down what it did."""
    import subprocess
    import sys

    marker = tmp_path / "stopped"
    child = (
        "import signal, sys, time, pathlib\n"
        f"def stop(*_): pathlib.Path({str(marker)!r}).write_text('ok'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runner._run_with_deadline([sys.executable, "-c", child], env=None, timeout_s=2)
    assert marker.read_text() == "ok"


def test_cancelled_run_still_accounts_for_usage():
    """Cancellation (how a timeout reaches the agent) leaves the run readable."""
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from guillemot.streaming import run_and_show_result

    def slow() -> str:
        """Take a while."""
        import time

        time.sleep(0.5)
        return "done"

    async def stream(messages, info):
        yield {0: DeltaToolCall(name="slow", json_args="{}")}

    agent = Agent(FunctionModel(stream_function=stream), tools=[slow])

    async def main():
        capture: dict = {}
        task = asyncio.ensure_future(run_and_show_result(agent, "go", capture=capture))
        await asyncio.sleep(1.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return capture["run"]

    run = asyncio.run(main())
    assert run.usage().requests >= 1
    assert (
        runner._tool_counts(list(run.ctx.state.message_history))["tool_calls"]["slow"]
        >= 1
    )


def test_read_file_summarises_patterns_and_fits(in_repo):
    from guillemot.tools.files import read_file

    x = np.linspace(10, 80, 701)
    y = (
        20
        + 1000 * np.exp(-((x - 29.2) ** 2) / 0.01)
        + 400 * np.exp(-((x - 41.7) ** 2) / 0.01)
    )
    np.savetxt(in_repo / "sample.xy", np.column_stack([x, y]))
    np.savetxt(
        in_repo / "riet_01_output.txt", np.column_stack([x, y, np.full_like(y, 20)])
    )

    pattern = read_file(str(in_repo / "sample.xy"))
    assert pattern.kind == "pattern_summary"
    assert (
        "701 points" in pattern.text
        and "29.20°" in pattern.text
        and "41.70°" in pattern.text
    )
    assert "Rwp" not in pattern.text

    fit = read_file(str(in_repo / "riet_01_output.txt"))
    assert "Rwp" in fit.text and "Largest misfits: " in fit.text
    assert "29.20° (obs" in fit.text

    # an .inp is never mistaken for data, however many numbers it has
    assert read_file(str(in_repo / "riet_01.inp")).kind == "text"


def test_deadline_note_only_when_time_is_short(monkeypatch):
    import time

    from guillemot import deadline

    monkeypatch.delenv("GUILLEMOT_DEADLINE", raising=False)
    assert deadline.minutes_remaining() is None and deadline.wrap_up_note() is None

    monkeypatch.setenv("GUILLEMOT_TIME_BUDGET_S", "5400")
    monkeypatch.setenv("GUILLEMOT_DEADLINE", str(time.time() + 3000))
    assert deadline.minutes_remaining() == pytest.approx(50, abs=0.2)
    assert deadline.wrap_up_note() is None  # 50 of 90 minutes left

    monkeypatch.setenv("GUILLEMOT_DEADLINE", str(time.time() + 1200))
    assert "FINAL" in deadline.wrap_up_note()  # 20 minutes: under a quarter left


def test_task_message_states_the_time_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GUILLEMOT_TIME_BUDGET_S", "5400")
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    message = runner.task_message(
        runner.prepare(entry, tmp_path, "structure"), tmp_path
    )
    assert "You have 90 minutes" in message


@pytest.mark.parametrize(
    "line",
    [
        "FINAL: riet_02",
        "FINAL: riet_02.inp",
        "FINAL: `riet_02.inp`",
        "**FINAL:** riet_02.inp",
        "FINAL: run_dir/bench/run/HKUST-A_pattern_4/riet_02.inp",
        "FINAL: C:\\topas\\guillemot\\riet_02.inp",
    ],
)
def test_final_line_names_the_declared_refinement(line):
    match = runner.FINAL_RE.search(f"Summary of the work.\n\n{line}\n")
    assert match and match.group(1) == "riet_02"


def test_dropped_model_connection_is_resumed_not_failed(tmp_path, monkeypatch):
    """A network blip mid-entry resumes the same run instead of ending the entry."""
    import asyncio

    import httpx
    from pydantic_ai import Agent
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from guillemot import main as app
    from guillemot.session import start_session

    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    def list_available_data() -> str:
        """List."""
        return "nothing"

    async def stream(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {0: DeltaToolCall(name="list_available_data", json_args="{}")}
        elif calls["n"] == 2:
            raise httpx.RemoteProtocolError("peer closed connection")
            yield ""
        else:
            yield "Done.\nFINAL: riet_01"

    agent = Agent(FunctionModel(stream_function=stream), tools=[list_available_data])

    def create_agent(prompt=None, session_id=None, exclude_tools=None):
        start_session(session_id)
        return agent

    monkeypatch.setattr(app, "create_agent", create_agent)
    monkeypatch.setattr(
        runner, "load_entry", lambda _: parse_entry(raw_entry([NACL]), "T/p")
    )
    monkeypatch.setattr(runner, "preflight_model", lambda model: None)

    outcome = asyncio.run(runner._run_agent("run", "T/p", "structure", None, 60, 0.01))
    assert outcome["status"] == "ok" and outcome["resumes"] == 1
    assert outcome["output"].endswith("FINAL: riet_01")
    # both halves are counted; the request that died mid-stream completed nothing
    assert outcome["usage"]["requests"] == 2
    assert outcome["tool_calls"] == {"list_available_data": 1}


def test_connection_errors_are_told_apart_from_agent_errors():
    import httpx

    wrapped = RuntimeError("model failed")
    wrapped.__cause__ = httpx.ConnectError("refused")
    assert runner._is_connection_error(wrapped)
    assert runner._is_connection_error(
        httpx.RemoteProtocolError("incomplete chunked read")
    )
    assert not runner._is_connection_error(ValueError("bad .inp"))


def test_answers_without_a_final_line_are_sent_back():
    """A model that announces its plan and stops is told it has not finished."""
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    from guillemot.streaming import run_and_show_result

    replies = iter(["I'll start by converting the CIF.", "Done.\nFINAL: riet_01"])

    async def stream(messages, info):
        yield next(replies)

    agent = Agent(FunctionModel(stream_function=stream))
    runner.require_final_answer(agent)
    result = asyncio.run(run_and_show_result(agent, "go"))
    assert result.output.endswith("FINAL: riet_01")
