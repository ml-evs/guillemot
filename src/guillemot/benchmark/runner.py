"""Running guillemot over opXRD entries, and the task bundles other methods share.

A benchmark *run* is one configuration (model, prompt, mode) applied to a list of
entries. Each entry is refined in a child process of its own, because the session
directory and the tracing are process-wide, and because a hard wall-clock limit is
then just a matter of killing the child. Everything an entry produced is in
`run_dir/bench/<run name>/<entry slug>/`, and scoring is done afterwards from that
directory alone, so a run can be re-scored when the scoring changes.

The modes say how much the agent is told, from most to least:

* ``structure`` — the labelled structure is handed over as a starting CIF, with its
  cell perturbed (see `perturb`) so recovering it is not free. This is the setting in
  which guillemot is comparable with rietx and the autodiff refiners, which all need a
  starting model; `export` writes exactly the same bundle for them.
* ``composition`` — only the elements present; the agent has to find the phases.
* ``blind`` — just the pattern and the wavelength.
"""

import asyncio
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import time

from guillemot.benchmark.opxrd import OpxrdEntry, load_entry, reference_structure
from guillemot.benchmark.scoring import score

MODES = ("structure", "composition", "blind")
BENCH_SESSION_ROOT = "bench"  # under run_dir/
DEFAULT_RESULTS_DIR = pathlib.Path("benchmark/results")
DEFAULT_PERTURB = 0.01
TASK_FILE = "task.json"
AGENT_RESULT_FILE = "agent_result.json"
# Tools the agent does not get in a benchmark: the task is self-contained, and these
# reach out to a datalab instance that has nothing to do with it.
EXCLUDED_TOOLS = {"get_samples", "get_sample", "list_data_files"}
# The declared answer: a bare name, a filename or a full path, optionally in backticks
# or bold. Only the file's stem matters.
FINAL_RE = re.compile(
    r"FINAL:\**\s*[`*]*(?:[^\s`*]*[/\\])?([A-Za-z0-9._-]+?)(?:\.inp)?[`*]*\s*$",
    re.MULTILINE,
)


def slug(entry_id: str) -> str:
    """'LBNL/LBNL-A/pattern_group_0_0' to 'LBNL-A_pattern_group_0_0'; safe for TOPAS."""
    parts = entry_id.split("/")
    return re.sub(r"[^A-Za-z0-9._-]", "_", "_".join(parts[-2:]))


def _unit_interval(*keys: str) -> float:
    digest = hashlib.sha256(":".join(keys).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def perturbation(entry_id: str, size: float) -> float:
    """A deterministic isotropic strain in [-size, size] (excluding the middle half),
    the same for every method and every run, so starting points are comparable."""
    u = _unit_interval("perturb", entry_id)
    sign = 1 if u < 0.5 else -1
    magnitude = size * (0.5 + (u % 0.5))  # in [size/2, size): never trivially small
    return sign * magnitude


def wavelength_text(entry: OpxrdEntry) -> str:
    if entry.kalpha1 and entry.kalpha2:
        return (
            f"a Kα doublet with Kα1 = {entry.kalpha1:.5f} Å and Kα2 = {entry.kalpha2:.5f} Å "
            "(Cu Kα; the Kα2 contribution is present unless you see otherwise)"
        )
    if entry.kalpha1:
        return f"a single wavelength, λ = {entry.kalpha1:.5f} Å"
    return "an unknown wavelength"


def prepare(
    entry: OpxrdEntry,
    directory: pathlib.Path,
    mode: str,
    perturb: float = DEFAULT_PERTURB,
) -> dict:
    """Write the files for one task into `directory` and describe it.

    The description (`task.json`) holds only what a method is allowed to know. The
    answer is never written here: it is recomputed from opXRD when scoring.
    """
    from pymatgen.io.cif import CifWriter

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    directory.mkdir(parents=True, exist_ok=True)
    name = slug(entry.entry_id)
    pattern = entry.write_xy(directory / f"{name}.xy")

    task = {
        "entry_id": entry.entry_id,
        "mode": mode,
        "pattern": pattern.name,
        "two_theta_range": [min(entry.two_theta), max(entry.two_theta)],
        "kalpha1": entry.kalpha1,
        "kalpha2": entry.kalpha2,
        "elements": entry.elements if mode in ("structure", "composition") else None,
        "starting_cifs": [],
        "perturbation": None,
    }
    if mode == "structure":
        strain = perturbation(entry.entry_id, perturb)
        task["perturbation"] = {"isotropic_strain": strain, "size": perturb}
        for k, phase in enumerate(p for p in entry.phases if p.has_structure):
            structure = reference_structure(phase)
            structure.scale_lattice(structure.volume * (1 + strain) ** 3)
            cif = directory / f"{name}_start_phase{k + 1}.cif"
            CifWriter(structure, symprec=0.01).write_file(cif)
            task["starting_cifs"].append(cif.name)

    (directory / TASK_FILE).write_text(json.dumps(task, indent=2) + "\n")
    return task


def task_message(task: dict, directory: pathlib.Path) -> str:
    """The instruction given to guillemot for one benchmark task."""
    lines = [
        f"Refine the powder X-ray diffraction pattern in `{directory / task['pattern']}`.",
        f"It is already in your session directory. It was measured with "
        f"{wavelength_text_from_task(task)}, from 2θ = {task['two_theta_range'][0]:.2f}° "
        f"to {task['two_theta_range'][1]:.2f}°. Nothing else is known about the "
        "instrument.",
    ]
    if task["mode"] == "structure":
        # Named, not pasted: a CIF's full list of symmetry operators invited one model
        # to spend an hour re-deriving Wyckoff multiplicities by hand instead of
        # converting it. The file is exactly what other methods are given.
        cifs = ", ".join(f"`{directory / name}`" for name in task["starting_cifs"])
        lines.append(
            f"A starting structural model for each phase present is in {cifs}. Its "
            "lattice parameters are approximate and must be refined. Convert it with "
            "`cif_to_topas_str` rather than reading or transcribing it."
        )
    elif task["mode"] == "composition":
        lines.append(
            f"The sample contains the elements {', '.join(task['elements'])}. Which "
            "phases are present, and how many, is for you to determine."
        )
    else:
        lines.append(
            "Nothing is known about its composition: identify the phases yourself."
        )
    lines += [
        "",
        *_time_budget_lines(),
        "This is a benchmark task, so work end to end without asking me anything. "
        "Refine against the whole measured 2θ range: do not exclude regions of the "
        "pattern. Keep improving the refinement until you judge it to be as good as "
        "you can make it.",
        "Finish with a short summary, and make the last line of your answer "
        "`FINAL: <name of the .inp file whose refinement is your final answer>`.",
    ]
    return "\n".join(lines)


def _time_budget_lines() -> list[str]:
    budget = os.getenv("GUILLEMOT_TIME_BUDGET_S")
    if not budget:
        return []
    return [
        f"You have {float(budget) / 60:.0f} minutes for this task, after which the run "
        "is stopped; refinement and plotting results tell you how much time is left. "
        "Aim to give your FINAL answer with time to spare.",
    ]


def wavelength_text_from_task(task: dict) -> str:
    stub = OpxrdEntry(task["entry_id"], "", [], [])
    stub.kalpha1, stub.kalpha2 = task["kalpha1"], task["kalpha2"]
    return wavelength_text(stub)


# --------------------------------------------------------------------------------------
# Running one entry (in a child process)
# --------------------------------------------------------------------------------------


def session_id_for(run_name: str, entry_id: str) -> str:
    return f"{BENCH_SESSION_ROOT}/{run_name}/{slug(entry_id)}"


def _tool_counts(messages) -> dict:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
    )

    calls: dict[str, int] = {}
    retries = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    calls[part.tool_name] = calls.get(part.tool_name, 0) + 1
        elif isinstance(message, ModelRequest):
            retries += sum(isinstance(p, RetryPromptPart) for p in message.parts)
    return {"tool_calls": calls, "retries": retries}


async def _run_agent(
    run_name: str,
    entry_id: str,
    mode: str,
    prompt: str | None,
    max_requests: int,
    perturb: float,
) -> dict:
    from pydantic_ai.usage import UsageLimits

    from guillemot import main as app
    from guillemot.session import session_dir
    from guillemot.streaming import run_and_show_result

    agent = app.create_agent(
        prompt,
        session_id=session_id_for(run_name, entry_id),
        exclude_tools=EXCLUDED_TOOLS,
    )
    require_final_answer(agent)
    directory = session_dir()
    task = prepare(load_entry(entry_id), directory, mode, perturb=perturb)
    message = task_message(task, directory)
    (directory / "task_message.md").write_text(message)

    outcome: dict = {"status": "ok", "resumes": 0}
    started = time.monotonic()
    totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    prompt, history, messages = message, None, []

    def account(run) -> list:
        """Add a (partial) run's usage to the totals; return its messages."""
        if run is None:
            return messages
        usage = run.usage()
        totals["requests"] += usage.requests
        totals["input_tokens"] += usage.input_tokens or 0
        totals["output_tokens"] += usage.output_tokens or 0
        return list(run.ctx.state.message_history)

    while True:
        capture: dict = {}
        # The parent asks us to stop with SIGTERM when the entry runs out of time;
        # cancel the run so the accounting below still happens, before SIGKILL.
        running = asyncio.ensure_future(
            run_and_show_result(
                agent,
                prompt,
                app.CONTEXT_LIMIT,
                usage_limits=UsageLimits(
                    request_limit=max(1, max_requests - totals["requests"])
                ),
                capture=capture,
                message_history=history,
            )
        )
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, running.cancel)
        try:
            result = await running
            outcome["output"] = result.output
            messages = account(capture.get("run"))
            break
        except asyncio.CancelledError:
            outcome["status"] = "timeout"
            messages = account(capture.get("run"))
            break
        except Exception as exc:  # the score still counts what was produced before this
            from pydantic_ai.exceptions import UsageLimitExceeded

            messages = account(capture.get("run"))
            # A dropped connection to the model is the network's fault, not the
            # agent's: wait for the server and carry on from where the run stopped.
            if (
                _is_connection_error(exc)
                and outcome["resumes"] < MAX_RESUMES
                and messages
                and await _wait_for_model()
            ):
                outcome["resumes"] += 1
                print(f"\n⚠ model connection lost ({exc!r}); resuming", flush=True)
                prompt, history = None, messages
                continue
            outcome["status"] = (
                "request_limit" if isinstance(exc, UsageLimitExceeded) else "error"
            )
            outcome["error"] = repr(exc)
            break

    if totals["requests"]:
        outcome["usage"] = totals
    outcome |= _tool_counts(messages)
    if messages:
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        (directory / "messages.json").write_bytes(
            ModelMessagesTypeAdapter.dump_json(messages)
        )
    outcome["agent_seconds"] = round(time.monotonic() - started, 1)
    (directory / AGENT_RESULT_FILE).write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def require_final_answer(agent) -> None:
    """Send back any answer that ends the run without a FINAL line.

    Some models announce what they will do ("I'll start by converting the CIF") and
    stop, which ends an agent run with nothing done; one wrote a whole fictional
    refinement instead of calling a tool. Neither is a benchmark answer, so the model
    is told it is not finished. A model that does give its FINAL line never sees this.
    """
    from pydantic_ai import ModelRetry

    @agent.output_validator
    def _has_final_line(output: str) -> str:
        if FINAL_RE.search(output or ""):
            return output
        raise ModelRetry(
            "You have not finished: your answer has no `FINAL:` line. If you have not "
            "yet run a refinement, do it now with your tools — call `cif_to_topas_str` "
            "on the starting CIF, save an input with `save_topas_inp` and run it. "
            "Otherwise end your summary with `FINAL: <name of your best .inp>`."
        )


def run_one_in_process(args) -> None:
    """Entry point of the child process: run the agent on one entry."""
    asyncio.run(
        _run_agent(
            args.run_name,
            args.entry,
            args.mode,
            args.prompt,
            args.max_requests,
            args.perturb,
        )
    )


# --------------------------------------------------------------------------------------
# Finding and scoring what an entry produced
# --------------------------------------------------------------------------------------


def find_refinements(directory: pathlib.Path) -> list[dict]:
    """Every TOPAS run in a session directory, oldest first.

    A run is a subdirectory synced back from the TOPAS machine, holding the .inp it ran;
    one that produced no `<stem>_output.txt` failed.
    """
    runs = []
    for inp in directory.glob("*/*.inp"):
        stem = inp.stem
        output = inp.with_name(f"{stem}_output.txt")
        runs.append(
            {
                "stem": stem,
                "directory": inp.parent,
                "output": output if output.is_file() else None,
                "out": inp.with_suffix(".out")
                if inp.with_suffix(".out").is_file()
                else None,
                "cifs": sorted(inp.parent.glob("*.cif")),
                "mtime": inp.stat().st_mtime,
            }
        )
    return sorted(runs, key=lambda r: (r["directory"].name, r["mtime"]))


def score_entry(directory: pathlib.Path, entry: OpxrdEntry | None = None) -> dict:
    """Score everything an entry produced: the declared final answer, and the best run."""
    task = json.loads((directory / TASK_FILE).read_text())
    entry = entry or load_entry(task["entry_id"])
    agent = {}
    if (directory / AGENT_RESULT_FILE).is_file():
        agent = json.loads((directory / AGENT_RESULT_FILE).read_text())

    runs = find_refinements(directory)
    succeeded = [r for r in runs if r["output"]]

    declared = None
    match = FINAL_RE.search(agent.get("output") or "")
    if match:
        declared = match.group(1)

    final_run = None
    if declared:
        candidates = [r for r in succeeded if r["stem"] == declared]
        final_run = candidates[-1] if candidates else None
    final_source = "declared" if final_run else "last_successful"
    if final_run is None and succeeded:
        final_run = succeeded[-1]
    if final_run is None:
        final_source = "none"

    def score_run(run):
        return score(entry, run["output"], run["cifs"], run["out"]).to_dict()

    scored = [(run, score_run(run)) for run in succeeded]
    valid = [(run, s) for run, s in scored if s["valid"]]
    best = min(valid, key=lambda pair: pair[1]["rwp"])[1] if valid else None
    final = next((s for run, s in scored if run is final_run), None)

    return {
        "entry_id": task["entry_id"],
        "mode": task["mode"],
        "status": agent.get("status", "missing"),
        "final_source": final_source,
        "declared_final": declared,
        "n_topas_runs": len(runs),
        "n_topas_successful": len(succeeded),
        "final": final,
        "best": best,
        "agent": {k: v for k, v in agent.items() if k != "output"},
        "answer_tail": (agent.get("output") or "")[-1500:],
        "session_dir": str(directory),
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


class InfrastructureError(RuntimeError):
    """TOPAS cannot be reached: no entry can be attempted fairly, so stop the run."""


def preflight_topas(require_idle: bool = True) -> None:
    """Check the remote TOPAS machine answers and is idle, before spending a model run.

    Without this an unreachable TOPAS machine is indistinguishable, in the results,
    from an agent that could not refine anything — and costs a whole request budget
    per entry to find out.
    """
    from guillemot.tools.topas import check_remote_topas_running

    if not os.getenv("GUILLEMOT_TOPAS_SSH_HOST"):
        return  # local TOPAS: nothing to reach
    try:
        status = check_remote_topas_running(timeout_s=45)
    except Exception as exc:
        raise InfrastructureError(f"TOPAS machine unreachable: {exc}") from exc
    if status.running and require_idle:
        raise InfrastructureError(
            f"TOPAS is already running on {status.host} ({status.processes}); a "
            "benchmark entry must not share the machine."
        )


# How often one entry may lose its model connection and carry on, and how long to wait
# for the model server to come back each time.
MAX_RESUMES = 5
MODEL_WAIT_S = 900


def _is_connection_error(exc: BaseException) -> bool:
    """Whether `exc`, or anything that caused it, is a network failure."""
    import httpx

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (httpx.TransportError, ConnectionError)):
            return True
        if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            return True
        if type(exc).__name__ == "ModelAPIError" and "onnection" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def _wait_for_model() -> bool:
    """Wait for the configured model server to answer again; False if it never does."""
    model = os.getenv("GUILLEMOT_AI_MODEL")
    deadline = time.monotonic() + MODEL_WAIT_S
    while time.monotonic() < deadline:
        try:
            preflight_model(model)
            return True
        except InfrastructureError:
            await asyncio.sleep(15)
    return False


# How long a child asked to stop gets to write down what it did before it is killed.
STOP_GRACE_S = 30


def _run_with_deadline(
    command: list[str], env: dict, timeout_s: int, log: pathlib.Path | None = None
) -> int:
    """Run the child; at the deadline ask it to stop (SIGTERM), then kill it.

    Raises `subprocess.TimeoutExpired` if the deadline was reached, whether or not the
    child then managed to stop by itself.
    """
    output = open(log, "a") if log else None
    try:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT if output else None,
        )
    finally:
        if output:
            output.close()  # the child holds its own copy
    try:
        return child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise


def preflight_model(model: str | None) -> None:
    """Check an ollama-served model answers before handing it an entry.

    A dropped tunnel otherwise fails every remaining entry in seconds, each recorded as
    the agent failing. Other providers are not checked: their outages are rarer and
    their health endpoints differ.
    """
    import httpx

    from guillemot.model import DEFAULT_OLLAMA_URL

    provider, _, model_id = (model or "").partition(":")
    if provider != "ollama":
        return
    base = os.getenv("GUILLEMOT_OLLAMA_URL", DEFAULT_OLLAMA_URL).removesuffix("/v1")
    try:
        served = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    except Exception as exc:
        raise InfrastructureError(
            f"ollama server at {base} unreachable: {exc!r}"
        ) from exc
    if model_id not in {m.get("name") for m in served}:
        raise InfrastructureError(
            f"ollama at {base} is up but does not serve {model_id!r}"
        )


def run_benchmark(
    entry_ids: list[str],
    run_name: str,
    mode: str,
    prompt: str | None = None,
    model: str | None = None,
    timeout_s: int = 3600,
    max_requests: int = 60,
    perturb: float = DEFAULT_PERTURB,
    results_dir: pathlib.Path = DEFAULT_RESULTS_DIR,
    split: str | None = None,
    concurrency: int = 1,
) -> pathlib.Path:
    """Refine each entry and append its score to `<results_dir>/<run_name>.jsonl`.

    With `concurrency` above 1, that many entries run at once, each logging to
    `run_dir/bench/<run_name>/<entry>.log`; their TOPAS runs are queued, one at a time.
    Match it to the model server's parallelism (`OLLAMA_NUM_PARALLEL` for ollama).

    Entries already present in the results file are skipped, so an interrupted run
    can be resumed by repeating the command.
    """
    from guillemot.session import RUN_DIR

    results_dir.mkdir(parents=True, exist_ok=True)
    results = results_dir / f"{run_name}.jsonl"
    config_path = results_dir / f"{run_name}.config.json"
    model = model or os.getenv("GUILLEMOT_AI_MODEL")
    config = {
        "run_name": run_name,
        "mode": mode,
        "split": split,
        "model": model,
        "prompt": prompt or os.getenv("GUILLEMOT_PROMPT", "default"),
        "timeout_s": timeout_s,
        "max_requests": max_requests,
        "perturb": perturb,
        "concurrency": concurrency,
        "topas_host": os.getenv("GUILLEMOT_TOPAS_SSH_HOST"),
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git": _git_state(),
    }
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    done = set()
    if results.is_file():
        done = {
            json.loads(line)["entry_id"]
            for line in results.read_text().splitlines()
            if line
        }

    env = dict(os.environ)
    if model:
        env["GUILLEMOT_AI_MODEL"] = model

    import threading
    from concurrent.futures import ThreadPoolExecutor

    write_lock = threading.Lock()
    stop = threading.Event()  # set when the infrastructure fails: start nothing new
    failures: list[InfrastructureError] = []

    def run_entry(n: int, entry_id: str) -> None:
        label = f"[{n}/{len(entry_ids)}] {entry_id}"
        if entry_id in done:
            print(f"{label}: already scored, skipping", flush=True)
            return
        if stop.is_set():
            return
        try:
            # Not recorded as a result: rerunning the same command resumes from here.
            preflight_model(model)
            # With entries in parallel, TOPAS being busy is expected: runs are queued.
            preflight_topas(require_idle=concurrency == 1)
        except InfrastructureError as exc:
            stop.set()
            failures.append(exc)
            return
        print(label, flush=True)
        command = [
            sys.executable, "-m", "guillemot.benchmark.cli", "_run-one",
            "--run-name", run_name, "--entry", entry_id, "--mode", mode,
            "--max-requests", str(max_requests), "--perturb", str(perturb),
        ]  # fmt: skip
        if prompt:
            command += ["--prompt", prompt]

        # The agent's tools report the time left from these, and tell it to finish
        # in good time; a few seconds are kept back for writing up after SIGTERM.
        entry_env = env | {
            "GUILLEMOT_DEADLINE": str(time.time() + timeout_s),
            "GUILLEMOT_TIME_BUDGET_S": str(timeout_s),
        }
        directory = RUN_DIR / session_id_for(run_name, entry_id)
        # In parallel, each entry's transcript goes to its own log, not the terminal.
        log = None
        if concurrency > 1:
            directory.parent.mkdir(parents=True, exist_ok=True)
            log = directory.parent / f"{slug(entry_id)}.log"
        started = time.monotonic()
        try:
            exit_status = _run_with_deadline(command, entry_env, timeout_s, log=log)
        except subprocess.TimeoutExpired:
            exit_status = "timeout"
        wall = round(time.monotonic() - started, 1)

        if (directory / TASK_FILE).is_file():
            record = score_entry(directory)
        else:
            record = {"entry_id": entry_id, "mode": mode, "status": "not_started"}
        if exit_status == "timeout":
            record["status"] = "timeout"
        record |= {"wall_seconds": wall, "exit": exit_status, "split": split}

        with write_lock:
            with open(results, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        final = record.get("final") or {}
        print(
            f"    → {entry_id}: status={record['status']} "
            f"topas_runs={record.get('n_topas_runs')} "
            f"Rwp(final)={final.get('rwp')} valid={final.get('valid')} wall={wall}s",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for future in [
            pool.submit(run_entry, n, e) for n, e in enumerate(entry_ids, 1)
        ]:
            future.result()
    if failures:
        raise failures[0]
    return results
