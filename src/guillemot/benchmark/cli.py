"""`guillemot-bench`: index opXRD, split it, export tasks, run and score guillemot.

guillemot-bench index                       # once, after unpacking the archive
guillemot-bench splits                      # rewrite benchmark/opxrd/splits.csv
guillemot-bench list --split test --tier structure
guillemot-bench export --split test --tier structure --out exports/opxrd-test
guillemot-bench run --name qwen-smoke --entry CNRS/pattern_1 --mode structure
guillemot-bench run --name qwen-val --split validation --tier structure --limit 20
guillemot-bench rescore --name qwen-val
guillemot-bench summary --name qwen-val
"""

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import warnings

from dotenv import load_dotenv

from guillemot.benchmark import runner, splits
from guillemot.benchmark.opxrd import TIER_STRUCTURE, load_entry

# pymatgen complains about the electronegativity of the noble gases adsorbed in some
# of the CNRS frameworks every time a formula is built; it is not news.
warnings.filterwarnings("ignore", message="No Pauling electronegativity")


def _read_entries_file(path: pathlib.Path) -> list[str]:
    """Entry ids from a file, one per line; `#` starts a comment."""
    ids = []
    for line in path.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            ids.append(entry)
    return ids


def _selection(args) -> list[str]:
    if args.entry or args.entries_file:
        return (args.entry or []) + (
            _read_entries_file(args.entries_file) if args.entries_file else []
        )
    rows = splits.select(
        splits.read_rows(args.manifest)
        if args.manifest.is_file()
        else splits.load_index(),
        split=args.split,
        tier=args.tier,
        sources=args.source,
        max_phases=args.max_phases,
        include_neutron=args.include_neutron,
    )
    # A stable shuffle, so `--limit 20` is a fair sample and the same one every time.
    rows.sort(
        key=lambda r: hashlib.sha256(f"{args.seed}:{r['entry_id']}".encode()).digest()
    )
    if args.one_per_group:
        # opXRD repeats some compounds many times (20 patterns of C9NCl in validation);
        # one pattern per compound gives a more diverse sample for the same effort.
        seen: set[str] = set()
        rows = [
            r for r in rows if not (r["group_key"] in seen or seen.add(r["group_key"]))
        ]
    ids = [r["entry_id"] for r in rows]
    return ids[: args.limit] if args.limit else ids


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entry", action="append", help="an entry id; repeatable")
    parser.add_argument(
        "--entries-file",
        type=pathlib.Path,
        help="a file of entry ids, one per line (e.g. benchmark/opxrd/dev.txt)",
    )
    parser.add_argument("--split", choices=splits.SPLITS)
    parser.add_argument(
        "--tier", default=TIER_STRUCTURE, help="label tier (default: structure)"
    )
    parser.add_argument(
        "--source", action="append", help="e.g. CNRS, HKUST-A; repeatable"
    )
    parser.add_argument("--max-phases", type=int, default=None)
    parser.add_argument(
        "--include-neutron",
        action="store_true",
        help="keep patterns that look like neutron data (left out by default)",
    )
    parser.add_argument(
        "--one-per-group",
        action="store_true",
        help="keep one pattern per compound (group), chosen by the --seed shuffle",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seed", default="0", help="seed for the sample taken by --limit"
    )
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=splits.DEFAULT_MANIFEST
    )


def cmd_index(args) -> None:
    rows = splits.build_index(workers=args.workers)
    path = splits.write_rows(rows, splits.index_path())
    print(f"Indexed {len(rows)} entries to {path}")
    print(json.dumps(splits.summarise(rows), indent=2))


def cmd_splits(args) -> None:
    path = splits.write_manifest(splits.load_index(), args.manifest)
    print(f"Wrote {path} and {path.with_suffix('.json')}")


def cmd_list(args) -> None:
    for entry_id in _selection(args):
        print(entry_id)


def cmd_export(args) -> None:
    """Task bundles for other methods: the same pattern, CIFs and task.json guillemot gets."""
    out = pathlib.Path(args.out)
    ids = _selection(args)
    for entry_id in ids:
        runner.prepare(
            load_entry(entry_id), out / runner.slug(entry_id), args.mode, args.perturb
        )
    (out / "entries.txt").write_text("\n".join(ids) + "\n")
    print(f"Exported {len(ids)} tasks to {out}")


def cmd_run(args) -> None:
    ids = _selection(args)
    if not ids:
        sys.exit("No entries selected.")
    try:
        path = runner.run_benchmark(
            ids,
            run_name=args.name,
            mode=args.mode,
            prompt=args.prompt,
            model=args.model,
            timeout_s=args.timeout,
            max_requests=args.max_requests,
            perturb=args.perturb,
            results_dir=args.results_dir,
            split=args.split,
            concurrency=args.concurrency,
        )
    except runner.InfrastructureError as exc:
        sys.exit(f"Stopping: {exc}\nRerun the same command to resume.")
    print(f"Results in {path}")


def cmd_run_one(args) -> None:
    runner.run_one_in_process(args)


def cmd_rescore(args) -> None:
    """Score every entry of a run again from its session directory."""
    from guillemot.session import RUN_DIR

    results = args.results_dir / f"{args.name}.jsonl"
    old = {}
    if results.is_file():
        old = {
            (r := json.loads(line))["entry_id"]: r
            for line in results.read_text().splitlines()
            if line
        }
    root = RUN_DIR / runner.BENCH_SESSION_ROOT / args.name
    records = []
    for directory in sorted(
        p for p in root.iterdir() if (p / runner.TASK_FILE).is_file()
    ):
        record = runner.score_entry(directory)
        previous = old.get(record["entry_id"], {})
        for key in ("wall_seconds", "exit", "split"):
            if key in previous:
                record[key] = previous[key]
        if previous.get("status") == "timeout":
            record["status"] = "timeout"
        records.append(record)
    results.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"Rescored {len(records)} entries into {results}")


def cmd_summary(args) -> None:
    results = args.results_dir / f"{args.name}.jsonl"
    records = [json.loads(line) for line in results.read_text().splitlines() if line]

    def stat(values):
        values = [v for v in values if v is not None]
        if not values:
            return "—"
        return f"median {statistics.median(values):.3g} (n={len(values)})"

    finals = [r.get("final") or {} for r in records]
    valid = [f for f in finals if f.get("valid")]
    print(f"{args.name}: {len(records)} entries")
    print(f"  completed refinement (final valid): {len(valid)}/{len(records)}")
    print(
        f"  status: { {s: sum(r.get('status') == s for r in records) for s in {r.get('status') for r in records}} }"
    )
    print(f"  Rwp (final, ours):   {stat(f['rwp'] for f in valid)}")
    print(
        f"  Rwp (best, ours):    {stat((r.get('best') or {}).get('rwp') for r in records)}"
    )
    print(
        f"  Rwp (TOPAS-reported):{stat(f.get('reported', {}).get('r_wp') for f in valid)}"
    )
    phases = [p for f in valid for p in f.get("phases", [])]
    print(f"  |ΔV|/V:              {stat(p.get('volume_rel_error') for p in phases)}")
    print(
        f"  max |Δa|/a:          {stat(p.get('max_length_rel_error') for p in phases)}"
    )
    print(f"  TOPAS runs/entry:    {stat(r.get('n_topas_runs') for r in records)}")
    print(f"  wall s/entry:        {stat(r.get('wall_seconds') for r in records)}")
    print(
        f"  tokens in/entry:     {stat((r.get('agent', {}).get('usage') or {}).get('input_tokens') for r in records)}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="guillemot-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="summarise every opXRD entry into the index")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser(
        "splits", help="write the committed split manifest from the index"
    )
    p.add_argument("--manifest", type=pathlib.Path, default=splits.DEFAULT_MANIFEST)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("list", help="list the entries a selection picks")
    _add_selection(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("export", help="write task bundles for other methods")
    _add_selection(p)
    p.add_argument("--mode", choices=runner.MODES, default="structure")
    p.add_argument("--perturb", type=float, default=runner.DEFAULT_PERTURB)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    for name, func, helptext in (
        ("run", cmd_run, "run guillemot on a selection and score it"),
        ("_run-one", cmd_run_one, argparse.SUPPRESS),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--mode", choices=runner.MODES, default="structure")
        p.add_argument("--prompt", default=None)
        p.add_argument("--max-requests", type=int, default=60)
        p.add_argument("--perturb", type=float, default=runner.DEFAULT_PERTURB)
        if name == "run":
            p.add_argument("--name", required=True, help="name of this benchmark run")
            _add_selection(p)
            p.add_argument("--model", default=None, help="overrides GUILLEMOT_AI_MODEL")
            p.add_argument(
                "--timeout", type=int, default=3600, help="seconds per entry"
            )
            p.add_argument(
                "--concurrency",
                type=int,
                default=1,
                help="entries to run at once; match the model server's parallelism",
            )
            p.add_argument(
                "--results-dir", type=pathlib.Path, default=runner.DEFAULT_RESULTS_DIR
            )
        else:
            p.add_argument("--run-name", required=True)
            p.add_argument("--entry", required=True)
        p.set_defaults(func=func)

    for name, func in (("rescore", cmd_rescore), ("summary", cmd_summary)):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument(
            "--results-dir", type=pathlib.Path, default=runner.DEFAULT_RESULTS_DIR
        )
        p.set_defaults(func=func)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    load_dotenv()
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
