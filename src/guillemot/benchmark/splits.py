"""The opXRD index, and deterministic train/validation/test/holdout splits over it.

opXRD publishes no split of its own, so we define one that other methods (rietx, the
autodiff refiners, ML models) can be held to exactly:

* Patterns are assigned by *group*, not individually: every pattern of the same
  compound — the same set of reduced formulas — lands in the same split, as does
  every pattern of one in situ series. Otherwise a model could be tuned on one
  pattern of a compound and tested on its near-twin.
* A group's split is decided by a salted SHA-256 of its key alone, so assignments
  never depend on what else is in the database, and survive it growing.
* The holdout split is for final numbers only. Do not look at holdout results while
  developing prompts or tools; that is what validation is for.

The assignment for every labelled pattern is committed in `benchmark/opxrd/splits.csv`,
so a split can be used without the code here, and a change to it shows up in review.
"""

import csv
import hashlib
import json
import os
import pathlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

from guillemot.benchmark.opxrd import (
    RADIATION_NEUTRON,
    TIER_NONE,
    iter_entry_ids,
    load_entry,
    opxrd_dir,
)

SPLIT_SALT = "guillemot-opxrd-v1"
# Cumulative upper bounds on the hash fraction for each split, in order.
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.1, "test": 0.2, "holdout": 0.1}
SPLITS = tuple(SPLIT_FRACTIONS)

DEFAULT_MANIFEST = pathlib.Path("benchmark/opxrd/splits.csv")


def index_path(root: str | os.PathLike | None = None) -> pathlib.Path:
    return opxrd_dir(root).parent / "index.csv"


def split_for(group_key: str, salt: str = SPLIT_SALT) -> str:
    """The split a group belongs to: a pure function of its key."""
    digest = hashlib.sha256(f"{salt}:{group_key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    upper = 0.0
    for name, share in SPLIT_FRACTIONS.items():
        upper += share
        if fraction < upper:
            return name
    return SPLITS[-1]


def _index_one(args: tuple[str, str]) -> dict:
    entry_id, root = args
    try:
        row = load_entry(entry_id, root).index_row()
        row["error"] = ""
    except Exception as exc:  # one bad file should not sink an index of 92,000
        row = {"entry_id": entry_id, "label_tier": TIER_NONE, "error": repr(exc)}
        row["group_key"] = f"entry:{entry_id}"
    row["split"] = split_for(row["group_key"])
    return row


def build_index(
    root: str | os.PathLike | None = None, workers: int | None = None
) -> list[dict]:
    """Read every entry once and summarise it. Takes a minute or two for all of opXRD."""
    root = str(opxrd_dir(root))
    ids = list(iter_entry_ids(root))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_index_one, [(i, root) for i in ids], chunksize=256))


def write_rows(rows: Iterable[dict], path: str | os.PathLike) -> pathlib.Path:
    rows = list(rows)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_index(root: str | os.PathLike | None = None) -> list[dict]:
    path = index_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No opXRD index at {path}. Build it with `guillemot-bench index`."
        )
    return read_rows(path)


MANIFEST_COLUMNS = (
    "entry_id",
    "split",
    "source",
    "is_simulated",
    "radiation",
    "label_tier",
    "group_key",
    "formulas",
    "spacegroups",
    "n_phases",
    "kalpha1",
    "kalpha2",
    "tth_min",
    "tth_max",
    "n_points",
)


def write_manifest(
    rows: list[dict], path: str | os.PathLike = DEFAULT_MANIFEST
) -> pathlib.Path:
    """Commit-able record of the split of every labelled pattern, plus a summary."""
    labelled = [
        {k: r.get(k, "") for k in MANIFEST_COLUMNS}
        for r in rows
        if r.get("label_tier") != TIER_NONE and not r.get("error")
    ]
    labelled.sort(key=lambda r: r["entry_id"])
    path = write_rows(labelled, path)

    summary = {
        "dataset": "opXRD",
        "zenodo_record": "15298026",
        "archive_md5": "e8e9800a8ecd2b7f01a764fdf77d5c53",
        "salt": SPLIT_SALT,
        "fractions": SPLIT_FRACTIONS,
        "grouping": "reduced formulas of all labelled phases; LBNL in situ series; else entry",
        "counts": summarise(rows),
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return path


def summarise(rows: list[dict]) -> dict:
    """Patterns and groups per split, per label tier."""
    out: dict = {}
    for tier in sorted({r.get("label_tier", TIER_NONE) for r in rows}):
        tier_rows = [r for r in rows if r.get("label_tier") == tier]
        patterns = Counter(r["split"] for r in tier_rows)
        groups = Counter(
            split for split, _ in {(r["split"], r["group_key"]) for r in tier_rows}
        )
        out[tier] = {
            s: {"patterns": patterns.get(s, 0), "groups": groups.get(s, 0)}
            for s in SPLITS
        }
    return out


def select(
    rows: list[dict],
    split: str | None = None,
    tier: str | None = None,
    sources: list[str] | None = None,
    max_phases: int | None = None,
    require_wavelength: bool = True,
    include_neutron: bool = False,
) -> list[dict]:
    """Filter index or manifest rows down to a benchmark population.

    Likely neutron patterns are left out unless asked for: the harness describes every
    task to the agent as X-ray diffraction.
    """

    def keep(r: dict) -> bool:
        if r.get("error"):
            return False
        if split and r.get("split") != split:
            return False
        if tier and r.get("label_tier") != tier:
            return False
        if sources and r.get("source") not in sources:
            return False
        if max_phases is not None and int(r.get("n_phases") or 0) > max_phases:
            return False
        if require_wavelength and not r.get("kalpha1"):
            return False
        if str(r.get("is_simulated", "")).lower() == "true":
            return False
        if not include_neutron and r.get("radiation") == RADIATION_NEUTRON:
            return False
        return True

    return [r for r in rows if keep(r)]
