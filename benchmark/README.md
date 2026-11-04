# Benchmarking guillemot

## opXRD

[opXRD](https://zenodo.org/records/15298026) (Hollarek *et al.*, 2025) collects 92,552
experimental powder patterns, 2,179 of them labelled. 912 of those labels give the
full structure (lattice and atomic positions) and the rest give partial information
(composition, space group). Those 912 are what a Rietveld benchmark can be scored on.

### Getting the data

```shell
mkdir -p datasets/opxrd
curl -L -o datasets/opxrd/opxrd.zip \
    https://zenodo.org/api/records/15298026/files/opxrd.zip/content
md5sum datasets/opxrd/opxrd.zip   # e8e9800a8ecd2b7f01a764fdf77d5c53
unzip -q datasets/opxrd/opxrd.zip -d datasets/opxrd/extracted   # 3.6 GB
uv run guillemot-bench index      # ~15 s; writes datasets/opxrd/index.csv
```

`datasets/` is git-ignored. It sits outside `data/` on purpose: `list_available_data`
walks `data/`, and the agent would otherwise get a listing of 92,000 files. Set
`GUILLEMOT_OPXRD_DIR` to keep the archive somewhere else.

### Splits

opXRD publishes no split, so we define one in `src/guillemot/benchmark/splits.py`:

| | train | validation | test | holdout |
|---|---|---|---|---|
| target fraction | 0.6 | 0.1 | 0.2 | 0.1 |
| structure-labelled patterns | 515 | 116 | 200 | 81 |
| … selectable by default | 453 | 105 | 170 | 75 |

* **Grouped by compound.** Every pattern of the same set of reduced formulas lands in
  the same split, and so does every pattern of an LBNL in situ series. opXRD has many
  repeats (25 patterns of As₂O₃, 13 of SnSe), and splitting them would leak.
* **Hash-assigned.** A group's split is a salted SHA-256 of its key and nothing else,
  so it does not change as the database grows.
* **Committed.** `benchmark/opxrd/splits.csv` lists the split of every labelled pattern,
  and `splits.json` records the counts and parameters. Other methods should take their
  entries from this file.
* **Holdout is for final numbers only.** Develop prompts and tools against validation.
  Don't look at holdout results until you are reporting.

"Selectable by default" excludes patterns with no wavelength, and patterns that look
like neutron data. opXRD does not record the radiation, but some CNRS labels have
single wavelengths of 2.08–2.52 Å measured out to 2θ = 166°, which no X-ray tube
produces. The guess is in the `radiation` column; `--include-neutron` keeps them.

Note how lopsided the structure-labelled set is: 891 of the 912 patterns come from
CNRS (Coudert's pdCIF collection, largely frameworks and molecular crystals), and
the other 21 from HKUST. Also, about a third were measured at synchrotrons.

### Modes

How much a method is told about a pattern:

* `structure`: the labelled structure is given as a starting CIF, with symmetry
  restored (opXRD stores labels expanded to P1). Its cell is isotropically strained
  by 0.5–1%, deterministically per entry, so recovering the lattice is not free. **This
  is the mode in which guillemot is comparable with rietx and the autodiff refiners**,
  all of which need a starting model.
* `composition`: the elements only. The method must identify the phases.
* `blind`: the pattern and the wavelength only.

### Running guillemot

```shell
uv run guillemot-bench list --split validation --limit 10
uv run guillemot-bench run --name qwen38-val-structure --split validation --limit 20 \
    --mode structure --model ollama:qwen3.8:27b
uv run guillemot-bench summary --name qwen38-val-structure
```

Each entry runs in its own process, with a wall-clock limit (`--timeout`, default one
hour) and a model request limit (`--max-requests`, default 60). Its session is in
`run_dir/bench/<name>/<entry>/`, and one JSON line per entry is appended to
`benchmark/results/<name>.jsonl`. The configuration and git commit are saved next to
it. Repeating the same command resumes an interrupted run.

The agent is told its time budget, and its refinement and plotting tools report the
time left (`GUILLEMOT_DEADLINE`, `GUILLEMOT_TIME_BUDGET_S`), telling it to give its final
answer once a quarter of the budget remains. At the deadline the harness asks the agent
process to stop and gives it 30 s to record its usage before killing it.

Before each entry, the harness checks that the TOPAS machine is reachable and idle,
and stops the run if not. That way an outage does not get recorded as the agent
failing.

The agent does not get the datalab tools in a benchmark, and it is asked to end its
answer with `FINAL: <inp name>`. That refinement is scored as its answer. If it
names none, the last successful refinement is scored instead. The best refinement
it ran is also reported.

### Scoring, and comparing other methods

`guillemot-bench export --split test --out exports/opxrd-test` writes the exact task
bundles guillemot gets: the pattern as `.xy`, the starting CIFs, and a `task.json`
with the wavelengths. Give these to rietx or any other method.

Every method is scored by `guillemot.benchmark.scoring.score` from the same two
artefacts:

* a calculated pattern: columns `2θ yobs ycalc`, as TOPAS's `Out_X_Yobs_Ycalc` writes;
* refined CIFs, for the lattice.

Rwp is **recomputed on the original opXRD data** with 1/yobs weights, as TOPAS defines
it, rather than taken from each program's report. Background-subtracted patterns that
go to zero or below fall back to unit weights, and say so. A refinement that models
less than 90% of the measured 2θ range is not counted, so trimming awkward regions
does not pay. Lattice errors are measured against the labelled cell, in the setting
of the starting CIF, as relative volume error and the worst relative error in a, b
and c.
