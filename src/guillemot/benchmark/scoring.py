"""Scoring a refinement against an opXRD label, independently of whoever did it.

Every method being compared — guillemot, rietx, an autodiff refiner — is scored by the
same code from the same two kinds of artefact:

* a calculated pattern: whitespace-separated columns `2θ  yobs  ycalc`, which is what
  TOPAS's `Out_X_Yobs_Ycalc` writes and what any other method can trivially export;
* zero or more refined structures as CIF, for the lattice comparison.

R-factors are recomputed here on the *original* opXRD data rather than taken from each
program's own report. Programs differ in their weighting, in background handling, and
in which points they include, and a refinement restricted to a convenient window of
2θ would otherwise look better than an honest one. The fraction of the measured range
that the calculated pattern covers is reported alongside, for the same reason.
"""

import math
import pathlib
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from guillemot.benchmark.opxrd import OpxrdEntry, reference_structure

# A refinement must model at least this much of the measured 2θ range to count.
MIN_COVERAGE = 0.9

_TOPAS_STAT_RE = {
    name: re.compile(rf"\b{name}\s+(-?[0-9.]+(?:[eE][-+]?\d+)?)")
    for name in ("r_wp", "r_exp", "r_p", "gof")
}
_CIF_CELL_RE = {
    key: re.compile(rf"^\s*_cell_{name}\s+(-?[0-9.]+)", re.MULTILINE)
    for key, name in (
        ("a", "length_a"),
        ("b", "length_b"),
        ("c", "length_c"),
        ("alpha", "angle_alpha"),
        ("beta", "angle_beta"),
        ("gamma", "angle_gamma"),
    )
}


@dataclass
class Lattice:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @property
    def volume(self) -> float:
        ca, cb, cg = (
            math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma)
        )
        return (
            self.a
            * self.b
            * self.c
            * math.sqrt(max(0.0, 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg))
        )


@dataclass
class PhaseScore:
    reference_formula: str | None
    refined_cif: str | None
    reference_lattice: dict | None
    refined_lattice: dict | None
    volume_rel_error: float | None = None
    max_length_rel_error: float | None = None
    max_angle_abs_error: float | None = None


@dataclass
class Score:
    entry_id: str
    calculated_pattern: str | None = None
    # Recomputed on the original data, over the calculated pattern's range.
    rwp: float | None = None
    rp: float | None = None
    weighting: str | None = None
    coverage: float | None = None
    n_points_scored: int = 0
    # What the refinement program reported about itself, where it said.
    reported: dict = field(default_factory=dict)
    phases: list[PhaseScore] = field(default_factory=list)
    valid: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def read_calculated_pattern(path: str | pathlib.Path) -> np.ndarray:
    """Columns 2θ, yobs, ycalc from a TOPAS `Out_X_Yobs_Ycalc` file or similar."""
    data = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    return data[np.argsort(data[:, 0])]


def r_factors(yobs: np.ndarray, ycalc: np.ndarray) -> tuple[float, float, str]:
    """(Rwp, Rp, weighting), in percent, as TOPAS reports them.

    Weights are 1/yobs — counting statistics — which also makes Rwp independent of
    the intensity scale, as it must be for opXRD's mixture of counts and normalised
    data. A few opXRD patterns were background-subtracted and dip to zero or below,
    where Poisson weights are meaningless; those are scored with unit weights instead,
    and say so.
    """
    if np.all(yobs > 0):
        weights, weighting = 1.0 / yobs, "poisson"
    else:
        weights, weighting = np.ones_like(yobs), "unit"
    rwp = math.sqrt(np.sum(weights * (yobs - ycalc) ** 2) / np.sum(weights * yobs**2))
    rp = np.sum(np.abs(yobs - ycalc)) / np.sum(np.abs(yobs))
    return 100 * rwp, 100 * rp, weighting


def read_reported_statistics(out_path: str | pathlib.Path) -> dict:
    """r_wp, r_exp, r_p and gof as TOPAS wrote them into a .out file, where present."""
    try:
        text = pathlib.Path(out_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    found = {}
    for name, pattern in _TOPAS_STAT_RE.items():
        match = pattern.search(text)
        if match:
            found[name] = float(match.group(1))
    return found


def read_cif_lattice(path: str | pathlib.Path) -> Lattice | None:
    """The cell of a CIF, read leniently: TOPAS writes esds, e.g. `5.4312(3)`."""
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = {}
    for key, pattern in _CIF_CELL_RE.items():
        match = pattern.search(text)
        if not match:
            return None
        values[key] = float(match.group(1))
    return Lattice(**values)


def reference_lattices(entry: OpxrdEntry) -> list[tuple[str | None, Lattice]]:
    """Each labelled phase's cell, in the setting of the starting CIF we hand out."""
    out = []
    for phase in entry.phases:
        if not phase.has_structure:
            continue
        lattice = reference_structure(phase).lattice
        out.append(
            (
                phase.reduced_formula(),
                Lattice(*lattice.abc, *lattice.angles),
            )
        )
    return out


def _compare(reference: Lattice, refined: Lattice) -> dict:
    lengths = [
        abs(getattr(refined, k) - getattr(reference, k)) / getattr(reference, k)
        for k in ("a", "b", "c")
    ]
    angles = [
        abs(getattr(refined, k) - getattr(reference, k))
        for k in ("alpha", "beta", "gamma")
    ]
    return {
        "volume_rel_error": abs(refined.volume - reference.volume) / reference.volume,
        "max_length_rel_error": max(lengths),
        "max_angle_abs_error": max(angles),
    }


def score_phases(entry: OpxrdEntry, cifs: list[pathlib.Path]) -> list[PhaseScore]:
    """Pair each labelled phase with the refined CIF closest to it in volume.

    Volume is the setting-independent quantity, so a refinement that permuted the
    axes still finds its partner; the per-axis errors that follow assume the setting
    of the starting CIF was kept, which is what a refinement normally does.
    """
    refined = [(cif, lat) for cif in cifs if (lat := read_cif_lattice(cif))]
    scores = []
    unused = list(refined)
    for formula, reference in reference_lattices(entry):
        if not unused:
            scores.append(PhaseScore(formula, None, asdict(reference), None))
            continue
        cif, lattice = min(
            unused, key=lambda pair: abs(pair[1].volume - reference.volume)
        )
        unused.remove((cif, lattice))
        scores.append(
            PhaseScore(
                reference_formula=formula,
                refined_cif=str(cif),
                reference_lattice=asdict(reference),
                refined_lattice=asdict(lattice),
                **_compare(reference, lattice),
            )
        )
    return scores


def score(
    entry: OpxrdEntry,
    calculated_pattern: str | pathlib.Path | None,
    cifs: list[pathlib.Path] | None = None,
    out_file: str | pathlib.Path | None = None,
) -> Score:
    """Score one refinement of `entry`. Any argument may be missing; the score says so."""
    result = Score(entry_id=entry.entry_id)
    if out_file:
        result.reported = read_reported_statistics(out_file)

    if calculated_pattern is None or not pathlib.Path(calculated_pattern).is_file():
        result.notes.append("no calculated pattern")
        result.phases = score_phases(entry, cifs or [])
        return result
    result.calculated_pattern = str(calculated_pattern)

    calculated = read_calculated_pattern(calculated_pattern)
    x_obs = np.asarray(entry.two_theta)
    y_obs = np.asarray(entry.intensities)
    lo, hi = calculated[0, 0], calculated[-1, 0]
    inside = (x_obs >= lo) & (x_obs <= hi)

    # Interpolate the model onto the measured points; scale by a single factor fitted
    # on the overlap, in case the method rescaled yobs (TOPAS does not, others may).
    y_calc = np.interp(x_obs[inside], calculated[:, 0], calculated[:, 2])
    y_ref = y_obs[inside]
    y_calc_obs = np.interp(x_obs[inside], calculated[:, 0], calculated[:, 1])
    if np.any(y_calc_obs) and np.any(y_ref):
        ratio = np.dot(y_ref, y_calc_obs) / np.dot(y_calc_obs, y_calc_obs)
        if not 0.999 < ratio < 1.001:
            result.notes.append(
                f"calculated pattern rescaled by {ratio:.4g} onto opXRD"
            )
        y_calc = y_calc * ratio

    span = x_obs.max() - x_obs.min()
    result.coverage = (
        float((min(hi, x_obs.max()) - max(lo, x_obs.min())) / span) if span else 0.0
    )
    result.n_points_scored = int(inside.sum())
    if result.n_points_scored > 10:
        result.rwp, result.rp, result.weighting = r_factors(y_ref, y_calc)
    result.phases = score_phases(entry, cifs or [])
    result.valid = (
        result.rwp is not None
        and result.coverage is not None
        and result.coverage >= MIN_COVERAGE
    )
    if result.coverage is not None and result.coverage < MIN_COVERAGE:
        result.notes.append(
            f"only {result.coverage:.0%} of the measured range was refined; not counted"
        )
    return result
