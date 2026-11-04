"""Reading the opXRD database (https://zenodo.org/records/15298026) for benchmarking.

The Zenodo archive unpacks to one JSON file per pattern, grouped by contributing
institution (`CNRS/pattern_0.json`, `LBNL/LBNL-A/pattern_group_0_0.json`, ...). An
entry is identified here by that relative path without the suffix — `CNRS/pattern_0` —
which is stable across machines and unambiguous, unlike a bare integer.

Three things about the raw files are worth knowing, because they would otherwise
quietly corrupt a refinement:

* `label` and `metadata` are JSON documents serialised *inside* strings, sometimes two
  levels deep (each atom of a basis is its own JSON string). `_decode` unpicks them.
* The Cu wavelengths are stored the wrong way round: `primary_wavelength` is 1.54439 Å
  (Kα2) and `secondary_wavelength` 1.54056 Å (Kα1) for most institutions. We report
  Kα1 as the shorter of the pair, which is always true of a Kα doublet.
* Structural labels are given as a P1-expanded list of every atom in the cell, usually
  with no space group. `reference_structure` rebuilds the symmetry, since a P1 model
  with hundreds of free sites is no fair starting point for anyone.
"""

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

DEFAULT_OPXRD_DIR = "datasets/opxrd/extracted"

# How much of a structure a label gives, from most to least.
TIER_STRUCTURE = "structure"  # every phase has a lattice and atomic positions
TIER_PARTIAL = "partial"  # some phase information: composition, space group or lattice
TIER_NONE = "none"  # no phase information at all

_LBNL_GROUP_RE = re.compile(r"pattern_group_(\d+)_\d+$")
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def opxrd_dir(path: str | os.PathLike | None = None) -> pathlib.Path:
    """Where the unpacked database lives: `path`, `GUILLEMOT_OPXRD_DIR`, or the default."""
    return pathlib.Path(path or os.getenv("GUILLEMOT_OPXRD_DIR", DEFAULT_OPXRD_DIR))


def _decode(value: Any) -> Any:
    """Undo opXRD's JSON-inside-strings, however many levels deep it goes."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return _decode(json.loads(stripped))
            except ValueError:
                return value
        return value
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def element_of(symbol: str) -> str | None:
    """'Ba0+' or 'O2-' to 'Ba' or 'O'; None for anything that is not an element symbol."""
    match = _ELEMENT_RE.match(symbol or "")
    return match.group(1) if match else None


@dataclass
class Phase:
    """One phase of a label, with whatever the contributor gave for it."""

    lattice: tuple[float, float, float, float, float, float] | None = None
    basis: list[dict] | None = None
    spacegroup: int | None = None
    composition: str | None = None
    phase_fraction: float | None = None

    @classmethod
    def from_label(cls, raw: dict) -> "Phase":
        lattice = raw.get("lattice")
        if isinstance(lattice, str):
            numbers = [float(x) for x in re.findall(r"[-+0-9.eE]+", lattice)]
            lattice = tuple(numbers) if len(numbers) == 6 else None
        elif isinstance(lattice, (list, tuple)) and len(lattice) == 6:
            lattice = tuple(float(x) for x in lattice)
        else:
            lattice = None

        spacegroup = raw.get("spacegroup")
        try:
            spacegroup = int(spacegroup) if spacegroup not in (None, "") else None
        except (TypeError, ValueError):
            spacegroup = None

        basis = raw.get("basis") or None
        return cls(
            lattice=lattice,
            basis=basis,
            spacegroup=spacegroup,
            composition=raw.get("chemical_composition") or None,
            phase_fraction=_as_float(raw.get("phase_fraction")),
        )

    @property
    def has_structure(self) -> bool:
        return self.lattice is not None and bool(self.basis)

    def element_amounts(self) -> dict[str, float]:
        """Element to amount, from the atoms of the basis or failing that the composition."""
        amounts: dict[str, float] = {}
        if self.basis:
            for atom in self.basis:
                element = element_of(atom.get("symbol", ""))
                if element:
                    occupancy = _as_float(atom.get("occupancy"))
                    amounts[element] = amounts.get(element, 0.0) + (
                        1.0 if occupancy is None else occupancy
                    )
            return amounts
        if self.composition:
            from pymatgen.core import Composition

            try:
                return {
                    str(el): amt
                    for el, amt in Composition(self.composition).as_dict().items()
                }
            except Exception:
                return {}
        return {}

    def reduced_formula(self) -> str | None:
        amounts = self.element_amounts()
        if not amounts:
            return None
        from pymatgen.core import Composition

        try:
            return Composition(amounts).reduced_formula
        except Exception:
            return None


# Radiation, as far as it can be inferred: opXRD does not record it, and a few dozen of
# the labelled CNRS patterns are evidently neutron data (single wavelengths of 2.08,
# 2.40, 2.44 Å, measured out to 2θ = 166°), which no X-ray refinement can fit.
RADIATION_LAB_DOUBLET = "xray_lab_doublet"
RADIATION_SYNCHROTRON = "xray_synchrotron"
RADIATION_NEUTRON = "neutron_likely"
RADIATION_SINGLE = "single_wavelength"
RADIATION_UNKNOWN = "unknown"


def guess_radiation(
    kalpha1: float | None, kalpha2: float | None, tth_max: float | None
) -> str:
    """Our best guess at the radiation, from the wavelengths and the angular range."""
    if not kalpha1:
        return RADIATION_UNKNOWN
    if kalpha2:
        return RADIATION_LAB_DOUBLET
    if kalpha1 < 1.2:
        return RADIATION_SYNCHROTRON
    # No X-ray tube in use emits above Cr Kα (2.29 Å), and Bragg-Brentano lab
    # instruments rarely go past 2θ = 140°; constant-wavelength neutron
    # diffractometers do both routinely.
    if kalpha1 > 2.0 or (tth_max or 0) > 140:
        return RADIATION_NEUTRON
    return RADIATION_SINGLE


@dataclass
class OpxrdEntry:
    """One pattern from opXRD, with its label decoded and its wavelengths put right."""

    entry_id: str
    institution: str
    two_theta: list[float]
    intensities: list[float]
    phases: list[Phase] = field(default_factory=list)
    kalpha1: float | None = None
    kalpha2: float | None = None
    is_simulated: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def label_tier(self) -> str:
        if self.phases and all(p.has_structure for p in self.phases):
            return TIER_STRUCTURE
        if any(
            p.lattice or p.basis or p.spacegroup or p.composition for p in self.phases
        ):
            return TIER_PARTIAL
        return TIER_NONE

    @property
    def elements(self) -> list[str]:
        found: set[str] = set()
        for phase in self.phases:
            found.update(phase.element_amounts())
        return sorted(found)

    @property
    def formulas(self) -> list[str]:
        return sorted({f for p in self.phases if (f := p.reduced_formula())})

    @property
    def group_key(self) -> str:
        """What must not be split across train and test.

        Patterns of the same compound — the same set of phases — go together, since
        refining one is a large hint for refining another. Unlabelled patterns fall
        back to their measurement series where the filename gives one (LBNL's
        `pattern_group_<n>_<m>` are in situ series), and to themselves otherwise.
        """
        if self.formulas:
            return "compound:" + "+".join(self.formulas)
        match = _LBNL_GROUP_RE.search(self.entry_id)
        if match:
            return f"series:{self.entry_id.rsplit('/', 1)[0]}/{match.group(1)}"
        return f"entry:{self.entry_id}"

    @property
    def source(self) -> str:
        """'CNRS/pattern_0' to 'CNRS'; 'LBNL/LBNL-A/...' to 'LBNL-A'."""
        parts = self.entry_id.split("/")
        return parts[-2] if len(parts) > 1 else ""

    def xy_lines(self) -> Iterator[str]:
        for x, y in zip(self.two_theta, self.intensities):
            yield f"{x:.6f} {y:.6f}"

    def write_xy(self, path: str | os.PathLike) -> pathlib.Path:
        """Write the pattern as two-column 2θ / intensity text, as TOPAS reads it."""
        path = pathlib.Path(path)
        path.write_text("\n".join(self.xy_lines()) + "\n", encoding="utf-8")
        return path

    def index_row(self) -> dict:
        """The summary of this entry kept in the index: everything but the arrays."""
        tth = self.two_theta
        steps = [b - a for a, b in zip(tth, tth[1:])]
        ys = self.intensities
        return {
            "entry_id": self.entry_id,
            "source": self.source,
            "institution": self.institution,
            "n_points": len(tth),
            "tth_min": round(min(tth), 4) if tth else None,
            "tth_max": round(max(tth), 4) if tth else None,
            "tth_step": round(sorted(steps)[len(steps) // 2], 5) if steps else None,
            "y_min": round(min(ys), 4) if ys else None,
            "y_max": round(max(ys), 4) if ys else None,
            "kalpha1": self.kalpha1,
            "kalpha2": self.kalpha2,
            "is_simulated": self.is_simulated,
            "radiation": guess_radiation(
                self.kalpha1, self.kalpha2, max(tth) if tth else None
            ),
            "label_tier": self.label_tier,
            "n_phases": len(self.phases),
            "formulas": "+".join(self.formulas),
            "elements": ",".join(self.elements),
            "spacegroups": ",".join(
                str(p.spacegroup) for p in self.phases if p.spacegroup
            ),
            "group_key": self.group_key,
        }


def _wavelengths(xray_info: dict | None) -> tuple[float | None, float | None]:
    """(Kα1, Kα2), correcting opXRD's swapped primary/secondary wavelengths.

    Within a Kα doublet Kα1 is always the shorter wavelength. A lone wavelength
    (synchrotron, or monochromated) is returned as Kα1 with no Kα2.
    """
    info = xray_info or {}
    values = [
        w
        for w in (
            _as_float(info.get("primary_wavelength")),
            _as_float(info.get("secondary_wavelength")),
        )
        if w
    ]
    if not values:
        return None, None
    if len(values) == 1 or abs(values[0] - values[1]) < 1e-6:
        return values[0], None
    return min(values), max(values)


def entry_path(entry_id: str, root: str | os.PathLike | None = None) -> pathlib.Path:
    return opxrd_dir(root) / f"{entry_id}.json"


def parse_entry(raw: dict, entry_id: str) -> OpxrdEntry:
    label = _decode(raw.get("label")) or {}
    metadata = _decode(raw.get("metadata")) or {}
    kalpha1, kalpha2 = _wavelengths(label.get("xray_info"))
    return OpxrdEntry(
        entry_id=entry_id,
        institution=metadata.get("institution") or "",
        two_theta=[float(x) for x in raw.get("two_theta_values", [])],
        intensities=[float(y) for y in raw.get("intensities", [])],
        phases=[Phase.from_label(p) for p in label.get("phases") or []],
        kalpha1=kalpha1,
        kalpha2=kalpha2,
        is_simulated=bool(label.get("is_simulated")),
        metadata=metadata,
    )


def load_entry(entry_id: str, root: str | os.PathLike | None = None) -> OpxrdEntry:
    """Load one opXRD entry by its id, e.g. 'CNRS/pattern_0'."""
    path = entry_path(entry_id, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No opXRD entry {entry_id!r} at {path}. Is GUILLEMOT_OPXRD_DIR pointing at "
            "the unpacked Zenodo archive?"
        )
    with open(path, encoding="utf-8") as handle:
        return parse_entry(json.load(handle), entry_id)


def iter_entry_ids(root: str | os.PathLike | None = None) -> Iterator[str]:
    base = opxrd_dir(root)
    for path in sorted(base.rglob("*.json")):
        yield path.relative_to(base).with_suffix("").as_posix()


def reference_structure(phase: Phase, symprec: float = 0.01, merge_tol: float = 0.1):
    """The labelled structure of `phase` as a pymatgen Structure, with symmetry restored.

    Atoms within `merge_tol` Å of each other are merged into one site first: labels
    list a disordered site once per species, and some list the same atom twice.
    Occupancies are summed per element and the total capped at 1, which gets both of
    those right (and a split site's two halves too). Symmetry is then found at
    `symprec`; if it cannot be, the P1 cell is returned rather than nothing.
    """
    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if not phase.has_structure:
        raise ValueError(
            "This phase has no lattice and basis to build a structure from"
        )

    elements, occupancies, coords = [], [], []
    for atom in phase.basis:
        element = element_of(atom.get("symbol", ""))
        if not element:
            continue
        occupancy = _as_float(atom.get("occupancy"))
        elements.append(element)
        occupancies.append(1.0 if occupancy is None else occupancy)
        coords.append([float(atom["x"]), float(atom["y"]), float(atom["z"])])

    lattice = Lattice.from_parameters(*phase.lattice)
    coords = np.array(coords) % 1.0
    if len(coords) > 1:
        distances = lattice.get_all_distances(coords, coords)
        np.fill_diagonal(distances, 0)
        clusters = fcluster(
            linkage(squareform((distances + distances.T) / 2), "single"),
            merge_tol,
            "distance",
        )
    else:
        clusters = np.ones(len(coords), dtype=int)

    species, sites = [], []
    for label in np.unique(clusters):
        members = np.flatnonzero(clusters == label)
        amounts: dict[str, float] = {}
        for i in members:
            amounts[elements[i]] = amounts.get(elements[i], 0.0) + occupancies[i]
        total = sum(amounts.values())
        if total > 1:
            amounts = {el: amt / total for el, amt in amounts.items()}
        species.append(amounts)
        sites.append(coords[members[0]])

    structure = Structure(lattice, species, sites)
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_refined_structure()
    except Exception:
        return structure
