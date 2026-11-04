"""Turning a CIF into the TOPAS `str` block for it, deterministically.

Writing site lines by hand is where the agent goes wrong most: symmetry constraints on
the lattice left out, special positions written as decimals, made-up keywords for the
displacement parameters. John Evans's `cif_to_str` (vendored, see
`guillemot/vendor/topas_inp_writer/UPSTREAM.md`) does it properly — Wyckoff-aware
constraints, exact fractions, Uiso/U_aniso converted to beq — so the agent should call
it rather than transcribe.
"""

import pathlib

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from guillemot.tools.files import READABLE_DIRS, _readable_roots
from guillemot.vendor.topas_inp_writer.cif_to_str import convert


class TopasStructure(BaseModel):
    cif_path: str
    str_block: str = Field(
        description="A complete `str` block: paste it into your .inp under `xdd`, then "
        "add `scale`, peak shape (e.g. CS_L), `Out_CIF_STR` and anything else you need."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Problems found while converting; check each before refining.",
    )


def cif_to_topas_str(cif_path: str, refine_lattice: bool = True) -> TopasStructure:
    """Convert a CIF into a ready-to-use TOPAS `str` block.

    Use this whenever you have a CIF — a starting model you were given, or one you
    saved — instead of writing the lattice and site lines yourself. The block keeps the
    space group's constraints: lattice parameters tied by symmetry (e.g. `b = Get(a);`
    for cubic), atoms on special positions with exact fractions, and one refinable
    `beq` per site (converted from the CIF's Uiso or anisotropic U where present).

    The block contains the lattice, space group and sites only. Add `scale @ 0.001`,
    peak shape such as `CS_L(@, 100)`, and `Out_CIF_STR(...)` yourself, inside the
    `str` (after the sites).

    Args:
        cif_path: path of the CIF, in this session's directory or `cifs/`, `data/` or
            `examples/`.
        refine_lattice: put `@` on the lattice parameters the crystal system leaves
            free (default). Parameters fixed or tied by symmetry are never refined.
    """
    path = pathlib.Path(cif_path)
    if not any(path.resolve().is_relative_to(root) for root in _readable_roots()):
        raise ModelRetry(
            f"{cif_path} is outside the files you can use: this session's directory, "
            f"{', '.join(f'{d}/' for d in READABLE_DIRS)}."
        )
    if not path.is_file():
        raise ModelRetry(
            f"No CIF at {cif_path}. Call `list_available_data` to see what exists."
        )
    try:
        block, warnings = convert(path, refine=refine_lattice)
    except Exception as exc:
        raise ModelRetry(
            f"Could not convert {cif_path}: {type(exc).__name__}: {exc}. Check it is a "
            "CIF with a cell, a space group or symmetry operators, and atom sites."
        ) from exc
    return TopasStructure(
        cif_path=cif_path, str_block=block.strip(), warnings=list(warnings)
    )


# --------------------------------------------------------------------------------------
# describe_structure: the facts about a structure that a model should never derive
# --------------------------------------------------------------------------------------

DESCRIBE_SYMPREC = 0.01
DESCRIBE_TOP_REFLECTIONS = 12


class SiteDescription(BaseModel):
    label: str
    species: str
    wyckoff: str
    site_symmetry: str
    position: str
    free_coordinates: str


class StructureDescription(BaseModel):
    cif_path: str
    summary: str = Field(description="Everything below, as text: read this first.")
    space_group: str
    crystal_system: str
    formula: str
    z: float
    sites: list[SiteDescription]
    reflections: list[str]


def _fraction(value: float, tol: float = 1e-4) -> str:
    """0.25 -> '1/4', 0.3333 -> '1/3', 0.0912 -> '0.0912'."""
    from fractions import Fraction

    value = value % 1.0
    if min(value, 1 - value) < tol:
        return "0"
    frac = Fraction(value).limit_denominator(24)
    return str(frac) if abs(float(frac) - value) < tol else f"{value:.4f}"


def _free_directions(
    position, rotations, translations, tol: float = 1e-3
) -> list[list[int]]:
    """Directions in which a site can move without breaking symmetry.

    The stabiliser of a site is the set of operations that map it onto itself; the
    site may move only within the subspace every one of those operations leaves
    fixed, i.e. the common null space of (R - I).
    """
    import numpy as np

    rows = []
    for rotation, translation in zip(rotations, translations):
        moved = rotation @ position + translation
        delta = moved - position
        if np.all(np.abs(delta - np.round(delta)) < tol):
            rows.append(rotation - np.eye(3))
    if not rows:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    _, s, vt = np.linalg.svd(np.vstack(rows))
    rank = int(np.sum(s > 1e-6))
    basis = vt[rank:]
    directions = []
    for v in basis:
        v = v / np.max(np.abs(v))
        directions.append(
            [
                int(round(c)) if abs(c - round(c)) < 1e-6 else round(float(c), 3)
                for c in v
            ]
        )
    return directions


def _describe_free(directions: list[list[float]]) -> str:
    """Name the coordinates a site may refine: 'y', 'x and z', or a tied form like (x, -x, x)."""
    if not directions:
        return "none (fixed by symmetry)"
    if len(directions) == 3:
        return "x, y and z"
    axes = "xyz"
    if all(sum(1 for c in d if c != 0) == 1 for d in directions):
        free = sorted(
            axes[next(i for i, c in enumerate(d) if c != 0)] for d in directions
        )
        return " and ".join(free) + " (the rest fixed)"
    if len(directions) == 1:
        d = directions[0]
        pivot = axes[next(i for i, c in enumerate(d) if c != 0)]
        terms = []
        for component in d:
            if component == 0:
                terms.append("fixed")
            elif component == 1:
                terms.append(pivot)
            elif component == -1:
                terms.append(f"-{pivot}")
            else:
                terms.append(f"{component}{pivot}")
        return f"one parameter, tied as ({', '.join(terms)})"
    return f"{len(directions)} tied parameters, along " + ", ".join(
        "(" + ", ".join(f"{c:g}" for c in d) + ")" for d in directions
    )


def _species_text(site) -> str:
    return ", ".join(
        f"{el}{'' if abs(occ - 1) < 1e-6 else f' {occ:.3g}'}"
        for el, occ in site.species.items()
    )


def describe_structure(
    cif_path: str,
    wavelength: float = 1.5406,
    two_theta_max: float = 90.0,
) -> StructureDescription:
    """Describe a crystal structure: symmetry, sites, composition and expected peaks.

    Use this instead of working anything out by hand from a CIF. It reports, from the
    structure itself:

    - space group, crystal system, formula and Z (formula units per cell), density;
    - every symmetry-distinct site: its Wyckoff position (multiplicity and letter), site
      symmetry, exact coordinates, and which coordinates are free to refine;
    - a check that the cell contents add up;
    - the strongest expected reflections (hkl, d, 2θ at `wavelength`, relative
      intensity), to compare with the peaks in your pattern;

    TOPAS works out multiplicities itself (`num_posns` is filled in on refinement), so
    these are for your understanding; to write the `str` block, use `cif_to_topas_str`.

    Args:
        cif_path: path of the CIF, in this session's directory or `cifs/`, `data/` or
            `examples/`.
        wavelength: in Å, for the expected reflections; the Kα1 wavelength for a lab
            Cu source (1.5406).
        two_theta_max: the highest 2θ to list reflections up to, in degrees.
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    path = pathlib.Path(cif_path)
    if not any(path.resolve().is_relative_to(root) for root in _readable_roots()):
        raise ModelRetry(
            f"{cif_path} is outside the files you can use: this session's directory, "
            f"{', '.join(f'{d}/' for d in READABLE_DIRS)}."
        )
    if not path.is_file():
        raise ModelRetry(
            f"No CIF at {cif_path}. Call `list_available_data` to see what exists."
        )
    try:
        structure = CifParser(path, occupancy_tolerance=1.05).parse_structures(
            primitive=False
        )[0]
        sga = SpacegroupAnalyzer(structure, symprec=DESCRIBE_SYMPREC)
        dataset = sga.get_symmetry_dataset()
    except Exception as exc:
        raise ModelRetry(
            f"Could not read the structure in {cif_path}: {type(exc).__name__}: {exc}"
        ) from exc

    composition = structure.composition
    reduced, factor = composition.get_reduced_composition_and_factor()
    multiplicity = {
        i: int((dataset.equivalent_atoms == i).sum())
        for i in set(dataset.equivalent_atoms)
    }

    sites: list[SiteDescription] = []
    for index in sorted(multiplicity):
        site = structure[int(index)]
        directions = _free_directions(
            site.frac_coords, dataset.rotations, dataset.translations
        )
        sites.append(
            SiteDescription(
                label=site.label or f"site{index}",
                species=_species_text(site),
                wyckoff=f"{multiplicity[index]}{dataset.wyckoffs[index]}",
                site_symmetry=dataset.site_symmetry_symbols[index],
                position="(" + ", ".join(_fraction(c) for c in site.frac_coords) + ")",
                free_coordinates=_describe_free(directions),
            )
        )

    pattern = XRDCalculator(wavelength=wavelength).get_pattern(
        structure, two_theta_range=(0, two_theta_max)
    )
    strongest = sorted(range(len(pattern.x)), key=lambda i: -pattern.y[i])[
        :DESCRIBE_TOP_REFLECTIONS
    ]
    reflections = []
    for i in sorted(strongest, key=lambda i: pattern.x[i]):
        hkl = pattern.hkls[i][0]["hkl"]
        reflections.append(
            f"({' '.join(str(h) for h in hkl)}) d={pattern.d_hkls[i]:.4f} Å  2θ={pattern.x[i]:.3f}°  I={pattern.y[i]:.0f}"
        )

    lattice = structure.lattice
    atoms_per_cell = sum(occ for site in structure for occ in site.species.values())
    lines = [
        f"{reduced.reduced_formula}: space group {sga.get_space_group_symbol()} (No. {sga.get_space_group_number()}), {sga.get_crystal_system()}.",
        f"Cell: a={lattice.a:.5f} b={lattice.b:.5f} c={lattice.c:.5f} Å, α={lattice.alpha:.3f} β={lattice.beta:.3f} γ={lattice.gamma:.3f}°, V={lattice.volume:.3f} Å³.",
        f"Contents: {composition.formula} per cell = Z {factor:g} × {reduced.formula}; {atoms_per_cell:.4g} atoms per cell (occupancy-weighted); density {structure.density:.3f} g/cm³.",
        "Cell contents by site: "
        + " + ".join(
            f"{s.wyckoff.rstrip('abcdefghijklmnopqrstuvwxyzA')} × "
            + (f"({s.species})" if "," in s.species else s.species)
            for s in sites
        )
        + ".",
        "Sites (one line per symmetry-distinct site; multiplicity × occupancy gives the cell contents):",
        *(
            f"  {s.label}: {s.species} on {s.wyckoff} (site symmetry {s.site_symmetry}) at {s.position}; free coordinates: {s.free_coordinates}"
            for s in sites
        ),
        f"Strongest reflections up to 2θ={two_theta_max:g}° at λ={wavelength} Å:",
        *(f"  {r}" for r in reflections),
    ]
    return StructureDescription(
        cif_path=cif_path,
        summary="\n".join(lines),
        space_group=f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()})",
        crystal_system=sga.get_crystal_system(),
        formula=reduced.reduced_formula,
        z=factor,
        sites=sites,
        reflections=reflections,
    )
