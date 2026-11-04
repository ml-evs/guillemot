# This file was edited with the assistance of an AI model and requires human review from the contributor.
"""cif_to_topas_str: the vendored converter, as the agent sees it."""

import pytest
from pydantic_ai import ModelRetry

from guillemot.benchmark import runner
from guillemot.benchmark.opxrd import parse_entry
from guillemot.session import start_session
from guillemot.tools.structures import cif_to_topas_str

from test_benchmark import NACL, raw_entry  # noqa: E402


def test_benchmark_starting_cif_becomes_a_constrained_str_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = start_session("bench/run/entry")
    entry = parse_entry(raw_entry([NACL]), "TEST/pattern_0")
    task = runner.prepare(entry, session.directory, "structure")

    result = cif_to_topas_str(str(session.directory / task["starting_cifs"][0]))
    block = result.str_block
    assert block.startswith("str")
    assert "b = Get(a);" in block and "c = Get(a);" in block  # cubic tie
    assert "Fm-3m" in block
    assert block.count("site ") == 2  # the asymmetric unit, not all 8 atoms
    assert "beq @" in block
    assert result.warnings == []


def test_refuses_cifs_outside_its_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start_session("bench/run/entry")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "answer.cif").write_text("data_x\n")
    with pytest.raises(ModelRetry, match="outside the files you can use"):
        cif_to_topas_str("datasets/answer.cif")


def test_describe_structure_reports_symmetry_the_model_should_not_derive(
    tmp_path, monkeypatch
):
    """Zinc blende with a mixed cation site: the case a model once spent an hour on."""
    from pymatgen.core import Lattice, Structure

    from guillemot.tools.structures import describe_structure

    monkeypatch.chdir(tmp_path)
    session = start_session("bench/run/entry")
    structure = Structure.from_spacegroup(
        "F-43m",
        Lattice.cubic(6.187),
        [{"Zn": 1 / 3, "In": 1 / 3, "Cu": 1 / 3}, "Te"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    cif = session.directory / "znincute.cif"
    structure.to(filename=str(cif))

    result = describe_structure(str(cif), wavelength=1.5406, two_theta_max=60)
    assert result.space_group == "F-43m (216)"
    wyckoffs = {s.wyckoff for s in result.sites}
    assert wyckoffs == {"4a", "4c"}
    assert all(s.free_coordinates.startswith("none") for s in result.sites)
    assert "4 × (Zn 0.333, In 0.333, Cu 0.333) + 4 × Te" in result.summary
    assert result.reflections[0].startswith("(1 1 1)")


def test_describe_structure_names_free_coordinates(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure

    from guillemot.tools.structures import describe_structure

    monkeypatch.chdir(tmp_path)
    session = start_session("bench/run/entry")
    spinel = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(8.24), ["Li", "Mn", "O"],
        [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0.263, 0.263, 0.263]],
    )  # fmt: skip
    cif = session.directory / "spinel.cif"
    spinel.to(filename=str(cif))
    by_species = {s.species: s for s in describe_structure(str(cif)).sites}
    assert by_species["O"].free_coordinates.startswith("one parameter, tied as")
    assert by_species["Li"].free_coordinates.startswith("none")
