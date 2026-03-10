"""Regression tests for ``CleavagedStructureData``."""

from __future__ import annotations

import numpy.testing as npt
from aiida import orm
from ase import Atoms
from ase.build import bulk

from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.tools.structure import get_cleavaged_structure, get_conventional_structure


def _assert_atoms_match(left: Atoms, right: Atoms, atol: float = 1.0e-10) -> None:
    """Assert that two ASE structures are identical for regression purposes."""
    assert left.get_chemical_symbols() == right.get_chemical_symbols()
    assert tuple(left.pbc) == tuple(right.pbc)
    npt.assert_allclose(left.cell.array, right.cell.array, atol=atol)
    npt.assert_allclose(left.get_positions(), right.get_positions(), atol=atol)


def test_cleavaged_structure_data_round_trips_attributes(aiida_profile_clean) -> None:
    """Stored ``CleavagedStructureData`` should preserve structure and metadata."""
    aluminum_fcc = bulk('Al', 'fcc', a=4.05)
    node = CleavagedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111')

    assert node.strukturbericht == 'A1'
    assert node.gliding_plane == '111'
    assert node.n_unit_cells == 4
    _assert_atoms_match(node.unit_cell, aluminum_fcc)
    _assert_atoms_match(node.get_ase(), aluminum_fcc)
    _assert_atoms_match(node.structure_data.get_ase(), aluminum_fcc)

    node.store()
    loaded = orm.load_node(node.pk)

    assert loaded.strukturbericht == 'A1'
    assert loaded.gliding_plane == '111'
    assert loaded.n_unit_cells == 4
    _assert_atoms_match(loaded.unit_cell, aluminum_fcc)


def test_cleavaged_structure_data_supercells_match_legacy_builders(aiida_profile_clean) -> None:
    """Conventional and cleavaged supercells should match the legacy builder outputs."""
    aluminum_fcc = bulk('Al', 'fcc', a=4.05)
    node = CleavagedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111').store()

    legacy_strukturbericht, legacy_conventional = get_conventional_structure(
        aluminum_fcc,
        gliding_plane='111',
    )
    _, legacy_cleavaged = get_cleavaged_structure(
        legacy_conventional,
        gliding_plane='111',
        n_unit_cells=4,
    )

    assert node.strukturbericht == legacy_strukturbericht
    _assert_atoms_match(node.get_conventional_structure(), legacy_conventional)
    _assert_atoms_match(node.get_cleavaged_structure(), legacy_cleavaged)
