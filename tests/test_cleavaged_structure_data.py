"""Regression tests for ``CleavagedStructureData`` and slab generation."""

from __future__ import annotations

import numpy.testing as npt
from aiida import orm
from ase import Atoms
from ase.build import bulk

from aiida_dislocation.calculations import generate_cleavaged_structures
from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.tools.structure_builder import build_atoms_surface
from aiida_dislocation.tools.structure_utils import group_by_layers
from aiida_dislocation.tools.structure import get_conventional_structure


def _assert_atoms_match(left: Atoms, right: Atoms, atol: float = 1.0e-10) -> None:
    """Assert that two ASE structures are identical for regression purposes."""
    assert left.get_chemical_symbols() == right.get_chemical_symbols()
    assert tuple(left.pbc) == tuple(right.pbc)
    npt.assert_allclose(left.cell.array, right.cell.array, atol=atol)
    npt.assert_allclose(left.get_positions(), right.get_positions(), atol=atol)


def test_cleavaged_structure_data_round_trips_attributes(aiida_profile_clean) -> None:
    """Stored ``CleavagedStructureData`` should preserve only configuration attributes."""
    node = CleavagedStructureData(n_unit_cells=4, gliding_plane='111', vacuum_spacings=[0.5, 1.0])

    assert node.n_unit_cells == 4
    assert node.gliding_plane == '111'
    assert node.vacuum_spacings == [0.5, 1.0]

    node.store()
    loaded = orm.load_node(node.pk)

    assert loaded.n_unit_cells == 4
    assert loaded.gliding_plane == '111'
    assert loaded.vacuum_spacings == [0.5, 1.0]


def test_generate_cleavaged_structures_match_legacy_builders(aiida_profile_clean) -> None:
    """Generated slabs should match the legacy cleavaged-structure builders."""
    aluminum_fcc = bulk('Al', 'fcc', a=4.05)
    structure = orm.StructureData(ase=aluminum_fcc)
    config = CleavagedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
        vacuum_spacings=[0.5, 1.0],
    ).store()

    generated = generate_cleavaged_structures(structure=structure, cleavaged_data=config)

    legacy_strukturbericht, legacy_conventional = get_conventional_structure(
        aluminum_fcc,
        gliding_plane='111',
    )
    assert legacy_strukturbericht == 'A1'

    _assert_atoms_match(generated['conventional_structure'].get_ase(), legacy_conventional)
    structure_map = generated['structure_map'].get_dict()
    assert len(structure_map) == 2

    legacy_layers = group_by_layers(legacy_conventional)

    for index, vacuum_spacing in enumerate([0.5, 1.0], start=1):
        legacy_cleavaged = build_atoms_surface(
            legacy_conventional,
            4,
            legacy_layers,
            vacuum_spacing=vacuum_spacing,
        )
        key = f'slab_idx_{index:03d}'
        assert key in generated
        assert key in structure_map
        _assert_atoms_match(generated[key].get_ase(), legacy_cleavaged)
        assert structure_map[key]['point_index'] == index
        assert structure_map[key]['vacuum_spacing'] == vacuum_spacing
        assert structure_map[key]['structure_uuid'] == generated[key].uuid
