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
    generated_slabs = {
        label: node for label, node in generated.items() if label.startswith('slab_')
    }
    generated_vacuum_spacings = {
        label: node.value for label, node in generated.items() if label.startswith('vacuum_spacing_')
    }
    assert len(generated_slabs) == 2
    assert generated_vacuum_spacings == {
        'vacuum_spacing_0_500000': 0.5,
        'vacuum_spacing_1_000000': 1.0,
    }

    legacy_layers = group_by_layers(legacy_conventional)

    for vacuum_spacing in [0.5, 1.0]:
        legacy_cleavaged = build_atoms_surface(
            legacy_conventional,
            4,
            legacy_layers,
            vacuum_spacing=vacuum_spacing,
        )
        key = f"slab_{vacuum_spacing:.6f}".replace('.', '_')
        assert key in generated_slabs
        _assert_atoms_match(generated_slabs[key].get_ase(), legacy_cleavaged)
