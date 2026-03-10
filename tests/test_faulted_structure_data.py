"""Regression tests for ``FaultedStructureData`` and provenance-aware structure generation."""

from __future__ import annotations

import numpy.testing as npt
import pytest
from aiida import orm
from aiida.common.exceptions import ModificationNotAllowed
from ase import Atoms
from ase.build import bulk

from aiida_dislocation.calculations import generate_faulted_structures
from aiida_dislocation.data.faulted_structure import FaultedStructureData
from aiida_dislocation.tools.structure import (
    get_cleavaged_structure,
    get_conventional_structure,
    get_faulted_structure,
)


def _assert_atoms_match(left: Atoms, right: Atoms, atol: float = 1.0e-10) -> None:
    """Assert that two ASE structures are identical for regression purposes."""
    assert left.get_chemical_symbols() == right.get_chemical_symbols()
    assert tuple(left.pbc) == tuple(right.pbc)
    npt.assert_allclose(left.cell.array, right.cell.array, atol=atol)
    npt.assert_allclose(left.get_positions(), right.get_positions(), atol=atol)


@pytest.fixture
def aluminum_fcc() -> Atoms:
    """Return a simple FCC aluminum primitive cell."""
    return bulk('Al', 'fcc', a=4.05)


def test_faulted_structure_data_requires_unit_cell_repeat_count(aluminum_fcc) -> None:
    """Creating ``FaultedStructureData`` from an ASE structure requires ``n_unit_cells``."""
    with pytest.raises(ValueError, match='n_unit_cells'):
        FaultedStructureData(ase=aluminum_fcc)


def test_faulted_structure_data_auto_detects_metadata(aiida_profile_clean, aluminum_fcc) -> None:
    """``FaultedStructureData`` should infer the Strukturbericht and default gliding plane."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4)

    assert node.strukturbericht == 'A1'
    assert node.gliding_plane == '111'
    assert node.gliding_system.strukturbericht == 'A1'


def test_faulted_structure_data_round_trips_attributes(aiida_profile_clean, aluminum_fcc) -> None:
    """Stored ``FaultedStructureData`` should preserve structure and metadata."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111')

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


def test_faulted_structure_data_is_immutable_after_store(aiida_profile_clean, aluminum_fcc) -> None:
    """Node attributes should not be mutable after storing the AiiDA data node."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111').store()

    with pytest.raises(ModificationNotAllowed):
        node.set_ase(aluminum_fcc.copy())


def test_faulted_structure_data_supercells_match_legacy_builders(aiida_profile_clean, aluminum_fcc) -> None:
    """Conventional and cleavaged supercells should match the legacy builder outputs."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111').store()

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


def test_generate_faulted_structures_matches_legacy_general_fault_path(aiida_profile_clean, aluminum_fcc) -> None:
    """The calcfunction outputs should keep the same cells and atomic positions as the legacy path."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111').store()

    generated = generate_faulted_structures(
        faulted_data=node,
        fault_mode=orm.Str('general'),
        fault_type=orm.Str('general'),
    )

    _, legacy_conventional = get_conventional_structure(aluminum_fcc, gliding_plane='111')
    _, legacy_cleavaged = get_cleavaged_structure(
        legacy_conventional,
        gliding_plane='111',
        n_unit_cells=4,
    )
    _, legacy_faulted = get_faulted_structure(
        legacy_conventional,
        fault_mode='general',
        fault_type='general',
        gliding_plane='111',
        n_unit_cells=4,
    )

    assert legacy_faulted is not None

    legacy_entries = legacy_faulted['structures']
    structure_map = generated['structure_map'].get_dict()

    _assert_atoms_match(generated['conventional_structure'].get_ase(), legacy_conventional)
    _assert_atoms_match(generated['cleavaged_structure'].get_ase(), legacy_cleavaged)
    assert len(structure_map) == len(legacy_entries)

    for index, legacy_entry in enumerate(legacy_entries, start=1):
        key = f'sfe_idx_{index:03d}'
        assert key in generated
        assert key in structure_map

        generated_structure = generated[key].get_ase()
        _assert_atoms_match(generated_structure, legacy_entry['structure'])

        metadata = structure_map[key]
        assert metadata['point_index'] == index
        assert metadata['structure_uuid'] == generated[key].uuid
        npt.assert_allclose(metadata['burger_vector'], legacy_entry['burger_vector'], atol=1.0e-12)


def test_generate_faulted_structures_preserves_general_fault_metadata(aiida_profile_clean, aluminum_fcc) -> None:
    """Generalized fault metadata should remain aligned with the generated structures."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='111').store()
    expected_points = node.get_faulted_structure(fault_mode='general', fault_type='general')

    assert expected_points is not None

    generated = generate_faulted_structures(
        faulted_data=node,
        fault_mode=orm.Str('general'),
        fault_type=orm.Str('general'),
    )
    structure_map = generated['structure_map'].get_dict()

    assert len(expected_points) == len(structure_map)

    for index, expected_point in enumerate(expected_points, start=1):
        key = f'sfe_idx_{index:03d}'
        metadata = structure_map[key]

        assert metadata['point_index'] == index
        assert metadata['direction_name'] == expected_point['direction_name']
        assert metadata['path_index'] == expected_point['path_index']
        assert metadata['step_index'] == expected_point['step_index']
        npt.assert_allclose(metadata['burger_vector'], expected_point['burger_vector'], atol=1.0e-12)
        _assert_atoms_match(generated[key].get_ase(), expected_point['structure'])


def test_generate_faulted_structures_raises_for_unsupported_fault(aiida_profile_clean, aluminum_fcc) -> None:
    """The calcfunction should fail cleanly when the requested fault is not available."""
    node = FaultedStructureData(ase=aluminum_fcc, n_unit_cells=4, gliding_plane='100').store()

    with pytest.raises(ValueError, match='No faulted structures could be generated'):
        generate_faulted_structures(
            faulted_data=node,
            fault_mode=orm.Str('removal'),
            fault_type=orm.Str('intrinsic'),
        )
