"""Regression tests for ``FaultedStructureData`` and provenance-aware GSFE structure generation."""

from __future__ import annotations

import numpy.testing as npt
import pytest
from aiida import orm
from aiida.common.exceptions import ModificationNotAllowed
from ase import Atoms
from ase.build import bulk

from aiida_dislocation.calculations import generate_faulted_structures
from aiida_dislocation.data.cleavaged_structure import CleavagedStructure, PlanarStructure
from aiida_dislocation.data.faulted_structure import FaultedStructureData
from aiida_dislocation.data.gliding_systems import get_gliding_system
from aiida_dislocation.tools.structure import (
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


def test_faulted_structure_data_round_trips_attributes(aiida_profile_clean) -> None:
    """Stored ``FaultedStructureData`` should preserve only configuration attributes."""
    node = FaultedStructureData(n_unit_cells=4, gliding_plane='111')

    assert node.n_unit_cells == 4
    assert node.gliding_plane == '111'

    node.store()
    loaded = orm.load_node(node.pk)

    assert loaded.n_unit_cells == 4
    assert loaded.gliding_plane == '111'


def test_faulted_structure_data_is_immutable_after_store(aiida_profile_clean) -> None:
    """Node attributes should not be mutable after storing the AiiDA data node."""
    node = FaultedStructureData(n_unit_cells=4, gliding_plane='111').store()

    with pytest.raises(ModificationNotAllowed):
        node.base.attributes.set(FaultedStructureData.N_UNIT_CELLS_KEY, 5)


def test_faulted_structure_helper_uses_shared_base_not_surface_helper() -> None:
    """Faulted and cleavaged helpers should share a base instead of inheriting from each other."""
    faulted_helper = FaultedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
    ).get_structure_builder(bulk('Al', 'fcc', a=4.05))

    assert isinstance(faulted_helper, PlanarStructure)
    assert not isinstance(faulted_helper, CleavagedStructure)


def test_gliding_system_general_fault_paths_use_single_tuple_layer() -> None:
    """Generalized fault definitions should store a single tuple of steps per direction."""
    gliding_system = get_gliding_system('C1_b')
    direction_steps = gliding_system.get_plane('011').general.burger_vectors['100']

    assert direction_steps == ((2, [1, 0, 0]),)
    assert isinstance(direction_steps[0][0], int)
    assert isinstance(direction_steps[0][1], list)


def test_generate_faulted_structures_matches_legacy_general_fault_path(aiida_profile_clean, aluminum_fcc) -> None:
    """The calcfunction outputs should keep the same cells and atomic positions as the legacy path."""
    structure = orm.StructureData(ase=aluminum_fcc)
    config = FaultedStructureData(n_unit_cells=4, gliding_plane='111').store()

    generated = generate_faulted_structures(
        structure=structure,
        faulted_data=config,
        fault_mode=orm.Str('general'),
        fault_type=orm.Str('general'),
    )

    _, legacy_conventional = get_conventional_structure(aluminum_fcc, gliding_plane='111')
    _, legacy_faulted = get_faulted_structure(
        legacy_conventional,
        fault_mode='general',
        fault_type='general',
        gliding_plane='111',
        n_unit_cells=4,
    )

    assert legacy_faulted is not None

    legacy_entries = legacy_faulted['structures']
    generated_structures = {
        label: node for label, node in generated.items() if label.startswith('sfe_idx_')
    }

    _assert_atoms_match(generated['conventional_structure'].get_ase(), legacy_conventional)
    assert len(generated_structures) == len(legacy_entries)
    assert 'structure_map' not in generated
    assert 'cleavaged_structure' not in generated

    for index, legacy_entry in enumerate(legacy_entries, start=1):
        key = f'sfe_idx_{index:03d}'
        assert key in generated

        generated_structure = generated[key].get_ase()
        _assert_atoms_match(generated_structure, legacy_entry['structure'])
        npt.assert_allclose(
            generated[f'burger_vector_{key}'].get_list(),
            legacy_entry['burger_vector'],
            atol=1.0e-12,
        )


def test_generate_faulted_structures_preserves_general_fault_metadata(aiida_profile_clean, aluminum_fcc) -> None:
    """Generalized fault metadata should remain aligned with the generated structures."""
    structure = orm.StructureData(ase=aluminum_fcc)
    config = FaultedStructureData(n_unit_cells=4, gliding_plane='111').store()
    expected_points = config.get_structure_builder(structure).get_faulted_structure(
        fault_mode='general',
        fault_type='general',
    )

    assert expected_points is not None

    generated = generate_faulted_structures(
        structure=structure,
        faulted_data=config,
        fault_mode=orm.Str('general'),
        fault_type=orm.Str('general'),
    )

    assert len(expected_points) == len([label for label in generated if label.startswith('sfe_idx_')])

    for index, expected_point in enumerate(expected_points, start=1):
        key = f'sfe_idx_{index:03d}'
        assert generated[f'direction_name_{key}'].value == expected_point['direction_name']
        assert generated[f'path_index_{key}'].value == expected_point['path_index']
        assert generated[f'step_index_{key}'].value == expected_point['step_index']
        npt.assert_allclose(
            generated[f'burger_vector_{key}'].get_list(),
            expected_point['burger_vector'],
            atol=1.0e-12,
        )
        _assert_atoms_match(generated[key].get_ase(), expected_point['structure'])


def test_generate_faulted_structures_raises_for_unsupported_fault(aiida_profile_clean, aluminum_fcc) -> None:
    """The calcfunction should fail cleanly when the requested fault is not available."""
    structure = orm.StructureData(ase=aluminum_fcc)
    config = FaultedStructureData(n_unit_cells=4, gliding_plane='100').store()

    with pytest.raises(ValueError, match='No faulted structures could be generated'):
        generate_faulted_structures(
            structure=structure,
            faulted_data=config,
            fault_mode=orm.Str('removal'),
            fault_type=orm.Str('intrinsic'),
        )
