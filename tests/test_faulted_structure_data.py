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


def _flatten_general_fault_result(generated: dict[str, dict[int, dict[str, object]]]) -> list[dict[str, object]]:
    """Flatten the nested generalized-fault mapping into direction/step order."""
    flattened: list[dict[str, object]] = []
    for direction_name, direction_steps in generated.items():
        for step_index, entry in sorted(direction_steps.items()):
            flattened.append({
                'direction_name': direction_name,
                'step_index': step_index,
                'structure': entry['structure'],
                'metadata': entry['metadata'],
            })
    return flattened


def _stringify_interface_slips(interface_slips: dict[int, list[float]]) -> dict[str, list[float]]:
    """Convert interface-slip keys to strings to match JSON/extras storage."""
    return {
        str(interface): [float(value) for value in shift]
        for interface, shift in interface_slips.items()
    }


def _assert_interface_slips_match(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    atol: float = 1.0e-12,
) -> None:
    """Assert that two serialized interface-slip snapshots are numerically equivalent."""
    assert set(left) == set(right)
    for interface, shift in left.items():
        npt.assert_allclose(shift, right[interface], atol=atol)


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


def test_l21_gliding_system_111_uses_twelve_layers() -> None:
    """L21 `111` plane configuration should expose the corrected layer count."""
    gliding_system = get_gliding_system('L2_1')

    assert gliding_system.get_plane('111').n_layers == 12


def test_faulted_structure_general_returns_nested_direction_steps(aluminum_fcc) -> None:
    """Generalized fault generation should return a nested direction -> step mapping."""
    generated = FaultedStructureData(n_unit_cells=4, gliding_plane='111').get_structure_builder(
        aluminum_fcc
    ).get_faulted_structure(
        fault_mode='general',
        fault_type='general',
    )

    assert generated is not None
    assert list(generated) == ['110']
    assert list(generated['110']) == list(range(17))
    assert generated['110'][0]['metadata']['label'] == 'sfe_110_000'
    assert generated['110'][16]['metadata']['label'] == 'sfe_110_016'


def test_faulted_structure_general_tracks_cumulative_interface_slips(aluminum_fcc) -> None:
    """Generalized fault metadata should snapshot cumulative slip on each interface."""
    generated = FaultedStructureData(n_unit_cells=4, gliding_plane='111').get_structure_builder(
        aluminum_fcc
    ).get_faulted_structure(
        fault_mode='general',
        fault_type='general',
    )

    assert generated is not None

    direction_results = generated['110']
    npt.assert_allclose(direction_results[0]['metadata']['total_cell_shift'], [0.0, 0.0, 0.0], atol=1.0e-12)
    assert direction_results[0]['metadata']['interface_slips'] == {}

    npt.assert_allclose(
        direction_results[8]['metadata']['interface_slips'][3],
        [1 / 3, 1 / 3, 0.0],
        atol=1.0e-12,
    )
    npt.assert_allclose(
        direction_results[9]['metadata']['interface_slips'][3],
        [1 / 3, 1 / 3, 0.0],
        atol=1.0e-12,
    )
    npt.assert_allclose(
        direction_results[9]['metadata']['interface_slips'][4],
        [1 / 24, 1 / 24, 0.0],
        atol=1.0e-12,
    )
    npt.assert_allclose(
        direction_results[16]['metadata']['total_cell_shift'],
        [2 / 3, 2 / 3, 0.0],
        atol=1.0e-12,
    )
    npt.assert_allclose(
        direction_results[16]['metadata']['interface_slips'][4],
        [1 / 3, 1 / 3, 0.0],
        atol=1.0e-12,
    )


def test_generate_faulted_structures_matches_legacy_general_fault_path(aiida_profile_clean, aluminum_fcc) -> None:
    """The calcfunction outputs should keep the same cells and atomic positions as the legacy path."""
    structure = orm.StructureData(ase=aluminum_fcc)
    config = FaultedStructureData(n_unit_cells=4, gliding_plane='111').store()
    expected_nested = config.get_structure_builder(structure).get_faulted_structure(
        fault_mode='general',
        fault_type='general',
    )

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
    assert expected_nested is not None

    legacy_entries = legacy_faulted['structures']
    expected_entries = _flatten_general_fault_result(expected_nested)
    generated_structures = {
        label: node for label, node in generated.items() if label.startswith('sfe_')
    }

    _assert_atoms_match(generated['conventional_structure'].get_ase(), legacy_conventional)
    assert len(generated_structures) == len(legacy_entries) == len(expected_entries)

    for expected_entry, legacy_entry in zip(expected_entries, legacy_entries, strict=True):
        metadata = expected_entry['metadata']
        key = metadata['label']
        assert key in generated

        generated_structure = generated[key].get_ase()
        _assert_atoms_match(generated_structure, legacy_entry['structure'])
        npt.assert_allclose(
            metadata['burger_vector'],
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

    flattened_expected = _flatten_general_fault_result(expected_points)
    assert len(flattened_expected) == len([label for label in generated if label.startswith('sfe_')])

    for expected_point in flattened_expected:
        metadata = expected_point['metadata']
        key = metadata['label']
        structure_node = generated[key]

        assert structure_node.base.extras.get('direction_name') == metadata['direction_name']
        assert structure_node.base.extras.get('step_index') == metadata['step_index']
        npt.assert_allclose(
            structure_node.base.extras.get('burger_vector'),
            metadata['burger_vector'],
            atol=1.0e-12,
        )
        npt.assert_allclose(
            structure_node.base.extras.get('total_cell_shift'),
            metadata['total_cell_shift'],
            atol=1.0e-12,
        )
        _assert_interface_slips_match(
            structure_node.base.extras.get('interface_slips'),
            _stringify_interface_slips(metadata['interface_slips']),
        )
        _assert_atoms_match(structure_node.get_ase(), expected_point['structure'])


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
