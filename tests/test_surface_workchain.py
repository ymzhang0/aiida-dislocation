"""Tests for ``GSFEWorkChain`` and ``SurfaceEnergyWorkChain`` surface-energy behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiida import orm
from ase.build import bulk

from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.workflows.gsfe import GSFEWorkChain
from aiida_dislocation.workflows.surface import SurfaceEnergyWorkChain


@pytest.fixture
def aluminum_structure(aiida_profile_clean) -> orm.StructureData:
    """Return a simple aluminum FCC structure."""
    return orm.StructureData(ase=bulk('Al', 'fcc', a=4.05))


def test_gsfe_workchain_no_longer_exposes_surface_energy_namespace() -> None:
    """GSFE should no longer carry its own surface-energy sub workflow."""
    builder = GSFEWorkChain.get_builder()

    assert 'surface_energy' not in builder


def test_surface_workchain_results_aggregates_vacuum_spacings(aiida_profile_clean, aluminum_structure) -> None:
    """Surface workchain results should retain one entry per evaluated vacuum spacing."""
    builder = SurfaceEnergyWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.cleavaged_structure_data = CleavagedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
        vacuum_spacings=[0.5, 1.0],
    )
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('surface_energy', None)

    process = SurfaceEnergyWorkChain(builder)
    process.ctx.surface_results = [
        {
            'iteration': 1,
            'structure_label': 'slab_idx_001',
            'structure_uuid': 'slab-uuid-001',
            'point_index': 1,
            'vacuum_spacing': 0.5,
            'structure_formula': 'Al4',
            'surface_multiplier': 4,
            'total_energy_ev': -10.0,
            'surface_energy_j_m2': 0.12,
            'workchain_pk': 101,
            'workchain_uuid': 'uuid-101',
        },
        {
            'iteration': 2,
            'structure_label': 'slab_idx_002',
            'structure_uuid': 'slab-uuid-002',
            'point_index': 2,
            'vacuum_spacing': 1.0,
            'structure_formula': 'Al4',
            'surface_multiplier': 4,
            'total_energy_ev': -9.5,
            'surface_energy_j_m2': 0.18,
            'workchain_pk': 102,
            'workchain_uuid': 'uuid-102',
        },
    ]
    process.ctx.surface_area = 7.5
    process.ctx.number_of_spacings = 2
    process.ctx.total_energy_conventional_geometry = -2.5

    captured_outputs = {}

    def _capture_output(label: str, node: orm.Data) -> None:
        captured_outputs[label] = node

    process.out = _capture_output  # type: ignore[method-assign]
    process.results()

    assert 'surface_results' in captured_outputs
    assert captured_outputs['surface_results'].is_stored

    results = captured_outputs['surface_results'].get_dict()

    assert results['surface_area_angstrom2'] == 7.5
    assert results['number_of_spacings'] == 2
    assert results['conventional_energy_ev'] == -2.5
    assert results['results']['0.500000']['surface_energy_j_m2'] == pytest.approx(0.12)
    assert results['results']['1.000000']['surface_energy_j_m2'] == pytest.approx(0.18)
    assert results['results']['0.500000']['structure_label'] == 'slab_idx_001'
    assert results['results']['1.000000']['point_index'] == 2
    assert results['results']['0.500000']['workchain_pk'] == 101
    assert results['results']['1.000000']['workchain_uuid'] == 'uuid-102'


def test_surface_workchain_generate_structures_uses_cleavaged_structure_data(
    aiida_profile_clean,
    aluminum_structure,
) -> None:
    """Surface workflow should build and expose ``CleavagedStructureData``."""
    builder = SurfaceEnergyWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.cleavaged_structure_data = CleavagedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
        vacuum_spacings=[0.5, 1.0],
    )
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('surface_energy', None)

    process = SurfaceEnergyWorkChain(builder)
    captured_outputs = {}

    def _capture_output(label: str, node: orm.Data) -> None:
        captured_outputs[label] = node

    process.out = _capture_output  # type: ignore[method-assign]
    result = process.generate_structures()

    assert result is None
    assert isinstance(process.ctx.cleavaged_structure_data, CleavagedStructureData)
    assert 'cleavaged_structure_data' in captured_outputs
    assert 'structure_map' in captured_outputs
    assert captured_outputs['cleavaged_structure_data'].uuid == process.ctx.cleavaged_structure_data.uuid


def test_surface_workchain_inspect_surface_energy_uses_structuredata_formula(
    aiida_profile_clean,
    aluminum_structure,
) -> None:
    """Surface-energy inspection should read the formula from ``StructureData`` without ASE-only APIs."""
    builder = SurfaceEnergyWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.cleavaged_structure_data = CleavagedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
        vacuum_spacings=[0.5],
    )
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('surface_energy', None)

    process = SurfaceEnergyWorkChain(builder)
    process.ctx.iteration = 1
    process.ctx.current_structure_key = 'slab_idx_001'
    process.ctx.current_structure_uuid = aluminum_structure.uuid
    process.ctx.current_point_index = 1
    process.ctx.current_spacing = 0.5
    process.ctx.current_structure = aluminum_structure
    process.ctx.surface_results = []
    process.ctx.workchain_surface_energy = SimpleNamespace(
        is_finished_ok=True,
        pk=101,
        uuid='uuid-101',
    )

    process._get_workchain_energy = lambda _: -10.0  # type: ignore[method-assign]
    process._calculate_structure_multiplier = lambda _: 4  # type: ignore[method-assign]
    process._calculate_surface_energy = lambda *_: 0.12  # type: ignore[method-assign]

    result = process.inspect_surface_energy()

    assert result is None
    assert process.ctx.surface_results[0]['structure_formula'] == aluminum_structure.get_formula()


def test_gsfe_workchain_results_output_is_stored(aiida_profile_clean, aluminum_structure) -> None:
    """GSFE aggregated results should be emitted as a stored ``Dict`` node."""
    builder = GSFEWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('sfe', None)

    process = GSFEWorkChain(builder)
    process.ctx.sfe_results = [
        {
            'structure_label': 'sfe_idx_001',
            'structure_uuid': 'structure-uuid-001',
            'iteration': 0,
            'point_index': 1,
            'direction_label': 'partial_path_000',
            'direction_name': 'partial',
            'path_index': 0,
            'step_index': 0,
            'burger_vector': [0.0, 0.0, 0.0],
            'total_energy_ev': -10.0,
            'gsfe_j_m2': 0.12,
            'workchain_pk': 201,
            'workchain_uuid': 'uuid-201',
        }
    ]
    process.ctx.surface_area = 7.5
    process.ctx.number_of_structures = 1
    process.ctx.total_energy_conventional_geometry = -2.5

    captured_outputs = {}

    def _capture_output(label: str, node: orm.Data) -> None:
        captured_outputs[label] = node

    process.out = _capture_output  # type: ignore[method-assign]
    process.results()

    assert 'gsfe_results' in captured_outputs
    assert captured_outputs['gsfe_results'].is_stored
    results = captured_outputs['gsfe_results'].get_dict()
    assert results['surface_area_angstrom2'] == 7.5
    assert results['number_of_structures'] == 1
    assert results['conventional_energy_ev'] == -2.5
    assert results['results']['partial_path_000']['001']['sfe_j_m2'] == pytest.approx(0.12)
