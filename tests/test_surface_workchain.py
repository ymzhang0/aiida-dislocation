"""Tests for ``GSFEWorkChain`` and ``SurfaceEnergyWorkChain`` surface-energy behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiida import orm
from aiida.engine import WorkChain
from ase.build import bulk

from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.data.faulted_structure import FaultedStructureData
from aiida_dislocation.workflows import gsfe as gsfe_module
from aiida_dislocation.workflows import mixins
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


def _build_process_for_cleanup_test(
    workchain_class: type[GSFEWorkChain] | type[SurfaceEnergyWorkChain],
    aluminum_structure: orm.StructureData,
    clean_workdir: bool,
):
    """Return a minimal process instance for ``on_terminated`` tests."""
    builder = workchain_class.get_builder()
    builder.structure = aluminum_structure
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(clean_workdir)

    if workchain_class is GSFEWorkChain:
        builder.faulted_structure_data = FaultedStructureData(n_unit_cells=4, gliding_plane='111')
        builder.pop('relax', None)
        builder.pop('scf', None)
        builder.pop('sfe', None)
    else:
        builder.cleavaged_structure_data = CleavagedStructureData(
            n_unit_cells=4,
            gliding_plane='111',
            vacuum_spacings=[0.5],
        )
        builder.pop('relax', None)
        builder.pop('scf', None)
        builder.pop('surface_energy', None)

    return workchain_class(builder)


@pytest.mark.parametrize('workchain_class', [GSFEWorkChain, SurfaceEnergyWorkChain])
def test_workchain_on_terminated_cleans_remote_folders(
    aiida_profile_clean,
    aluminum_structure,
    monkeypatch: pytest.MonkeyPatch,
    workchain_class,
) -> None:
    """Both workchains should clean descendant remote folders when requested."""
    cleaned_calls: list[int] = []
    report_messages: list[str] = []

    class FakeRemoteFolder:
        def __init__(self, pk: int, should_raise: bool = False) -> None:
            self.pk = pk
            self.should_raise = should_raise

        def _clean(self) -> None:
            if self.should_raise:
                raise OSError('cleanup failed')
            cleaned_calls.append(self.pk)

    class FakeCalcJobNode:
        def __init__(self, pk: int, should_raise: bool = False) -> None:
            self.pk = pk
            self.outputs = SimpleNamespace(remote_folder=FakeRemoteFolder(pk, should_raise=should_raise))

    process = _build_process_for_cleanup_test(workchain_class, aluminum_structure, clean_workdir=True)
    descendants = [FakeCalcJobNode(11), FakeCalcJobNode(12, should_raise=True), object()]

    monkeypatch.setattr(WorkChain, 'on_terminated', lambda self: None)
    monkeypatch.setattr(mixins.orm, 'CalcJobNode', FakeCalcJobNode)
    monkeypatch.setattr(type(process.node), 'called_descendants', property(lambda _: descendants))
    process.report = report_messages.append  # type: ignore[method-assign]

    process.on_terminated()

    assert cleaned_calls == [11]
    assert any('cleaned remote folders of calculations: 11' in message for message in report_messages)


@pytest.mark.parametrize('workchain_class', [GSFEWorkChain, SurfaceEnergyWorkChain])
def test_workchain_on_terminated_ignores_runtime_cleanup_failures(
    aiida_profile_clean,
    aluminum_structure,
    monkeypatch: pytest.MonkeyPatch,
    workchain_class,
) -> None:
    """Cleanup failures caused by transport runtime errors should not break termination."""
    cleaned_calls: list[int] = []
    report_messages: list[str] = []

    class FakeRemoteFolder:
        def __init__(self, pk: int, should_raise: bool = False) -> None:
            self.pk = pk
            self.should_raise = should_raise

        def _clean(self) -> None:
            if self.should_raise:
                raise RuntimeError('no greenback portal is available')
            cleaned_calls.append(self.pk)

    class FakeCalcJobNode:
        def __init__(self, pk: int, should_raise: bool = False) -> None:
            self.pk = pk
            self.outputs = SimpleNamespace(remote_folder=FakeRemoteFolder(pk, should_raise=should_raise))

    process = _build_process_for_cleanup_test(workchain_class, aluminum_structure, clean_workdir=True)
    descendants = [FakeCalcJobNode(31, should_raise=True), FakeCalcJobNode(32)]

    monkeypatch.setattr(WorkChain, 'on_terminated', lambda self: None)
    monkeypatch.setattr(mixins.orm, 'CalcJobNode', FakeCalcJobNode)
    monkeypatch.setattr(type(process.node), 'called_descendants', property(lambda _: descendants))
    process.report = report_messages.append  # type: ignore[method-assign]

    process.on_terminated()

    assert cleaned_calls == [32]
    assert any('cleaned remote folders of calculations: 32' in message for message in report_messages)


@pytest.mark.parametrize('workchain_class', [GSFEWorkChain, SurfaceEnergyWorkChain])
def test_workchain_on_terminated_respects_clean_workdir_false(
    aiida_profile_clean,
    aluminum_structure,
    monkeypatch: pytest.MonkeyPatch,
    workchain_class,
) -> None:
    """Both workchains should skip cleanup when ``clean_workdir`` is disabled."""
    report_messages: list[str] = []

    class FakeCalcJobNode:
        def __init__(self, pk: int) -> None:
            self.pk = pk
            self.outputs = SimpleNamespace(remote_folder=SimpleNamespace(_clean=lambda: None))

    process = _build_process_for_cleanup_test(workchain_class, aluminum_structure, clean_workdir=False)
    descendants = [FakeCalcJobNode(21)]

    monkeypatch.setattr(WorkChain, 'on_terminated', lambda self: None)
    monkeypatch.setattr(mixins.orm, 'CalcJobNode', FakeCalcJobNode)
    monkeypatch.setattr(type(process.node), 'called_descendants', property(lambda _: descendants))
    process.report = report_messages.append  # type: ignore[method-assign]

    process.on_terminated()

    assert report_messages == ['remote folders will not be cleaned']


def test_gsfe_workchain_generate_structures_indexes_faulted_outputs(
    aiida_profile_clean,
    aluminum_structure,
) -> None:
    """GSFE workflow should build faulted-structure entries from direct calcfunction outputs."""
    builder = GSFEWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.faulted_structure_data = FaultedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
    )
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('sfe', None)

    process = GSFEWorkChain(builder)
    captured_outputs = {}

    def _capture_output(label: str, node: orm.Data) -> None:
        captured_outputs[label] = node

    process.out = _capture_output  # type: ignore[method-assign]
    result = process.generate_structures()

    assert result is None
    assert process.ctx.number_of_structures > 0
    assert process.ctx.generated_structures[0]['structure_key'] == 'sfe_110_000'
    assert process.ctx.generated_structures[0]['direction_name']
    assert isinstance(process.ctx.generated_structures[0]['burger_vector'], list)
    assert process.ctx.generated_structures[0]['interface_slips'] == {}
    assert captured_outputs == {}


def test_gsfe_workchain_reuses_first_faulted_structure_kpoints(
    aiida_profile_clean,
    aluminum_structure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GSFE should generate faulted-structure kpoints once from the first slip state and then reuse them."""
    builder = GSFEWorkChain.get_builder()
    builder.structure = aluminum_structure
    builder.faulted_structure_data = FaultedStructureData(
        n_unit_cells=4,
        gliding_plane='111',
    )
    builder.kpoints_distance = orm.Float(0.3)
    builder.clean_workdir = orm.Bool(False)
    builder.pop('relax', None)
    builder.pop('scf', None)
    builder.pop('sfe', None)

    process = GSFEWorkChain(builder)
    create_calls: list[str] = []

    def _fake_create_kpoints_from_distance(**inputs):
        create_calls.append(inputs['metadata']['call_link_label'])
        kpoints = orm.KpointsData()
        if inputs['metadata']['call_link_label'] == 'create_kpoints_from_distance':
            kpoints.set_kpoints_mesh([4, 4, 2])
        else:
            kpoints.set_kpoints_mesh([4, 4, 1])
        return kpoints

    monkeypatch.setattr(gsfe_module, 'create_kpoints_from_distance', _fake_create_kpoints_from_distance)

    assert process.generate_structures() is None
    process.setup()

    assert create_calls == [
        'create_kpoints_from_distance',
        'create_kpoints_from_distance_sfe',
    ]
    assert process.ctx.kpoints_sfe.get_kpoints_mesh()[0] == [4, 4, 1]


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
    process.ctx.results = {
        'slab_0_500000': {
            'vacuum_spacing': 0.5,
            'structure_uuid': 'slab-uuid-001',
            'total_energy_ev': -10.0,
            'surface_energy_j_m2': 0.12,
            'workchain_uuid': 'uuid-101',
        },
        'slab_1_000000': {
            'vacuum_spacing': 1.0,
            'structure_uuid': 'slab-uuid-002',
            'total_energy_ev': -9.5,
            'surface_energy_j_m2': 0.18,
            'workchain_uuid': 'uuid-102',
        },
    }

    captured_outputs = {}

    def _capture_output(label: str, node: orm.Data) -> None:
        captured_outputs[label] = node

    process.out = _capture_output  # type: ignore[method-assign]
    process.results()

    assert 'results' in captured_outputs
    assert captured_outputs['results'].is_stored

    results = captured_outputs['results'].get_dict()

    assert results['slab_0_500000']['surface_energy_j_m2'] == pytest.approx(0.12)
    assert results['slab_1_000000']['surface_energy_j_m2'] == pytest.approx(0.18)
    assert results['slab_0_500000']['vacuum_spacing'] == pytest.approx(0.5)
    assert results['slab_1_000000']['vacuum_spacing'] == pytest.approx(1.0)
    assert results['slab_0_500000']['structure_uuid'] == 'slab-uuid-001'
    assert results['slab_1_000000']['workchain_uuid'] == 'uuid-102'


def test_surface_workchain_generate_structures_indexes_by_vacuum_spacing(
    aiida_profile_clean,
    aluminum_structure,
) -> None:
    """Surface workflow should build slab entries from direct calcfunction outputs."""
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
    assert process.ctx.number_of_spacings == 2
    assert process.ctx.generated_structures[0]['vacuum_spacing'] == pytest.approx(0.5)
    assert process.ctx.generated_structures[1]['vacuum_spacing'] == pytest.approx(1.0)
    assert process.ctx.generated_structures[0]['structure'].uuid is not None
    assert captured_outputs == {}


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
    process.ctx.current_call_link_label = 'slab_0_500000'
    process.ctx.current_spacing = 0.5
    process.ctx.current_structure = aluminum_structure
    process.ctx.results = {}
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
    assert process.ctx.results['slab_0_500000']['vacuum_spacing'] == pytest.approx(0.5)
    assert process.ctx.results['slab_0_500000']['structure_uuid'] == aluminum_structure.uuid
    assert process.ctx.results['slab_0_500000']['surface_energy_j_m2'] == pytest.approx(0.12)


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
            'label': 'sfe_110_000',
            'structure_uuid': 'structure-uuid-001',
            'direction_name': '110',
            'step_index': 0,
            'burger_vector': [0.0, 0.0, 0.0],
            'total_cell_shift': [0.0, 0.0, 0.0],
            'interface_slips': {},
            'energy': -10.0,
            'sfe': 0.12,
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

    assert 'results' in captured_outputs
    assert captured_outputs['results'].is_stored
    results = captured_outputs['results'].get_dict()
    assert results['surface_area_angstrom2'] == 7.5
    assert results['number_of_structures'] == 1
    assert results['conventional_energy_ev'] == -2.5
    assert results['results']['110']['0']['sfe'] == pytest.approx(0.12)
    assert results['results']['110']['0']['energy'] == pytest.approx(-10.0)
    assert results['results']['110']['0']['interface_slips'] == {}
