from __future__ import annotations

import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ExitCode, WorkChain, append_, if_, while_
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_dislocation.calculations import generate_faulted_structures
from aiida_dislocation.data.faulted_structure import FaultedStructureData

from .mixins import (
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
    clean_workchain_calcs,
)

class GSFEWorkChain(
    ProtocolMixin,
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
    WorkChain):
    """GSFE WorkChain"""

    _NAMESPACE = 'gsfe'

    _RELAX_NAMESPACE = "relax"
    _SCF_NAMESPACE = "scf"
    _SFE_NAMESPACE = "sfe"

    _RY2eV = 13.605693122990
    _eVA22Jm2 = 1.602176634E-19 * 1E+20
    
    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input(
            'faulted_structure_data',
            valid_type=FaultedStructureData,
            required=False,
            default=lambda: FaultedStructureData(n_unit_cells=4),
            help='Configuration for GSFE faulted-structure generation.',
        )
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.3),
                help='The distance between kpoints for the kpoints generation')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._RELAX_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._SCF_NAMESPACE,
            exclude=(
                'pw.structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for SCF calculation.'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._SFE_NAMESPACE,
            exclude=(
                'pw.structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for USF calculation.'
            }
        )

        spec.outline(
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.generate_structures,
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            while_(cls.should_run_sfe)(
                cls.run_sfe,
                cls.inspect_sfe,
            ),
            cls.results,
        )
        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace=cls._RELAX_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._SCF_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.output(
            'gsfe_results',
            valid_type=orm.Dict,
            required=False,
            help='Aggregated GSFE results for all evaluated faulted structures.',
        )
        
        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_RELAX",
            message='The `PwBaseWorkChain` for the GSF run failed.',
        )
        
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )
        spec.exit_code(
            405,
            "ERROR_NO_STRUCTURE_TYPE_DETECTED",
            message='The structure type is not detected.',
        )
        
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict[str, ty.Any]:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml
        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(
            cls,
            code,
            structure,
            protocol='moderate',
            overrides=None,
            n_repeats: ty.Optional[int | orm.Int] = None,
            gliding_plane: ty.Optional[str | orm.Str] = None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (code, structure, protocol)

        builder = cls.get_builder()

        # Set up the sub-workchains
        for namespace, workchain_type in [
            (cls._RELAX_NAMESPACE, PwRelaxWorkChain),
            (cls._SCF_NAMESPACE, PwBaseWorkChain),
            (cls._SFE_NAMESPACE, PwBaseWorkChain),
        ]:
            overrides = inputs.get(namespace, {})

            if workchain_type == PwRelaxWorkChain:
                overrides.setdefault('base_relax', {})['pseudo_family'] = inputs.get('pseudo_family', None)
                overrides.setdefault('base_init_relax', {})['pseudo_family'] = inputs.get('pseudo_family', None)
            else:
                overrides['pseudo_family'] = inputs.get('pseudo_family', None)

            sub_builder = workchain_type.get_builder_from_protocol(
                *args,
                overrides=overrides,
            )
            sub_builder.pop('structure', None)
            sub_builder.pop('clean_workdir', None)

            if namespace != cls._RELAX_NAMESPACE:
                sub_builder.pop('kpoints', None)
                sub_builder.pop('kpoints_distance', None)

            builder[namespace]._data = sub_builder._data

        if cls._RELAX_NAMESPACE in builder:
            builder[cls._RELAX_NAMESPACE].pop('base_init_relax', None)
            if 'base_relax' in builder[cls._RELAX_NAMESPACE]:
                builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints', None)
                builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints_distance', None)
        
        builder.structure = structure
        resolved_n_repeats = n_repeats.value if isinstance(n_repeats, orm.Int) else n_repeats
        resolved_gliding_plane = gliding_plane.value if isinstance(gliding_plane, orm.Str) else gliding_plane
        builder.faulted_structure_data = FaultedStructureData(
            n_unit_cells=inputs.get('n_repeats', 4) if resolved_n_repeats is None else resolved_n_repeats,
            gliding_plane=inputs.get('gliding_plane', '') if resolved_gliding_plane is None else resolved_gliding_plane,
        )
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder


    def should_run_relax(self) -> bool:
        return self._RELAX_NAMESPACE in self.inputs

    def run_relax(self) -> dict[str, orm.ProcessNode]:
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._RELAX_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._RELAX_NAMESPACE
        inputs.structure = self.inputs.structure
        inputs.base_relax.kpoints_distance = self.inputs.kpoints_distance
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for primitive structure')
        return {f"workchain_relax": running}

    def inspect_relax(self) -> ty.Optional[ExitCode]:
        workchain = self.ctx.workchain_relax
        if not workchain.is_finished_ok:
            self.report(f'PwRelaxWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'PwRelaxWorkChain<{self.ctx.workchain_relax.pk}> finished')

        self.ctx.current_structure = workchain.outputs.output_structure
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._RELAX_NAMESPACE
            )
        )

    def generate_structures(self) -> ty.Optional[ExitCode]:
        """Generate provenance-tracked structures for GSFE calculations."""
        
        if 'current_structure' not in self.ctx:
            self.ctx.current_structure = self.inputs.structure

        try:
            generated_structures = generate_faulted_structures(
                structure=self.ctx.current_structure,
                faulted_data=self.inputs.faulted_structure_data,
                fault_mode=orm.Str('general'),
                fault_type=orm.Str('general'),
            )
        except ValueError as exception:
            self.report(f'Failed to generate GSFE structures: {exception}')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        self.ctx.generated_structures = []

        for output_label, output_node in generated_structures.items():
            if output_label in ('conventional_structure', 'surface_area') or not isinstance(output_node, orm.StructureData):
                continue

            direction_name = output_node.base.extras.get('direction_name', None)
            step_index = output_node.base.extras.get('step_index', None)
            burger_vector = output_node.base.extras.get('burger_vector', None)
            total_cell_shift = output_node.base.extras.get('total_cell_shift', None)
            interface_slips = output_node.base.extras.get('interface_slips', None)

            if None in (direction_name, step_index, burger_vector, total_cell_shift, interface_slips):
                self.report(f'Incomplete faulted-structure entry generated for `{output_label}`.')
                return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

            self.ctx.generated_structures.append({
                'structure_key': output_label,
                'structure': output_node,
                'direction_name': str(direction_name),
                'step_index': int(step_index),
                'burger_vector': [float(value) for value in burger_vector],
                'total_cell_shift': [float(value) for value in total_cell_shift],
                'interface_slips': {
                    str(interface): [float(value) for value in interface_shift]
                    for interface, interface_shift in interface_slips.items()
                },
            })

        self.ctx.generated_structures.sort(
            key=lambda entry: (entry['direction_name'], entry['step_index'])
        )

        if not self.ctx.generated_structures:
            self.report('No generalized fault path is available for the selected structure and gliding plane.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        self.ctx.number_of_structures = len(self.ctx.generated_structures)
        self.ctx.conventional_structure = generated_structures['conventional_structure']
        self.ctx.surface_area = generated_structures['surface_area'].value
        
        self.report(f'Surface area of the conventional geometry: {self.ctx.surface_area} Angstrom^2')

        self.ctx.unit_cell_multiplier = self._calculate_structure_multiplier(self.ctx.current_structure)
        self.ctx.conventional_multiplier = self._calculate_structure_multiplier(self.ctx.conventional_structure)

    def _get_kpoints_scf(self) -> orm.KpointsData:
        """Get or create kpoints_scf. Returns kpoints_scf KpointsData object."""
        if 'kpoints_scf' in self.ctx:
            kpoints_scf = self.ctx.kpoints_scf
        else:
            inputs = {
                'structure': self.ctx.conventional_structure,
                'distance': self.inputs.kpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints_scf = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg
        
        return kpoints_scf

    def setup(self) -> None:
        self.ctx.iteration = 0
        self.ctx.sfe_results = []
        # Get kpoints_scf
        kpoints_scf = self._get_kpoints_scf()
        
        self.ctx.kpoints_scf = kpoints_scf

    def should_run_scf(self) -> bool:
        return self._SCF_NAMESPACE in self.inputs

    def run_scf(self) -> dict[str, orm.ProcessNode]:
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SCF_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._SCF_NAMESPACE

        inputs.pw.structure = self.ctx.conventional_structure
        inputs.kpoints = self.ctx.kpoints_scf

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for conventional structure')

        return {f"workchain_scf": running}

    def inspect_scf(self) -> ty.Optional[ExitCode]:
        workchain = self.ctx.workchain_scf
        if not workchain.is_finished_ok:
            self.report(f'PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f'PwBaseWorkChain<{self.ctx.workchain_scf.pk}> finished')
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SCF_NAMESPACE
            )
        )
        self.ctx.total_energy_conventional_geometry = self._get_workchain_energy(workchain)

    def should_run_sfe(self) -> bool:

        if self._SFE_NAMESPACE not in self.inputs:
            return False

        if self.ctx.iteration >= self.ctx.number_of_structures:
            return False

        current_entry = self.ctx.generated_structures[self.ctx.iteration]
        self.ctx.current_structure_key = current_entry['structure_key']
        self.ctx.current_structure = current_entry['structure']
        self.ctx.current_direction_name = current_entry['direction_name']
        self.ctx.current_step_index = current_entry['step_index']
        self.ctx.current_burger_vector = current_entry['burger_vector']
        self.ctx.current_total_cell_shift = current_entry['total_cell_shift']
        self.ctx.current_interface_slips = current_entry['interface_slips']
        self.ctx.current_multiplier = self._calculate_structure_multiplier(self.ctx.current_structure)
        self.ctx.kpoints_sfe = self._calculate_kpoints_for_structure(
            self.ctx.current_structure,
            self.ctx.kpoints_scf,
        )

        return True

    def run_sfe(self) -> dict[str, ty.Any]:
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self.ctx.current_structure_key

        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints_sfe

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(
            f'launching PwBaseWorkChain<{running.pk}> for faulted structure '
            f'{self.ctx.iteration + 1}/{self.ctx.number_of_structures} ({self.ctx.current_structure_key}).'
        )

        return {f"workchain_sfe": append_(running)}

    def inspect_sfe(self) -> ty.Optional[ExitCode]:
        workchain = self.ctx.workchain_sfe[-1]
        if not workchain.is_finished_ok:
            self.report(f'PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(f'PwBaseWorkChain<{workchain.pk}> finished')

        total_energy_faulted_geometry = self._get_workchain_energy(workchain)
        gsfe_j_m2 = None

        if 'total_energy_conventional_geometry' in self.ctx:
            gsfe_j_m2 = self._calculate_stacking_fault_energy(
                total_energy_faulted_geometry,
                self.ctx.current_multiplier,
                'generalized stacking fault'
            )

        self.ctx.sfe_results.append({
            'label': self.ctx.current_structure_key,
            'structure_uuid': self.ctx.current_structure.uuid,
            'direction_name': self.ctx.current_direction_name,
            'step_index': self.ctx.current_step_index,
            'burger_vector': [float(value) for value in self.ctx.current_burger_vector],
            'total_cell_shift': [float(value) for value in self.ctx.current_total_cell_shift],
            'interface_slips': {
                str(interface): [float(value) for value in interface_shift]
                for interface, interface_shift in self.ctx.current_interface_slips.items()
            },
            'energy': float(total_energy_faulted_geometry),
            'sfe': float(gsfe_j_m2) if gsfe_j_m2 is not None else None,
            'workchain_uuid': workchain.uuid,
        })
        self.ctx.iteration += 1

    def results(self) -> None:
        """Output collected results."""
        nested_results: dict[str, dict[str, dict[str, ty.Any]]] = {}

        for point_result in self.ctx.sfe_results:
            direction_results = nested_results.setdefault(point_result['direction_name'], {})
            direction_results[str(point_result['step_index'])] = {
                'label': point_result['label'],
                'structure_uuid': point_result['structure_uuid'],
                'step_index': point_result['step_index'],
                'burger_vector': point_result['burger_vector'],
                'total_cell_shift': point_result['total_cell_shift'],
                'interface_slips': point_result['interface_slips'],
                'energy': point_result['energy'],
                'sfe': point_result['sfe'],
                'workchain_uuid': point_result['workchain_uuid'],
            }

        results = {
            'results': nested_results,
            'surface_area_angstrom2': float(self.ctx.surface_area),
            'number_of_structures': self.ctx.number_of_structures,
        }

        if 'total_energy_conventional_geometry' in self.ctx:
            results['conventional_energy_ev'] = float(self.ctx.total_energy_conventional_geometry)

        self.out('gsfe_results', orm.Dict(dict=results).store())

    def on_terminated(self) -> None:
        """Clean child calculation working directories if ``clean_workdir`` is enabled."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
