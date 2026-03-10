from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, if_, while_
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_dislocation.data.faulted_structure import FaultedStructure

from .mixins import (
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
)

class SurfaceEnergyWorkChain(
    ProtocolMixin,
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
    WorkChain):
    """Surface Energy WorkChain"""

    _NAMESPACE = 'surface'

    _RELAX_NAMESPACE = "relax"
    _SCF_NAMESPACE = "scf"
    _SURFACE_ENERGY_NAMESPACE = "surface_energy"
    
    _RY2eV = 13.605693122990
    _eVA22Jm2 = 1.602176634E-19 * 1E+20
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('n_repeats', valid_type=orm.Int, required=False, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('gliding_plane', valid_type=orm.Str, required=False, default=lambda: orm.Str(),
                help='The normal vector for the supercell. Note that please always put the z axis at the last.')
        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.3),
                help='The distance between kpoints for the kpoints generation')
        spec.input('vacuum_spacings', valid_type=orm.List, required=False, default=lambda: orm.List(default_value=[1.0]),
                    help='The vacuum spacings for the surface energy calculation.')
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
            namespace=cls._SURFACE_ENERGY_NAMESPACE,
            exclude=(
                'pw.structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for surface energy calculation.'
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
            while_(cls.should_run_surface_energy)(
                cls.run_surface_energy,
                cls.inspect_surface_energy,
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
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._SURFACE_ENERGY_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.output(
            'surface_results',
            valid_type=orm.Dict,
            required=False,
            help='Aggregated surface-energy results for all evaluated vacuum spacings.',
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
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwBaseWorkChain` for the surface energy run failed.',
        )
        spec.exit_code(
            404,
            "ERROR_NO_STRUCTURE_TYPE_DETECTED",
            message='The structure type is not detected.',
        )
        
    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml
        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)


    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
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
            (cls._SURFACE_ENERGY_NAMESPACE, PwBaseWorkChain),
        ]:
            overrides = inputs.get(namespace, {})

            if workchain_type == PwRelaxWorkChain:
                overrides.setdefault('base', {})['pseudo_family'] = inputs.get('pseudo_family', None)
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
            builder[cls._RELAX_NAMESPACE].pop('base_final_scf', None)
            if 'base' in builder[cls._RELAX_NAMESPACE]:
                builder[cls._RELAX_NAMESPACE]['base'].pop('kpoints', None)
                builder[cls._RELAX_NAMESPACE]['base'].pop('kpoints_distance', None)
        
        builder.structure = structure
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.gliding_plane = orm.Str(inputs.get('gliding_plane', ''))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.vacuum_spacings = orm.List(inputs['vacuum_spacings'])

        return builder


    def should_run_relax(self):
        return self._RELAX_NAMESPACE in self.inputs

    def run_relax(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._RELAX_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._RELAX_NAMESPACE
        inputs.structure = self.inputs.structure
        inputs.base.kpoints_distance = self.inputs.kpoints_distance
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for primitive structure')
        return {f"workchain_relax": running}

    def inspect_relax(self):
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

    def generate_structures(self):
        """Generate base structures (conventional and cleavaged). 
        Subclasses should override generate_faulted_structure() to generate faulted structures."""
        
        if 'current_structure' not in self.ctx:
            self.ctx.current_structure = self.inputs.structure
        
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        faulted_structure = FaultedStructure(
            self.ctx.current_structure.get_ase(),
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )
        self.ctx.faulted_structure = faulted_structure
        # Store structures directly in context
        self.ctx.conventional_structure = faulted_structure.get_conventional_structure()

        self.ctx.surface_area = faulted_structure.surface_area
        
        self.report(f'Surface area of the conventional geometry: {self.ctx.surface_area} Angstrom^2')

        # Calculate and store multipliers using helper method
        self.ctx.unit_cell_multiplier = self._calculate_structure_multiplier(
            self.ctx.current_structure.get_ase()
        )
        self.ctx.conventional_multiplier = self._calculate_structure_multiplier(
            self.ctx.conventional_structure
        )

    def _get_kpoints_scf(self):
        """Get or create kpoints_scf. Returns kpoints_scf KpointsData object."""
        if 'kpoints_scf' in self.ctx:
            kpoints_scf = self.ctx.kpoints_scf
        else:
            inputs = {
                'structure': orm.StructureData(
                    ase=self.ctx.conventional_structure
                    ),
                'distance': self.inputs.kpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints_scf = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg
        
        return kpoints_scf

    def setup(self):
        self.ctx.iteration = 1
        self.ctx.vacuum_spacings = sorted(self.inputs.vacuum_spacings.get_list(), reverse=True)
        self.ctx.surface_results = []
        self.ctx.cleavaged_structure = self.ctx.faulted_structure.get_cleavaged_structure(
            vacuum_spacing=self.ctx.vacuum_spacings[0]
        )
        self.ctx.number_of_spacings = len(self.ctx.vacuum_spacings)
        # Get kpoints_scf
        kpoints_scf = self._get_kpoints_scf()
        
        self.ctx.kpoints_scf = kpoints_scf

        # Calculate kpoints for surface energy using helper method
        self.ctx.kpoints_surface_energy = self._setup_surface_energy_kpoints(kpoints_scf)

    def should_run_scf(self):
        return self._SCF_NAMESPACE in self.inputs

    def run_scf(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SCF_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._SCF_NAMESPACE

        inputs.pw.structure = orm.StructureData(
            ase=self.ctx.conventional_structure
            )        
        
        inputs.kpoints = self.ctx.kpoints_scf

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for conventional structure')

        return {f"workchain_scf": running}

    def inspect_scf(self):
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

    def should_run_surface_energy(self):

        if self._SURFACE_ENERGY_NAMESPACE not in self.inputs:
            return False

        if self.ctx.vacuum_spacings == []:
            return False

        self.ctx.current_spacing = self.ctx.vacuum_spacings.pop() 
        self.ctx.cleavaged_structure = self.ctx.faulted_structure.get_cleavaged_structure(
            vacuum_spacing=self.ctx.current_spacing
        )

        return True

    def run_surface_energy(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SURFACE_ENERGY_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = f"spacing_{self.ctx.iteration:02d}"
        self.ctx.current_structure = self.ctx.faulted_structure.get_cleavaged_structure(
            vacuum_spacing=self.ctx.current_spacing
        )
        inputs.pw.structure = orm.StructureData(ase=self.ctx.current_structure)
        inputs.kpoints = self.ctx.kpoints_surface_energy 

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for cleavaged structure {self.ctx.iteration}/{self.ctx.number_of_spacings}...')

        self.ctx.iteration += 1

        return {f"workchain_surface_energy": running}

    def inspect_surface_energy(self):
        workchain = self.ctx.workchain_surface_energy
        if not workchain.is_finished_ok:
            self.report(f'PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

        self.report(f'PwBaseWorkChain<{self.ctx.workchain_surface_energy.pk}> finished')
        total_energy_slab = self._get_workchain_energy(workchain)
        surface_multiplier = self._calculate_structure_multiplier(self.ctx.current_structure)
        surface_energy_j_m2 = self._calculate_surface_energy(
            total_energy_slab,
            surface_multiplier,
        )

        self.ctx.surface_results.append({
            'iteration': self.ctx.iteration - 1,
            'vacuum_spacing': float(self.ctx.current_spacing),
            'structure_formula': self.ctx.current_structure.get_chemical_formula(),
            'surface_multiplier': surface_multiplier,
            'total_energy_ev': float(total_energy_slab),
            'surface_energy_j_m2': float(surface_energy_j_m2) if surface_energy_j_m2 is not None else None,
            'workchain_pk': workchain.pk,
            'workchain_uuid': workchain.uuid,
        })

    def results(self):
        """Output collected results."""
        nested_results = {}

        for surface_result in self.ctx.surface_results:
            spacing_label = f"{surface_result['vacuum_spacing']:.6f}"
            nested_results[spacing_label] = {
                'iteration': surface_result['iteration'],
                'vacuum_spacing': surface_result['vacuum_spacing'],
                'structure_formula': surface_result['structure_formula'],
                'surface_multiplier': surface_result['surface_multiplier'],
                'total_energy_ev': surface_result['total_energy_ev'],
                'surface_energy_j_m2': surface_result['surface_energy_j_m2'],
                'workchain_pk': surface_result['workchain_pk'],
                'workchain_uuid': surface_result['workchain_uuid'],
            }

        results = {
            'results': nested_results,
            'surface_area_angstrom2': float(self.ctx.surface_area),
            'number_of_spacings': self.ctx.number_of_spacings,
        }

        if 'total_energy_conventional_geometry' in self.ctx:
            results['conventional_energy_ev'] = float(self.ctx.total_energy_conventional_geometry)

        self.out('surface_results', orm.Dict(dict=results))
