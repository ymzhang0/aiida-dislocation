from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from math import ceil

from aiida_dislocation.tools import (
    get_faulted_structure,
    calculate_surface_area, 
    get_conventional_structure,
    get_cleavaged_structure,
)
from ase.formula import Formula

from .mixins import (
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
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
    _SURFACE_ENERGY_NAMESPACE = "surface_energy"
    
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
            while_(cls.should_run_sfe)(
                cls.run_sfe,
                cls.inspect_sfe,
            ),
            if_(cls.should_run_surface_energy)(
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
            404,
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwBaseWorkChain` for the surface energy run failed.',
        )
        spec.exit_code(
            405,
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
            (cls._SFE_NAMESPACE, PwBaseWorkChain),
            (cls._SURFACE_ENERGY_NAMESPACE, PwBaseWorkChain),
        ]:
            overrides = inputs.get(namespace, {})

            if workchain_type == PwRelaxWorkChain:
                overrides['base_relax']['pseudo_family'] = inputs.get('pseudo_family', None)
                overrides['base_init_relax']['pseudo_family'] = inputs.get('pseudo_family', None)
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
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.gliding_plane = orm.Str(inputs.get('gliding_plane', ''))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

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
        inputs.base_relax.kpoints_distance = self.inputs.kpoints_distance
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
        
        # Get conventional structure
        strukturbericht, conventional_structure = get_conventional_structure(
            self.ctx.current_structure.get_ase(),
            gliding_plane=gliding_plane,
        )
        if strukturbericht:
            self.report(f'{strukturbericht} structure is detected.')
        else:
            self.report(f'Strukturbericht can not be detected.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED
        
        _, faulted_structure_data = get_faulted_structure(
            conventional_structure,
            fault_type='general',
            additional_spacing=0.0,
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
            fault_mode='general',
        )

        if faulted_structure_data is None or not faulted_structure_data.get('structures'):
            self.report(f'Faulted structure not available for spacing {current_spacing}. Skipping.')
            return False

        self.ctx.faulted_structure = faulted_structure_data['structures']

        # Get cleavaged structure (based on conventional cell)
        _, cleavaged_structure = get_cleavaged_structure(
            conventional_structure,
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )

        # Store structures directly in context
        self.ctx.conventional_structure = conventional_structure
        self.ctx.cleavaged_structure = cleavaged_structure

        self.ctx.surface_area = calculate_surface_area(conventional_structure.cell)
        
        self.report(f'Surface area of the conventional geometry: {self.ctx.surface_area} Angstrom^2')
        
        unit_cell_formula = Formula(self.ctx.current_structure.get_ase().get_chemical_formula())
        _, unit_cell_multiplier = unit_cell_formula.reduce()
        
        # Calculate and store multipliers using helper method
        self.ctx.unit_cell_multiplier = self._calculate_structure_multiplier(
            self.ctx.current_structure.get_ase()
        )
        self.ctx.conventional_multiplier = self._calculate_structure_multiplier(
            conventional_structure
        )
        self.ctx.surface_multiplier = self._calculate_structure_multiplier(
            cleavaged_structure
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
        self.ctx.number_of_structures = len(self.ctx.faulted_structure)
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

    def should_run_sfe(self):

        if self._SFE_NAMESPACE not in self.inputs:
            return False

        if self.ctx.faulted_structure == []:
            return False
        
        faulted_structure = self.ctx.faulted_structure.pop(0)
        self.ctx.current_structure = faulted_structure['structure']
        self.ctx.current_slipping_vector = faulted_structure['burger_vector']

        z_ratio = self.ctx.current_structure.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        kpoints_mesh = self.ctx.kpoints_scf.get_kpoints_mesh()[0]
        
        kpoints_sfe = orm.KpointsData()
        kpoints_sfe.set_kpoints_mesh(kpoints_mesh[:2] + [ceil(kpoints_mesh[2] / z_ratio)])
        
        self.ctx.kpoints_sfe = kpoints_sfe

        return True

    def run_sfe(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = f"structure_{self.ctx.iteration:02d}"

        inputs.pw.structure = orm.StructureData(ase=self.ctx.current_structure)
        inputs.kpoints = self.ctx.kpoints_sfe

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for faulted structure {self.ctx.iteration}/{self.ctx.number_of_structures}...')

        self.ctx.iteration += 1

        return {f"workchain_sfe": running}

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe
        if not workchain.is_finished_ok:
            self.report(f'PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(f'PwBaseWorkChain<{self.ctx.workchain_sfe.pk}> finished')

    def should_run_surface_energy(self):
        return self._SURFACE_ENERGY_NAMESPACE in self.inputs

    def run_surface_energy(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SURFACE_ENERGY_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._SURFACE_ENERGY_NAMESPACE
        inputs.pw.structure = orm.StructureData(
            ase=self.ctx.cleavaged_structure
            )
        inputs.kpoints = self.ctx.kpoints_surface_energy
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for cleavaged structure')
        return {f"workchain_surface_energy": running}

    def inspect_surface_energy(self):
        workchain = self.ctx.workchain_surface_energy
        if not workchain.is_finished_ok:
            self.report(f'PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

        self.report(f'PwBaseWorkChain<{self.ctx.workchain_surface_energy.pk}> finished')

        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SURFACE_ENERGY_NAMESPACE
            )
        )


    def results(self):
        """Output collected results."""
        self.report(f'Results')
