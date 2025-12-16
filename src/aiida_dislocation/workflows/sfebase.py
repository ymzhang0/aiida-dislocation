from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_dislocation.tools import (
    calculate_surface_area, 
    get_faulted_structure,
    get_conventional_structure,
    get_cleavaged_structure,
)
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from ase.formula import Formula
from math import ceil
import numpy

from .mixins import (
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
)


class SFEBaseWorkChain(
    ProtocolMixin,
    StructureGenerationMixin,
    EnergyCalculationMixin,
    KpointsSetupMixin,
    WorkflowInspectionMixin,
    WorkChain
):
    """SFEBase WorkChain"""

    _NAMESPACE = 'sfebase'
    _RELAX_NAMESPACE = "relax"
    _SCF_NAMESPACE = "scf"
    _SFE_NAMESPACE = "sfe"
    _SURFACE_ENERGY_NAMESPACE = "surface_energy"

    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
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
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._RELAX_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
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
                'kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._SFE_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._SURFACE_ENERGY_NAMESPACE,
            exclude=(
                'pw.structure',
                'clean_workdir',
                'kpoints',
                'kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
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
                cls.setup_supercell_kpoints,
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
            PwRelaxWorkChain,
            namespace=cls._SFE_NAMESPACE,
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
            message='The `PwBaseWorkChain` for the relax run failed.',
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message='The `PwBaseWorkChain` for the scf run failed.',
        )
        spec.exit_code(
            403,
            "ERROR_NO_STRUCTURE_TYPE_DETECTED",
            message='The structure type can not be detected.',
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwBaseWorkChain` for the surface energy calculation failed.',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._SFE_NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml
        from . import protocols

        path = files(protocols) / f"{cls._SFE_NAMESPACE}.yaml"
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
            (cls._SFE_NAMESPACE, PwRelaxWorkChain),
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
            # sub_builder.pop('structure', None)
            sub_builder.pop('clean_workdir', None)

            if namespace != cls._RELAX_NAMESPACE:
                sub_builder.pop('kpoints', None)
                sub_builder.pop('kpoints_distance', None)

            builder[namespace]._data = sub_builder._data

        builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints', None)
        builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints_distance', None)
        builder[cls._RELAX_NAMESPACE].pop('base_init_relax', None)
        builder[cls._SFE_NAMESPACE].pop('base_init_relax', None)

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
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} unit cell geometry.')

        self.to_context(workchain_relax = running)

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{workchain.pk}> for {self.inputs.structure.get_formula()} unit cell geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'PwRelaxWorkChain<{workchain.pk}> for {self.inputs.structure.get_formula()} unit cell geometry finished OK')

        self.ctx.current_structure = workchain.outputs.output_structure
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._RELAX_NAMESPACE,
            ),
        )
        self.ctx.total_energy_unit_cell = workchain.outputs.output_parameters.get('energy')
        self.report(f"Total energy of unit cell after relaxation: {self.ctx.total_energy_unit_cell / self._RY2eV} Ry")

    def _get_fault_type(self):
        """Return the fault type for this workchain. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_fault_type()")

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
        """Get or create kpoints_scf. Returns kpoints_scf and its mesh."""
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
        
        kpoints_scf_mesh = kpoints_scf.get_kpoints_mesh()[0]
        return kpoints_scf, kpoints_scf_mesh

    def setup(self):
        """
        Setup kpoints for supercell calculations.
        Common implementation that can be overridden by subclasses if needed.
        """
        # Get fault structure (should be set by generate_faulted_structure in subclasses)
        fault_type = self._get_fault_type()
        fault_structure_attr = f'{fault_type}_structure'
        
        # if not hasattr(self.ctx, fault_structure_attr):
        #     raise ValueError(f'{fault_type.capitalize()} fault structure not found in context. '
        #                    f'Make sure generate_faulted_structure() was called.')
        
        # actual_structure = getattr(self.ctx, fault_structure_attr)
        
        # current_structure = orm.StructureData(ase=actual_structure)
        
        # fault_formula = Formula(actual_structure.get_chemical_formula())
        # _, fault_multiplier = fault_formula.reduce()
        
        # Store multiplier with fault_type name
        # setattr(self.ctx, f'{fault_type}_multiplier', fault_multiplier)
        
        # Get kpoints_scf
        _, kpoints_scf_mesh = self._get_kpoints_scf()
        
        # Calculate kpoints for surface energy using helper method
        self.ctx.kpoints_surface_energy = self._setup_surface_energy_kpoints(kpoints_scf_mesh)
        # self.ctx.current_structure = current_structure

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

        inputs.kpoints_distance = self.inputs.kpoints_distance

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} conventional geometry.')

        return {f"workchain_scf": running}

    def inspect_scf(self):
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf
        
        # Use helper method for inspection
        exit_code = self._inspect_workchain(
            workchain,
            'PwBaseWorkChain',
            'conventional structure',
            self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF,
            namespace=self._SCF_NAMESPACE,
            workchain_class=PwBaseWorkChain
        )
        if exit_code:
            return exit_code
        
        # Extract kpoints from workchain
        self.ctx.kpoints_scf = workchain.base.links.get_outgoing(
            link_label_filter='create_kpoints_from_distance'
        ).first().node.outputs.result

        # Extract and report energy
        self.ctx.total_energy_conventional_geometry = self._get_workchain_energy(workchain)
        self._report_energy(
            self.ctx.total_energy_conventional_geometry,
            self.ctx.conventional_multiplier,
            'conventional cell',
            'unit cells'
        )

        # Report energy difference if unit cell energy available
        if 'total_energy_unit_cell' in self.ctx:
            energy_difference = (
                self.ctx.total_energy_conventional_geometry 
                - self.ctx.total_energy_unit_cell 
                / self.ctx.unit_cell_multiplier 
                * self.ctx.conventional_multiplier
            )
            self.report(
                f'Energy difference between conventional and unit cell: '
                f'{energy_difference / self._RY2eV} Ry'
            )

    def run_sfe(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._SFE_NAMESPACE
                )
            )

        inputs.structure = self.ctx.current_structure
        inputs.base_relax.kpoints = self.ctx.kpoints_sfe
        settings = inputs.base_relax.pw.settings.get_dict()
        settings['USE_FRACTIONAL'] = True

        FIXED_COORDS = numpy.full_like( 
            self.ctx.current_structure.get_ase().get_positions(),
            fill_value=True,
            dtype=bool
            )

        settings['FIXED_COORDS'] = FIXED_COORDS.tolist()

        inputs.base_relax.pw.settings = orm.Dict(settings)

        return inputs

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
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')

        return {f"workchain_surface_energy": running}

    def inspect_surface_energy(self):
        """Verify that the surface energy calculation successfully finished."""
        workchain = self.ctx.workchain_surface_energy
        
        # Use helper method for inspection
        exit_code = self._inspect_workchain(
            workchain,
            'PwBaseWorkChain',
            'surface energy calculation',
            self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY,
            namespace=self._SURFACE_ENERGY_NAMESPACE,
            workchain_class=PwBaseWorkChain
        )
        if exit_code:
            return exit_code

        # Extract and report energy
        total_energy_slab = self._get_workchain_energy(workchain)
        self._report_energy(
            total_energy_slab,
            self.ctx.surface_multiplier,
            'cleaved geometry',
            'unit cells'
        )
        
        # Calculate surface energy
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = (
                total_energy_slab 
                - self.ctx.total_energy_conventional_geometry 
                / self.ctx.conventional_multiplier 
                * self.ctx.surface_multiplier
            )
            surface_energy = energy_difference / 2 / self.ctx.surface_area * self._eVA22Jm2
            self.report(
                f'Surface energy evaluated from conventional geometry: {surface_energy} J/m^2'
            )

    def results(self):
        pass
