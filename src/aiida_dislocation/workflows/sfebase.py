from tkinter import W
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_dislocation.tools import (
    is_primitive_cell, 
    calculate_surface_area, 
    get_faulted_structure,
    get_conventional_structure,
    get_cleavaged_structure,
)
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from ase.formula import Formula
from math import ceil

class SFEBaseWorkChain(ProtocolMixin, WorkChain):
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
            PwBaseWorkChain,
            namespace=cls._SFE_NAMESPACE,
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
            cls.setup,
            cls.generate_structures,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.setup_supercell_kpoints,
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
            (cls._SFE_NAMESPACE, PwBaseWorkChain),
            (cls._SURFACE_ENERGY_NAMESPACE, PwBaseWorkChain),
        ]:
            overrides = inputs.get(namespace, {})
            if namespace == cls._RELAX_NAMESPACE:
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

        builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints', None)
        builder[cls._RELAX_NAMESPACE]['base_relax'].pop('kpoints_distance', None)
        builder[cls._RELAX_NAMESPACE].pop('base_init_relax', None)

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
        self.report(f"Totel energy of unit cell after relaxation: {self.ctx.total_energy_unit_cell / self._RY2eV} Ry")

    def setup(self):

        if 'current_structure' not in self.ctx:
            self.ctx.current_structure = self.inputs.structure

    def _get_fault_type(self):
        """Return the fault type for this workchain. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_fault_type()")

    def generate_structures(self):
        """Generate all structures including conventional, cleavaged, and faulted. Common implementation for all subclasses."""
        fault_type = self._get_fault_type()
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

        # Get faulted structure (based on conventional cell)
        _, faulted_structure = get_faulted_structure(
            conventional_structure,
            fault_type=fault_type,
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )

        # Verify that the requested fault structure was generated
        if faulted_structure is None:
            self.report(f'{fault_type.capitalize()} fault structure is not available for this gliding system.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        # Store structures in context
        self.ctx.structures = AttributeDict({
            'conventional': conventional_structure,
            'cleavaged': cleavaged_structure,
            fault_type: faulted_structure,
        })

        self.ctx.surface_area = calculate_surface_area(conventional_structure.cell)
        
        self.report(f'Surface area of the conventional geometry: {self.ctx.surface_area} Angstrom^2')
        
        unit_cell_formula = Formula(self.ctx.current_structure.get_ase().get_chemical_formula())
        _, unit_cell_multiplier = unit_cell_formula.reduce()
        
        self.ctx.unit_cell_multiplier = unit_cell_multiplier

        conventional_formula = Formula(conventional_structure.get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        
        self.ctx.conventional_multiplier = conventional_multiplier

        surface_formula = Formula(cleavaged_structure.get_chemical_formula())
        _, surface_multiplier = surface_formula.reduce()
        
        self.ctx.surface_multiplier = surface_multiplier

    def _extract_faulted_structure(self, fault_structure_data, fault_type_name):
        """
        Extract the actual structure from fault structure data.
        
        Args:
            fault_structure_data: Tuple of (structure_data, fault_type) from get_faulted_structure
            fault_type_name: Name of the fault type for error messages
            
        Returns:
            Tuple of (actual_structure_ase, multiplier_name) where multiplier_name is like 'intrinsic_multiplier'
        """
        fault_structure, fault_type = fault_structure_data
        
        # Handle different fault types
        if fault_type == 'removal':
            # For removal type, fault_structure is a tuple: (structure, removed_layers)
            actual_structure = fault_structure[0]
        elif fault_type == 'gliding':
            # For gliding type, fault_structure is a list of tuples
            if not fault_structure:
                raise ValueError(f'{fault_type_name.capitalize()} gliding structure list is empty')
            actual_structure = fault_structure[0][0]  # Get first structure from first tuple
        else:
            raise ValueError(f'Unknown fault type: {fault_type}')
        
        return actual_structure

    def _get_kpoints_scf(self):
        """Get or create kpoints_scf. Returns kpoints_scf and its mesh."""
        if 'kpoints_scf' in self.ctx:
            kpoints_scf = self.ctx.kpoints_scf
        else:
            inputs = {
                'structure': orm.StructureData(
                    ase=self.ctx.structures.conventional
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

    def setup_supercell_kpoints(self):
        """
        Setup kpoints for supercell calculations.
        Common implementation that can be overridden by subclasses if needed.
        """
        fault_type = self._get_fault_type()
        
        # Get fault structure (guaranteed to exist after generate_structures)
        fault_structure_data = getattr(self.ctx.structures, fault_type)
        actual_structure = self._extract_faulted_structure(fault_structure_data, fault_type)
        
        current_structure = orm.StructureData(ase=actual_structure)
        
        fault_formula = Formula(actual_structure.get_chemical_formula())
        _, fault_multiplier = fault_formula.reduce()
        
        # Store multiplier with fault_type name
        setattr(self.ctx, f'{fault_type}_multiplier', fault_multiplier)
        
        # Get kpoints_scf
        _, kpoints_scf_mesh = self._get_kpoints_scf()
        
        # Calculate kpoints for SFE
        kpoints_sfe = orm.KpointsData()
        sfe_z_ratio = actual_structure.cell.cellpar()[2] / self.ctx.structures.conventional.cell.cellpar()[2]
        kpoints_sfe.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / sfe_z_ratio)])
        
        # Calculate kpoints for surface energy
        kpoints_surface_energy = orm.KpointsData()
        surface_energy_z_ratio = self.ctx.structures.cleavaged.cell.cellpar()[2] / self.ctx.structures.conventional.cell.cellpar()[2]
        kpoints_surface_energy.set_kpoints_mesh(
            kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / surface_energy_z_ratio)])
        
        self.ctx.kpoints_sfe = kpoints_sfe
        self.ctx.kpoints_surface_energy = kpoints_surface_energy
        self.ctx.current_structure = current_structure

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
            ase=self.ctx.structures.conventional
            )

        inputs.kpoints_distance = self.inputs.kpoints_distance

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} conventional geometry.')

        self.to_context(workchain_scf = running)

    def inspect_scf(self):
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f"scf {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f'PwBaseWorkChain<{self.ctx.workchain_scf.pk}> for conventional structure finished OK')
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SCF_NAMESPACE,
            ),
        )
        self.ctx.kpoints_scf = workchain.base.links.get_outgoing(
            link_label_filter='create_kpoints_from_distance'
            ).first().node.outputs.result

        self.ctx.total_energy_conventional_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f"Totel energy of conventional cell [{self.ctx.conventional_multiplier} unit cells]: {self.ctx.total_energy_conventional_geometry / self._RY2eV} Ry")

        if 'total_energy_unit_cell' in self.ctx:
            energy_difference = self.ctx.total_energy_conventional_geometry - self.ctx.total_energy_unit_cell / self.ctx.unit_cell_multiplier * self.ctx.conventional_multiplier
            self.report(f'Energy difference between conventional and unit cell: {energy_difference / self._RY2eV} Ry')

    def should_run_sfe(self):
        return self._SFE_NAMESPACE in self.inputs
    
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
            ase=self.ctx.structures.cleavaged
            )
        inputs.kpoints = self.ctx.kpoints_surface_energy

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')

        self.to_context(workchain_surface_energy = running)

    def inspect_surface_energy(self):

        workchain = self.ctx.workchain_surface_energy

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{workchain.pk}> for surface energy calculation failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY
        
        self.report(f'PwBaseWorkChain<{workchain.pk}> for surface energy calculation finished OK')

        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SURFACE_ENERGY_NAMESPACE,
            ),
        )

        total_energy_slab = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of the cleaved geometry [{self.ctx.surface_multiplier} unit cells]: {total_energy_slab / self._RY2eV} Ry')
        # SE = ( self.ctx.total_energy_unit_cell * self.ctx.multiplicity_cl - total_energy_slab ) * 2 / self.ctx.surface_area * self._eVA22Jm2
        # self.report(f'Surface energy: {SE.value} J/m^2')
        # self.ctx.SE = SE
        # Rice_ratio = self.ctx.USF / SE
        # self.report(f'Rice ratio: {Rice_ratio.value}')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = total_energy_slab - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.surface_multiplier
            surface_energy = energy_difference / 2 / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'Surface energy evaluated from conventional and unit cell: {surface_energy} J/m^2')

    def results(self):
        pass
