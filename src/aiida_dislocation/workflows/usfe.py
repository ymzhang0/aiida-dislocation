from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida.engine import append_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida import orm
from ase.formula import Formula
from aiida_dislocation.tools import get_faulted_structure
from math import ceil

class USFEWorkChain(SFEBaseWorkChain):
    """USFE WorkChain"""

    _SFE_NAMESPACE = "usfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.input('additional_spacings', valid_type=orm.List, required=False, default=lambda: orm.List(list=[0.0]),
                    help='The additional spacing to add to the structure.')
        
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )

    @classmethod
    def get_builder_from_protocol(
            cls,
            code,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = super().get_builder_from_protocol(
            code, structure, protocol, overrides, **kwargs)
        builder.additional_spacings = orm.List(list=inputs.get('additional_spacings', [0.0]))
        return builder
    
    def _get_fault_type(self):
        """Return the fault type for USFE workchain."""
        return 'unstable'

    def setup(self):
        super().setup()
        self.ctx.iteration = 1
        self.ctx.additional_spacings = self.inputs.additional_spacings.get_list()
        

    def setup_supercell_kpoints(self):
        """
        Setup kpoints for USFE. 
        Note: current_structure and multiplier are set in should_run_sfe for each iteration.
        """
        # Get unstable fault structure
        if not hasattr(self.ctx, 'current_structure'):
            raise ValueError('Current structure not found in context.')
        
        # Calculate z_ratio for kpoints
        # current_structure is StructureData, need to get ASE atoms first
        current_structure_ase = self.ctx.current_structure.get_ase()
        sfe_z_ratio = current_structure_ase.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        
        # Get kpoints_scf
        _, kpoints_scf_mesh = self._get_kpoints_scf()
        
        # Calculate kpoints for SFE
        kpoints_sfe = orm.KpointsData()
        kpoints_sfe.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / sfe_z_ratio)])
        
        # Calculate kpoints for surface energy
        kpoints_surface_energy = orm.KpointsData()
        surface_energy_z_ratio = self.ctx.cleavaged_structure.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        kpoints_surface_energy.set_kpoints_mesh(
            kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / surface_energy_z_ratio)])
        
        self.ctx.kpoints_sfe = kpoints_sfe
        self.ctx.kpoints_surface_energy = kpoints_surface_energy

    def should_run_sfe(self):

        if not self._SFE_NAMESPACE in self.inputs:
            return False
        
        if self.ctx.additional_spacings == []:
            return False

        fault_type = self._get_fault_type()
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        
        self.ctx.current_spacing = self.ctx.additional_spacings.pop()

        self.report(f'Running USFE iteration {self.ctx.iteration} with additional spacing: {self.ctx.current_spacing}')
        # Get faulted structure (based on conventional cell)
        _, faulted_structure_data = get_faulted_structure(
            self.ctx.conventional_structure,
            fault_type=fault_type,
            additional_spacing=self.ctx.current_spacing,
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )
        
        # Verify that the requested fault structure was generated
        if faulted_structure_data is None:
            self.report(f'{fault_type.capitalize()} fault structure is not available for this gliding system.')
            return False

        # Extract the first structure from the fault data
        structures = faulted_structure_data.get('structures', [])
        if not structures:
            self.report(f'{fault_type.capitalize()} fault structure list is empty.')
            return False

        first_entry = structures[0]
        actual_structure = first_entry.get('structure')
        
        if actual_structure is None:
            self.report(f'{fault_type.capitalize()} fault structure is missing structure data.')
            return False

        # Store the ASE atoms object for later use
        self.ctx.current_structure_ase = actual_structure
        # Create StructureData from the ASE atoms object
        self.ctx.current_structure = orm.StructureData(ase=actual_structure)
        
        unstable_formula = Formula(actual_structure.get_chemical_formula())
        _, unstable_multiplier = unstable_formula.reduce()
        
        self.ctx.unstable_multiplier = unstable_multiplier

        return True
    
    def run_sfe(self):
        inputs = super().run_sfe()
        inputs.structure = self.ctx.current_structure
        inputs.base_relax.kpoints = self.ctx.kpoints_sfe
        inputs.metadata.call_link_label = f'usfe_{self.ctx.iteration}'
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for unstable stacking fault.')

        return {f"workchain_sfe_{self.ctx.iteration}": append_(running)}

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe
        self.ctx.iteration += 1
        if not workchain.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{workchain.pk}> for unstable faulted geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(
            f'PwRelaxWorkChain<{workchain.pk}> for unstable faulted geometry finished OK'
            )

        self.ctx.total_energy_usf_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of unstable faulted geometry [{self.ctx.unstable_multiplier} unit cells]: {self.ctx.total_energy_usf_geometry / self._RY2eV} Ry')
        
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = self.ctx.total_energy_usf_geometry - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.unstable_multiplier
            unstable_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'unstable stacking fault energy of evaluated from conventional geometry: {unstable_stacking_fault_energy} J/m^2')
