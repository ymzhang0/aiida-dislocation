from .sfebase import SFEBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from ase.formula import Formula
from aiida import orm
from aiida_dislocation.tools import get_faulted_structure

class ESFEWorkChain(SFEBaseWorkChain):
    """ESFE WorkChain"""
    
    _SFE_NAMESPACE = "esfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.input('additional_spacing', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.0),
                    help='The additional spacing to add to the structure.')
        
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_ESF",
            message='The `PwBaseWorkChain` for the ESF run failed.',
        )

    def _get_fault_type(self):
        """Return the fault type for ESFE workchain."""
        return 'extrinsic'

    def generate_faulted_structure(self):
        """Generate extrinsic faulted structure."""
        fault_type = self._get_fault_type()
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        additional_spacing = self.inputs.get('additional_spacing', orm.Float(0.0)).value
        
        # Get faulted structure (based on conventional cell)
        _, faulted_structure_data = get_faulted_structure(
            self.ctx.conventional_structure,
            fault_type=fault_type,
            additional_spacing=additional_spacing,
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )

        # Verify that the requested fault structure was generated
        if faulted_structure_data is None:
            self.report(f'{fault_type.capitalize()} fault structure is not available for this gliding system.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        # Extract the first structure from the fault data
        structures = faulted_structure_data.get('structures', [])
        if not structures:
            self.report(f'{fault_type.capitalize()} fault structure list is empty.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        first_entry = structures[0]
        actual_structure = first_entry.get('structure')
        
        if actual_structure is None:
            self.report(f'{fault_type.capitalize()} fault structure is missing structure data.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        # Store faulted structure directly in context
        self.ctx.extrinsic_structure = actual_structure
        
        # Store fault metadata if needed
        mode = faulted_structure_data.get('mode')
        if mode == 'removal':
            self.ctx.removed_layers = first_entry.get('layers')
        elif mode == 'gliding':
            self.ctx.current_burger_vector = first_entry.get('burger_vector')
        
        return None

    def should_run_sfe(self):

        if not self._SFE_NAMESPACE in self.inputs:
            return False
        
        # Check if extrinsic fault structure is available
        if not hasattr(self.ctx, 'extrinsic_structure') or self.ctx.extrinsic_structure is None:
            self.report('Extrinsic fault structure is not available. Skipping ESFE calculation.')
            return False
        
        current_structure = orm.StructureData(ase=self.ctx.extrinsic_structure)
        self.ctx.current_structure = current_structure
        
        extrinsic_formula = Formula(self.ctx.extrinsic_structure.get_chemical_formula())
        _, extrinsic_multiplier = extrinsic_formula.reduce()
        
        self.ctx.extrinsic_multiplier = extrinsic_multiplier

        return True

    def run_sfe(self):

        inputs = super().run_sfe()
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for extrinsic stacking fault.')

        self.to_context(workchain_sfe = running)

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe

        if not workchain.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{workchain.pk}> for extrinsic faulted geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ESF

        self.report(
            f'PwRelaxWorkChain<{workchain.pk}> for extrinsic faulted geometry finished OK'
            )
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._SFE_NAMESPACE,
            ),
        )

        self.ctx.total_energy_esf_geometry = self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of extrinsic faulted geometry [{self.ctx.extrinsic_multiplier} unit cells]: {self.ctx.total_energy_esf_geometry / self._RY2eV} Ry')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = self.ctx.total_energy_esf_geometry - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.extrinsic_multiplier
            extrinsic_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'extrinsic stacking fault energy of evaluated from conventional geometry: {extrinsic_stacking_fault_energy} J/m^2')
        # energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        # isfe = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        # self.ctx.isfe = isfe
        # self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')
