"""Mixins and helper classes for workflow organization."""

from aiida import orm
from ase.formula import Formula
from math import ceil


class StructureGenerationMixin:
    """Mixin for structure generation related methods."""
    
    def _calculate_structure_multiplier(self, structure):
        """Calculate the multiplier for a given structure.
        
        :param structure: ASE Atoms object
        :return: multiplier value
        """
        formula = Formula(structure.get_chemical_formula())
        _, multiplier = formula.reduce()
        return multiplier
    
    def _store_structure_multiplier(self, structure, multiplier_name):
        """Store structure and its multiplier in context.
        
        :param structure: ASE Atoms object
        :param multiplier_name: Name for the multiplier in context (e.g., 'intrinsic_multiplier')
        """
        multiplier = self._calculate_structure_multiplier(structure)
        setattr(self.ctx, multiplier_name, multiplier)
        return multiplier
    
    def _validate_faulted_structure(self, faulted_structure_data, fault_type):
        """Validate that a faulted structure was generated.
        
        :param faulted_structure_data: Result from get_faulted_structure
        :param fault_type: Type of fault (for error messages)
        :return: tuple (is_valid, error_code_or_none)
        """
        if faulted_structure_data is None:
            self.report(f'{fault_type.capitalize()} fault structure is not available for this gliding system.')
            return False, self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED
        
        structures = faulted_structure_data.get('structures', [])
        if not structures:
            self.report(f'{fault_type.capitalize()} fault structure list is empty.')
            return False, self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED
        
        first_entry = structures[0]
        actual_structure = first_entry.get('structure')
        
        if actual_structure is None:
            self.report(f'{fault_type.capitalize()} fault structure is missing structure data.')
            return False, self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED
        
        return True, None


class EnergyCalculationMixin:
    """Mixin for energy calculation related methods."""
    
    def _calculate_stacking_fault_energy(
        self,
        total_energy_faulted,
        fault_multiplier,
        fault_type_name
    ):
        """Calculate stacking fault energy from faulted and conventional geometries.
        
        :param total_energy_faulted: Total energy of faulted geometry
        :param fault_multiplier: Multiplier for faulted structure
        :param fault_type_name: Name of fault type (for reporting)
        :return: Stacking fault energy in J/m^2 or None if conventional energy not available
        """
        if 'total_energy_conventional_geometry' not in self.ctx:
            return None
        
        energy_difference = (
            total_energy_faulted
            - self.ctx.total_energy_conventional_geometry 
            / self.ctx.conventional_multiplier 
            * fault_multiplier
        )
        stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
        
        self.report(
            f'{fault_type_name} stacking fault energy evaluated from conventional geometry: '
            f'{stacking_fault_energy} J/m^2'
        )
        
        return stacking_fault_energy
    
    def _report_energy(self, energy, multiplier, structure_type, unit_cells_description):
        """Report energy in a consistent format.
        
        :param energy: Energy value
        :param multiplier: Multiplier value
        :param structure_type: Type of structure (for reporting)
        :param unit_cells_description: Description of unit cells
        """
        self.report(
            f'Total energy of {structure_type} [{multiplier} {unit_cells_description}]: '
            f'{energy / self._RY2eV} Ry'
        )


class KpointsSetupMixin:
    """Mixin for kpoints setup related methods."""
    
    def _calculate_kpoints_for_structure(self, structure, kpoints_scf):
        """Calculate kpoints mesh for a given structure based on z-ratio.
        
        :param structure: ASE Atoms object
        :param kpoints_scf_mesh: Base kpoints mesh from SCF calculation
        :return: KpointsData object
        """
        kpoints_scf_mesh = kpoints_scf.get_kpoints_mesh()[0]
        z_ratio = structure.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / z_ratio)])
        return kpoints
    
    def _setup_surface_energy_kpoints(self, kpoints_scf):
        """Setup kpoints for surface energy calculation.
        
        :param kpoints_scf_mesh: Base kpoints mesh from SCF calculation
        :return: KpointsData object for surface energy
        """
        return self._calculate_kpoints_for_structure(
            self.ctx.cleavaged_structure,
            kpoints_scf
        )


class WorkflowInspectionMixin:
    """Mixin for workflow inspection and error handling."""
    
    def _inspect_workchain(
        self,
        workchain,
        workchain_type_name,
        structure_type,
        exit_code_on_failure,
        namespace=None,
        workchain_class=None
    ):
        """Generic method to inspect a workchain and handle outputs.
        
        :param workchain: The workchain node to inspect
        :param workchain_type_name: Name of workchain type (for reporting)
        :param structure_type: Type of structure (for reporting)
        :param exit_code_on_failure: Exit code to return on failure
        :param namespace: Optional namespace for exposing outputs
        :param workchain_class: Optional workchain class for exposing outputs
        :return: Exit code if failed, None if successful
        """
        if not workchain.is_finished_ok:
            self.report(
                f"{workchain_type_name}<{workchain.pk}> for {structure_type} "
                f"failed with exit status {workchain.exit_status}"
            )
            return exit_code_on_failure
        
        self.report(
            f'{workchain_type_name}<{workchain.pk}> for {structure_type} finished OK'
        )
        
        if namespace and workchain_class:
            self.out_many(
                self.exposed_outputs(workchain, workchain_class, namespace=namespace)
            )
        
        return None
    
    def _get_workchain_energy(self, workchain):
        """Extract energy from workchain outputs.
        
        :param workchain: Workchain node
        :return: Energy value
        """
        return workchain.outputs.output_parameters.get('energy')

