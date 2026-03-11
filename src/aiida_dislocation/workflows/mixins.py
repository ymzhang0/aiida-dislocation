"""Mixins and helper classes for workflow organization."""

from __future__ import annotations

import typing as ty

from aiida import orm
from aiida.engine import ExitCode
from ase import Atoms
from ase.formula import Formula
from math import ceil


class StructureGenerationMixin:
    """Mixin for structure generation related methods."""

    @staticmethod
    def _ensure_ase_structure(structure: orm.StructureData | Atoms) -> Atoms:
        """Return an ASE structure from either ASE Atoms or `StructureData`."""
        return structure.get_ase() if isinstance(structure, orm.StructureData) else structure
    
    def _calculate_structure_multiplier(self, structure: orm.StructureData | Atoms) -> int:
        """Calculate the multiplier for a given structure.
        
        :param structure: ASE Atoms object
        :return: multiplier value
        """
        ase_structure = self._ensure_ase_structure(structure)
        formula = Formula(ase_structure.get_chemical_formula())
        _, multiplier = formula.reduce()
        return multiplier
    
    def _store_structure_multiplier(self, structure: orm.StructureData | Atoms, multiplier_name: str) -> int:
        """Store structure and its multiplier in context.
        
        :param structure: ASE Atoms object
        :param multiplier_name: Name for the multiplier in context (e.g., 'intrinsic_multiplier')
        """
        multiplier = self._calculate_structure_multiplier(structure)
        setattr(self.ctx, multiplier_name, multiplier)
        return multiplier
    
    def _validate_faulted_structure(
        self,
        faulted_structure_data: ty.Optional[dict[str, ty.Any]],
        fault_type: str,
    ) -> tuple[bool, ty.Optional[ExitCode]]:
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

    def _get_physical_constant(self, name: str) -> float:
        """Return a required physical constant from the workchain."""
        if not hasattr(self, name):
            raise AttributeError(f'`{type(self).__name__}` must define `{name}` to use energy mixin helpers.')
        return getattr(self, name)
    
    def _calculate_stacking_fault_energy(
        self,
        total_energy_faulted: float,
        fault_multiplier: int,
        fault_type_name: str
    ) -> ty.Optional[float]:
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
        stacking_fault_energy = energy_difference / self.ctx.surface_area * self._get_physical_constant('_eVA22Jm2')
        
        self.report(
            f'{fault_type_name} stacking fault energy evaluated from conventional geometry: '
            f'{stacking_fault_energy} J/m^2'
        )
        
        return stacking_fault_energy

    def _calculate_surface_energy(self, total_energy_slab: float, surface_multiplier: int) -> ty.Optional[float]:
        """Calculate a two-surface slab energy in J/m^2."""
        if 'total_energy_conventional_geometry' not in self.ctx:
            return None

        energy_difference = (
            total_energy_slab
            - self.ctx.total_energy_conventional_geometry
            / self.ctx.conventional_multiplier
            * surface_multiplier
        )
        return energy_difference / (2 * self.ctx.surface_area) * self._get_physical_constant('_eVA22Jm2')
    
    def _report_energy(
        self,
        energy: float,
        multiplier: int,
        structure_type: str,
        unit_cells_description: str,
    ) -> None:
        """Report energy in a consistent format.
        
        :param energy: Energy value
        :param multiplier: Multiplier value
        :param structure_type: Type of structure (for reporting)
        :param unit_cells_description: Description of unit cells
        """
        self.report(
            f'Total energy of {structure_type} [{multiplier} {unit_cells_description}]: '
            f'{energy / self._get_physical_constant("_RY2eV")} Ry'
        )


class KpointsSetupMixin:
    """Mixin for kpoints setup related methods."""
    
    def _calculate_kpoints_for_structure(
        self,
        structure: orm.StructureData | Atoms,
        kpoints_scf: orm.KpointsData,
    ) -> orm.KpointsData:
        """Calculate kpoints mesh for a given structure based on z-ratio.
        
        :param structure: ASE Atoms object
        :param kpoints_scf_mesh: Base kpoints mesh from SCF calculation
        :return: KpointsData object
        """
        kpoints_scf_mesh = kpoints_scf.get_kpoints_mesh()[0]
        structure_ase = StructureGenerationMixin._ensure_ase_structure(structure)
        conventional_ase = StructureGenerationMixin._ensure_ase_structure(self.ctx.conventional_structure)
        z_ratio = structure_ase.cell.cellpar()[2] / conventional_ase.cell.cellpar()[2]
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / z_ratio)])
        return kpoints
    
    def _setup_surface_energy_kpoints(self, kpoints_scf: orm.KpointsData) -> orm.KpointsData:
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
        workchain: orm.ProcessNode,
        workchain_type_name: str,
        structure_type: str,
        exit_code_on_failure: ExitCode,
        namespace: ty.Optional[str] = None,
        workchain_class: ty.Optional[type] = None
    ) -> ty.Optional[ExitCode]:
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
    
    def _get_workchain_energy(self, workchain: orm.ProcessNode) -> float:
        """Extract energy from workchain outputs.
        
        :param workchain: Workchain node
        :return: Energy value
        """
        return float(workchain.outputs.output_parameters.get('energy'))


def clean_calcjob_remote(node: orm.CalcJobNode) -> bool:
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False

    try:
        node.outputs.remote_folder._clean()  # noqa: SLF001
        cleaned = True
    except (OSError, KeyError, RuntimeError):
        pass

    return cleaned


def clean_workchain_calcs(workchain: orm.WorkChainNode) -> list[int]:
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs: list[int] = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode) and clean_calcjob_remote(called_descendant):
            cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs
