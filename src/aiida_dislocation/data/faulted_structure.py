from __future__ import annotations

import typing as ty
from copy import deepcopy

import numpy
from ase import Atoms

from aiida_dislocation.tools.structure_utils import group_by_layers
from aiida_dislocation.data.cleavaged_structure import CleavagedStructure, CleavagedStructureData
from aiida_dislocation.data.gliding_systems import (
    FaultConfig,
)
from aiida_dislocation.tools.structure_builder import (
    build_atoms_from_stacking_removal,
    build_atoms_from_stacking_mirror,
    build_atoms_from_burger_vector_with_vacuum,
    build_atoms_from_burger_vector_general,
    build_atoms_from_burger_vector,
    update_faults
)


class GeneralFaultStructurePoint(ty.TypedDict):
    """Single point on a generalized stacking fault path."""

    structure: Atoms
    burger_vector: list[float]
    direction_name: str
    path_index: int
    step_index: int


GeneralFaultStructureResult = list[GeneralFaultStructurePoint]
FaultedStructureResult = ty.Union[Atoms, list[dict[str, ty.Any]], GeneralFaultStructureResult]


class FaultedStructure(CleavagedStructure):
    """
    A class to handle dislocation structures and their manipulations using ASE Atoms.
    """
    
    def get_faulted_structure(self,
                            fault_mode: str,
                            fault_type: str,
                            additional_spacing: float = 0.0,
                            vacuum_ratio: float = 0.0,
                            print_info: bool = False,
                            **kwargs) -> ty.Optional[FaultedStructureResult]:
        """
        Generate faulted structure.
        Returns faulted structures for the requested mode.
        """
        if fault_mode not in ['removal', 'vacuum', 'general']:
            raise ValueError(f"fault_mode must be one of 'removal', 'vacuum', 'general', got '{fault_mode}'")

        if fault_mode == 'removal' and fault_type not in ['intrinsic', 'unstable', 'extrinsic']:
            raise ValueError(f"fault_type must be one of 'intrinsic', 'unstable', or 'extrinsic', got '{fault_type}'")

        if print_info:
            print(f'Strukturbericht {self.strukturbericht} detected')

        conventional_structure = self.get_conventional_structure()
        
        plane_config = self._prepare_plane_data()
        
        layers_dict = group_by_layers(conventional_structure)
        
        if len(layers_dict) != plane_config.n_layers:
            raise ValueError(
                f'Layer count mismatch: found {len(layers_dict)} layers, but expected {plane_config.n_layers}.'
            )

        fault_config = getattr(plane_config, fault_type)
        if not fault_config.possible:
            return None
            
        faulted_result = None

        # Removal Mode
        if fault_mode == 'removal' and fault_config.removal_layers is not None:
            structure = build_atoms_from_stacking_removal(
                conventional_structure,
                self.n_unit_cells,
                fault_config.removal_layers,
                layers_dict,
                additional_spacing=(fault_config.interface, additional_spacing),
                print_info=print_info
            )
            faulted_result = structure

        # Vacuum Mode
        if fault_mode == 'vacuum' and vacuum_ratio > 0.0 and fault_config.burger_vectors is not None:
            structures_list = []
            for burger_vector in fault_config.burger_vectors:
                structure = build_atoms_from_burger_vector_with_vacuum(
                    conventional_structure,
                    self.n_unit_cells,
                    burger_vector,
                    layers_dict,
                    vacuum_ratio=vacuum_ratio,
                    print_info=print_info
                )
                structures_list.append({
                    'structure': structure,
                    'burger_vector': burger_vector,
                })
            faulted_result = structures_list

        # General Mode
        if fault_mode == 'general' and fault_config.burger_vectors is not None:
            structures_list: list[GeneralFaultStructurePoint] = []
            nsteps = kwargs.get('nsteps', fault_config.nsteps)
            stacking_order = ''.join(layers_dict.keys())
            
            zs = [(value['z'] + layer) / self.n_unit_cells for layer in range(self.n_unit_cells) for value in layers_dict.values()]
            stacking_order_supercell = stacking_order * self.n_unit_cells

            new_cell = conventional_structure.cell.array.copy()
            new_cell[-1] *= self.n_unit_cells

            if isinstance(fault_config.burger_vectors, dict):
                for direction_name, path_points in fault_config.burger_vectors.items():
                    for path_index, segment in enumerate(path_points):
                        burgers_vector_for_cell = numpy.zeros(3)
                        faults = numpy.zeros((len(stacking_order_supercell), 3))
                        step_index = 0

                        # Initial state (0 displacement)
                        structure = build_atoms_from_burger_vector_general(
                            new_cell, deepcopy(zs), layers_dict, stacking_order_supercell,
                            burgers_vector_for_cell, faults, print_info=print_info
                        )
                        structures_list.append({
                            'structure': structure,
                            'burger_vector': burgers_vector_for_cell.tolist(),
                            'direction_name': direction_name,
                            'path_index': path_index,
                            'step_index': step_index,
                        })

                        for interface, burgers_vector in segment:
                            burgers_vector_step = numpy.array(burgers_vector) / nsteps
                            for _ in range(1, 1+nsteps):
                                step_index += 1
                                faults = update_faults(faults, interface, burgers_vector_step)
                                burgers_vector_for_cell += burgers_vector_step
                                structure = build_atoms_from_burger_vector_general(
                                    new_cell, deepcopy(zs), layers_dict, stacking_order_supercell,
                                    burgers_vector_for_cell, faults, print_info=print_info
                                )
                                structures_list.append({
                                    'structure': structure,
                                    'burger_vector': burgers_vector_for_cell.tolist(),
                                    'direction_name': direction_name,
                                    'path_index': path_index,
                                    'step_index': step_index,
                                })
                                
            faulted_result = structures_list
            
        return faulted_result

    def _build_faulted_structure_helper(
        self,
        config: FaultConfig,
        ase_atoms_t: Atoms,
        layers_dict: dict[str, dict[str, ty.Any]],
        print_info: bool = False,
    ) -> ty.Optional[FaultedStructureResult]:
        """Internal helper for unstable/intrinsic fault building."""
        if not config.possible:
            return None
        
        if config.removal_layers is not None:
            structure = build_atoms_from_stacking_removal(
                ase_atoms_t, self.n_unit_cells, config.removal_layers, layers_dict,
                additional_spacing=(config.interface, 0.0), print_info=print_info
            )
            return structure
        
        if config.burger_vectors is not None and isinstance(config.burger_vectors, list):
            structures_list = []
            for bv in config.burger_vectors:
                structure = build_atoms_from_burger_vector(
                    ase_atoms_t, self.n_unit_cells, bv, layers_dict, print_info=print_info
                )
                structures_list.append({
                    'structure': structure,
                    'burger_vector': bv,
                })
            return structures_list
        return None

class FaultedStructureData(CleavagedStructureData, FaultedStructure):
    """AiiDA data node embedding faulted-structure logic on top of cleavaged-structure data."""
