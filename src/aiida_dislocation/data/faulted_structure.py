from __future__ import annotations

import typing as ty
from copy import deepcopy

import numpy
from aiida.common.exceptions import ModificationNotAllowed
from aiida.orm import Data
from ase import Atoms
from ase.build import make_supercell
from ase.io.jsonio import decode as ase_decode
from ase.io.jsonio import encode

from aiida_dislocation.tools.structure_utils import (
    get_strukturbericht,
    group_by_layers,
)
from aiida_dislocation.data.gliding_systems import (
    GlidingSystem,
    GlidingPlaneConfig,
    get_gliding_system,
    FaultConfig,
)
from aiida_dislocation.tools.structure_builder import (
    build_atoms_surface,
    build_atoms_from_stacking_removal,
    build_atoms_from_stacking_mirror,
    build_atoms_from_burger_vector_with_vacuum,
    build_atoms_from_burger_vector_general,
    build_atoms_from_burger_vector,
    update_faults
)

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class GeneralFaultStructurePoint(ty.TypedDict):
    """Single point on a generalized stacking fault path."""

    structure: Atoms
    burger_vector: list[float]
    direction_name: str
    path_index: int
    step_index: int


GeneralFaultStructureResult = list[GeneralFaultStructurePoint]
FaultedStructureResult = ty.Union[Atoms, list[dict[str, ty.Any]], GeneralFaultStructureResult]


class FaultedStructure:
    """
    A class to handle dislocation structures and their manipulations using ASE Atoms.
    """
    
    def __init__(
        self,
        ase_atoms: Atoms,
        n_unit_cells: int,
        gliding_plane: ty.Optional[str] = None,
    ) -> None:
        """
        Initialize with an ASE Atoms object (assumed unit cell).
        """
        self._ase_atoms = ase_atoms
        self._n_unit_cells = n_unit_cells
        if not gliding_plane:
            self._gliding_plane = self.gliding_system.default_plane
        else:
            self._gliding_plane = gliding_plane

    @property   
    def unit_cell(self) -> Atoms:
        """Get the original unit cell structure."""
        return self._ase_atoms

    @property
    def n_unit_cells(self) -> int:
        """Get the number of repeated unit cells."""
        return self._n_unit_cells

    @property
    def gliding_plane(self) -> str:
        """Get the stored gliding plane."""
        return self._gliding_plane

    @property
    def strukturbericht(self) -> str:
        """Get the Strukturbericht designation."""
        # get_strukturbericht in tools/structure_utils.py takes "atoms_to_check"
        # It uses AseAtomsAdaptor.get_structure(atoms_to_check), so it works with ASE Atoms!
        strukturbericht = get_strukturbericht(self.unit_cell)
        if not strukturbericht:
            raise ValueError('No match found in the provided list of prototypes.')
        return strukturbericht

    @property
    def gliding_system(self) -> GlidingSystem:
        """Get the GlidingSystem instance."""
        gliding_system = get_gliding_system(self.strukturbericht)
        if not gliding_system:
            raise ValueError('No match found in the provided list of prototypes.')
        return gliding_system
    
    @property
    def is_primitive(self) -> bool:
        """Check if the unit cell is primitive."""
        pmg_struct = AseAtomsAdaptor.get_structure(self.unit_cell)
        prim_pmg = pmg_struct.get_primitive_structure()
        return pmg_struct.composition == prim_pmg.composition
    
    @property
    def wyckoff_elements(self) -> dict[str, str]:
        """Get Wyckoff symbols for ASE atoms."""
        pmg_struct = AseAtomsAdaptor.get_structure(self.unit_cell)
        sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-5)
        symmetrized = sga.get_symmetrized_structure()
        return {w: e.symbol for w, e in zip(symmetrized.wyckoff_letters, symmetrized.elements)}

    @property
    def surface_area(self) -> float:
        """Calculate surface area of the unit cell (XY plane)."""
        cell = self.get_conventional_structure().cell
        # Assuming surface is defined by vector 0 and 1 (standard for this package)
        return numpy.linalg.norm(numpy.cross(cell[0], cell[1]))
    
    def _get_effective_gliding_plane(self) -> str:
        """Return the configured gliding plane or the default for the detected gliding system."""
        return self.gliding_plane or self.gliding_system.default_plane

    def _prepare_plane_data(self) -> GlidingPlaneConfig:
        """Helper to get plane config for the effective gliding plane."""
        return self.gliding_system.get_plane(self._get_effective_gliding_plane())

    def get_conventional_structure(
        self,
        P: ty.Optional[ty.Union[list[ty.Any], 'numpy.ndarray']] = None,
        print_info: bool = False,
    ) -> Atoms:
        """
        Generate conventional structure.
        """
        if print_info:
            print(f'Strukturbericht {self.strukturbericht} detected')
            
        plane_config = self._prepare_plane_data()

        if P is None:
            P = plane_config.transformation_matrix
        else:
            P = numpy.array(P)

        ase_atoms_conventional = make_supercell(self.unit_cell, P)
        
        return ase_atoms_conventional

    def get_cleavaged_structure(
        self,
        vacuum_spacing: float = 1.0,
        print_info: bool = False,
    ) -> Atoms:
        """
        Generate cleavaged surface structure from a conventional structure.
        """
        if print_info:
            print(f'Strukturbericht {self.strukturbericht} detected')

        conventional_structure = self.get_conventional_structure()
        plane_config = self._prepare_plane_data()
        
        layers_dict = group_by_layers(conventional_structure)
        
        if len(layers_dict) != plane_config.n_layers:
            raise ValueError(
                f'Layer count mismatch: found {len(layers_dict)} layers, but expected {plane_config.n_layers} for '
                f'{self.strukturbericht} with gliding plane {self._get_effective_gliding_plane()}.'
            )
            
        cleavaged_atoms = build_atoms_surface(
            conventional_structure, self.n_unit_cells, layers_dict, print_info=print_info,
            vacuum_spacing=vacuum_spacing
        )
        return cleavaged_atoms

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

class FaultedStructureData(Data, FaultedStructure):
    """
    AiiDA Data class embedding FaultedStructure logic.
    Serialized ASE atoms are stored in attributes.
    """

    ASE_ATOMS_KEY = 'ase_atoms_json'
    N_UNIT_CELLS_KEY = 'n_unit_cells'
    GLIDING_PLANE_KEY = 'gliding_plane'
    STRUKTURBERICHT_KEY = 'strukturbericht'

    def __init__(
        self,
        ase: ty.Optional[Atoms] = None,
        n_unit_cells: ty.Optional[int] = None,
        gliding_plane: ty.Optional[str] = None,
        strukturbericht: ty.Optional[str] = None,
        **kwargs: ty.Any,
    ) -> None:
        """
        Initialize AiiDA Data node.
        Pass `ase`, `n_unit_cells`, and optional metadata before storing.
        """
        super().__init__(**kwargs)
        if ase is None:
            return

        if n_unit_cells is None:
            raise ValueError('`n_unit_cells` must be provided when initializing `FaultedStructureData` with `ase`.')

        resolved_strukturbericht = strukturbericht or get_strukturbericht(ase)
        if resolved_strukturbericht is None:
            raise ValueError('Failed to detect `strukturbericht` from the provided ASE structure.')

        resolved_gliding_system = get_gliding_system(resolved_strukturbericht)
        if resolved_gliding_system is None:
            raise ValueError(f'No gliding system found for Strukturbericht `{resolved_strukturbericht}`.')

        resolved_gliding_plane = gliding_plane or resolved_gliding_system.default_plane

        self.set_ase(ase)
        self._set_attribute(self.N_UNIT_CELLS_KEY, int(n_unit_cells))
        self._set_attribute(self.GLIDING_PLANE_KEY, resolved_gliding_plane)
        self._set_attribute(self.STRUKTURBERICHT_KEY, resolved_strukturbericht)

    def _set_attribute(self, key: str, value: ty.Any) -> None:
        """Set an attribute before storing the node."""
        if self.is_stored:
            raise ModificationNotAllowed('`FaultedStructureData` attributes cannot be modified after storing.')
        self.base.attributes.set(key, value)

    def set_ase(self, ase_atoms: Atoms) -> None:
        """Set the ASE atoms content, serializing to JSON."""
        self._set_attribute(self.ASE_ATOMS_KEY, encode(ase_atoms))
        
    @property
    def unit_cell(self) -> Atoms:
        """Retrieve ASE atoms from attributes."""
        json_str = self.base.attributes.get(self.ASE_ATOMS_KEY, None)
        if json_str is None:
            raise AttributeError('No ASE atoms set for this FaultedStructureData.')
        return ase_decode(json_str)

    def get_ase(self) -> Atoms:
        """Return the stored structure as ASE atoms."""
        return self.unit_cell

    @property
    def n_unit_cells(self) -> int:
        """Return the stored number of repeated unit cells."""
        return int(self.base.attributes.get(self.N_UNIT_CELLS_KEY))

    @property
    def gliding_plane(self) -> str:
        """Return the stored gliding plane."""
        return self.base.attributes.get(self.GLIDING_PLANE_KEY)

    @property
    def strukturbericht(self) -> str:
        """Return the stored Strukturbericht designation."""
        return self.base.attributes.get(self.STRUKTURBERICHT_KEY)

    @property
    def gliding_system(self) -> GlidingSystem:
        """Resolve the gliding system from the stored Strukturbericht."""
        gliding_system = get_gliding_system(self.strukturbericht)
        if gliding_system is None:
            raise ValueError(f'No gliding system found for Strukturbericht `{self.strukturbericht}`.')
        return gliding_system

    @property
    def structure_data(self) -> 'orm.StructureData':
        """Return the stored unit cell as AiiDA `StructureData`."""
        from aiida import orm

        return orm.StructureData(ase=self.unit_cell)
