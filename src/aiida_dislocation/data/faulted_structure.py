import typing as ty
import numpy
from math import ceil
from ase import Atoms
from ase.build import make_supercell
from copy import deepcopy
from aiida.orm import Data
from ase.io.jsonio import encode, decode as ase_decode

from aiida_dislocation.tools.structure_utils import (
    get_strukturbericht,
    group_by_layers,
    # is_primitive_cell, # Requires StructureData? Let's check or reimplement for ASE
    get_elements_for_wyckoff_symbols as tool_get_elements_for_wyckoff_symbols, # Might require StructureData
    # get_kpoints_mesh_for_supercell, # Helper for AiiDA Kpoints, maybe keep but takes primitive types?
    calculate_surface_area as tool_calculate_surface_area,
    AttributeDict
)
from aiida_dislocation.data.gliding_systems import (
    GlidingSystem,
    get_gliding_system,
    FaultConfig
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

# Implementation Note: 
# tools.is_primitive_cell and tools.get_elements_for_wyckoff_symbols in structure_utils.py 
# currently take orm.StructureData. If I cannot change them easily without breaking tools,
# I might need to adapt them here or provide alternative implementations for ASE atoms.
# Since user said "remove aiida relation", I should probably reimplement minimal checks for ASE atoms
# or assume structure_utils can handle ASE atoms (I need to check structure_utils).

# Checked structure_utils previously:
# is_primitive_cell(structure: orm.StructureData) -> calls get_pymatgen()
# get_elements_for_wyckoff_symbols(structure: orm.StructureData) -> calls get_pymatgen_structure()
# So valid concern. I will import adapters or rewrite small helpers using pymatgen directly if needed.

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class FaultedStructure:
    """
    A class to handle dislocation structures and their manipulations using ASE Atoms.
    """
    
    def __init__(
        self, ase_atoms: Atoms,
        n_unit_cells: int,
        gliding_plane: str = None,
        ):
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
    def wyckoff_elements(self) -> dict:
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
    
    def _prepare_plane_data(self):
        """Helper to get plane config and default plane."""
        if not self._gliding_plane:
            self._gliding_plane = self.gliding_system.default_plane
        
        plane_config = self.gliding_system.get_plane(self._gliding_plane)
        return plane_config

    def get_conventional_structure(self, 
                                 P: ty.Optional[ty.Union[list, 'numpy.ndarray']] = None,
                                 print_info: bool = False) -> Atoms:
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

    def get_cleavaged_structure(self, vacuum_spacing: float = 1.0,
                              print_info: bool = False) -> Atoms:
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
                f'{self.strukturbericht} with gliding plane {gliding_plane}.'
            )
            
        cleavaged_atoms = build_atoms_surface(
            conventional_structure, self._n_unit_cells, layers_dict, print_info=print_info,
            vacuum_spacing=vacuum_spacing
        )
        return cleavaged_atoms

    def get_faulted_structure(self,
                            fault_mode: str,
                            fault_type: str,
                            additional_spacing: float = 0.0,
                            vacuum_ratio: float = 0.0,
                            print_info: bool = False,
                            **kwargs) -> ty.Optional[Atoms]:
        """
        Generate faulted structure.
        Returns: (result_dict containing ASE Atoms)
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
                self._n_unit_cells,
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
                    self._n_unit_cells,
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
            structures_list = []
            nsteps = kwargs.get('nsteps', fault_config.nsteps)
            stacking_order = ''.join(layers_dict.keys())
            
            zs = [(value['z'] + layer)/self._n_unit_cells for layer in range(self._n_unit_cells) for value in layers_dict.values()]
            stacking_order_supercell = stacking_order * self._n_unit_cells

            new_cell = conventional_structure.cell.array.copy()
            new_cell[-1] *= (self._n_unit_cells)

            if isinstance(fault_config.burger_vectors, dict):
                for direction_name, path_points in fault_config.burger_vectors.items():
                    for segment in path_points:
                        burgers_vector_for_cell = numpy.zeros(3)
                        faults = numpy.zeros((len(stacking_order_supercell), 3))
                        
                        # Initial state (0 displacement)
                        structure = build_atoms_from_burger_vector_general(
                            new_cell, deepcopy(zs), layers_dict, stacking_order_supercell,
                            burgers_vector_for_cell, faults, print_info=print_info
                        )
                        structures_list.append({
                            'structure': structure,
                            'burger_vector': burgers_vector_for_cell.tolist(),
                        })
                        
                        for interface, burgers_vector in segment:
                            burgers_vector_step = numpy.array(burgers_vector) / nsteps
                            for _ in range(1, 1+nsteps):
                                faults = update_faults(faults, interface, burgers_vector_step)
                                burgers_vector_for_cell += burgers_vector_step
                                structure = build_atoms_from_burger_vector_general(
                                    new_cell, deepcopy(zs), layers_dict, stacking_order_supercell,
                                    burgers_vector_for_cell, faults, print_info=print_info
                                )
                                structures_list.append({
                                    'structure': structure,
                                    'burger_vector': burgers_vector_for_cell.tolist(),
                                })
                                
            faulted_result = structures_list
            
        return faulted_result

    def _build_faulted_structure_helper(
        self, 
        config: FaultConfig, 
        ase_atoms_t, 
        layers_dict, 
        print_info=False
    ):
        """Internal helper for unstable/intrinsic fault building."""
        if not config.possible:
            return None
        
        if config.removal_layers is not None:
             structure = build_atoms_from_stacking_removal(
                ase_atoms_t, self._n_unit_cells, config.removal_layers, layers_dict,
                additional_spacing=(config.interface, 0.0), print_info=print_info
            )
             return structure
        
        if config.burger_vectors is not None and isinstance(config.burger_vectors, list):
            structures_list = []
            for bv in config.burger_vectors:
                structure = build_atoms_from_burger_vector(
                    ase_atoms_t, self._n_unit_cells, bv, layers_dict, print_info=print_info
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
    
    def __init__(self, ase: ty.Optional[Atoms] = None, **kwargs):
        """
        Initialize AiiDA Data node.
        pass `ase` to set the content.
        """
        # Call Data.__init__
        super().__init__(**kwargs)
        if ase is not None:
            self.set_ase(ase)
            
    def set_ase(self, ase_atoms: Atoms):
        """Set the ASE atoms content, serializing to JSON."""
        # Clean current attributes if needed? (Data usually immutable once stored)
        self.base.attributes.set('ase_atoms_json', encode(ase_atoms))
        # Clear cached properties in parent FaultedStructure if any?
        # FaultedStructure uses self._strukturbericht, etc.
        # But this instance is re-initialized or new.
        # If set_ase is called, we should invalidate caches.
        self._strukturbericht = None
        self._gliding_system = None
        
    @property
    def unit_cell(self) -> Atoms:
        """Retrieve ASE atoms from attributes."""
        # Override FaultedStructure.unit_cell
        json_str = self.base.attributes.get('ase_atoms_json')
        if json_str is None:
             raise AttributeError("No ASE atoms set for this FaultedStructureData.")
        return ase_decode(json_str)

    @property
    def structure_data(self):
         # FaultedStructure uses _structure_data in __init__? 
         # Wait, new FaultedStructure uses _ase_atoms in __init__.
         # But FaultedStructure.__init__ sets self._ase_atoms = ase_atoms.
         # FaultedStructureData does NOT call FaultedStructure.__init__.
         # It calls Data.__init__ (super().__init__).
         # So self._ase_atoms is not set.
         # But we override unit_cell property, which FaultedStructure uses.
         # So FaultedStructure methods calling self.unit_cell will call our property.
         # This works correctly via polymorphism.
         pass
