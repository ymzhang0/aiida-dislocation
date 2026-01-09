from aiida import orm
from math import sqrt, acos, pi, ceil
import numpy
import numpy.linalg as la
import logging
from ase import Atoms
from ase.spacegroup import get_spacegroup
from ase.build import make_supercell
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pathlib
import typing as ty
from copy import deepcopy
import itertools
from deprecated import deprecated
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import from new modules
from .structure_utils import (
    AttributeDict, 
    group_by_layers, 
    read_structure_from_file, 
    get_strukturbericht,
    is_primitive_cell,
    get_elements_for_wyckoff_symbols,
    check_bravais_lattice,
    calculate_surface_area
)
from .gliding_systems import (
    FaultConfig, 
    GlidingPlaneConfig, 
    GlidingSystem, 
    get_gliding_system,
    A1GlidingSystem,
    A2GlidingSystem, 
    B1GlidingSystem, 
    B2GlidingSystem,
    C1bGlidingSystem, 
    L21GlidingSystem, 
    E21GlidingSystem,
    _GLIDING_SYSTEM_REGISTRY,
    _GLIDING_SYSTEM_CACHE
)
from .structure_builder import (
    build_atoms_surface,
    build_atoms_from_stacking_removal,
    build_atoms_from_stacking_mirror,
    build_atoms_from_burger_vector,
    build_atoms_from_burger_vector_general,
    build_atoms_from_burger_vector_with_vacuum,
    update_faults
)

# Re-export necessary functions and classes
__all__ = [
    'FaultConfig', 'GlidingPlaneConfig', 'GlidingSystem',
    'get_gliding_system', 'AttributeDict', 'read_structure_from_file',
    'group_by_layers', 'get_strukturbericht', 'get_unstable_faulted_structure',
    'get_conventional_structure', 'get_cleavaged_structure', 
    'get_faulted_structure', 'get_unstable_faulted_structure_and_kpoints',
    'is_primitive_cell', 'get_elements_for_wyckoff_symbols',
    'get_kpoints_mesh_for_supercell', 'calculate_surface_area'
]

def _build_base_structure(
    structure_type: str,
    ase_atoms_t,
    n_unit_cells: int,
    layers_dict: dict,
    plane_config: GlidingPlaneConfig,
    print_info: bool = False,
):
    """
    Build base structures (unfaulted/conventional, cleavaged, twinning).
    
    Args:
        structure_type: Type of structure to build ('unfaulted', 'cleavaged', 'twinning')
        ase_atoms_t: Transformed atoms structure
        n_unit_cells: Number of unit cells
        layers_dict: Dictionary of layers
        plane_config: GlidingPlaneConfig object
        print_info: Whether to print debug information
        
    Returns:
        Structure (ASE Atoms object) or None
    """
    if structure_type == 'unfaulted':
        return ase_atoms_t
    elif structure_type == 'cleavaged':
        return build_atoms_surface(
            ase_atoms_t, n_unit_cells, layers_dict, print_info=print_info,
        )
    elif structure_type == 'twinning':
        if plane_config.n_layers > 2:
            return build_atoms_from_stacking_mirror(
                ase_atoms_t, n_unit_cells, layers_dict, print_info=print_info,
            )
        else:
            return None
    else:
        raise ValueError(f'Unknown base structure type: {structure_type}')


def _build_faulted_structure(
    config: FaultConfig,
    ase_atoms_t,
    n_unit_cells: int,
    layers_dict: dict,
    print_info: bool = False
):
    """
    Internal helper to build faulted structure from config.
    """
    if not config.possible:
        return None
    
    if config.removal_layers is not None:
        structure = build_atoms_from_stacking_removal(
            ase_atoms_t,
            n_unit_cells,
            config.removal_layers,
            layers_dict,
            additional_spacing=(config.interface, 0.0),
            print_info=print_info
        )
        return {
            'mode': 'removal',
            'structures': [{
                'structure': structure,
                'layers': config.removal_layers,
            }],
        }
    
    if config.burger_vectors is not None:
        structures_list = []
        # Handle list format (intrinsic/unstable/extrinsic usually)
        if isinstance(config.burger_vectors, list):
            for bv in config.burger_vectors:
                structure = build_atoms_from_burger_vector(
                    ase_atoms_t,
                    n_unit_cells,
                    bv,
                    layers_dict,
                    print_info=print_info
                )
                structures_list.append({
                    'structure': structure,
                    'burger_vector': bv,
                })

        if structures_list:
            return {
                'mode': 'gliding',
                'structures': structures_list,
            }
            
    return None


def _prepare_structure_data(
    ase_atoms_conventional,
    gliding_plane: ty.Optional[str] = None,
    print_info: bool = False,
) -> tuple[str, GlidingSystem, GlidingPlaneConfig, dict]:
    """
    Prepare common structure data for building faulted and cleavaged structures.
    This function works on conventional cell, not unit cell.
    
    Args:
        ase_atoms_conventional: ASE Atoms object representing the conventional cell
        gliding_plane: Gliding plane direction (e.g., '111', '011'). 
                       If None, uses the default plane from gliding system
        print_info: Whether to print debug information
    
    Returns:
        Tuple of (strukturbericht, gliding_system, plane_config, layers_dict)
    """
    strukturbericht = get_strukturbericht(ase_atoms_conventional)
    if not strukturbericht:
        raise ValueError('No match found in the provided list of prototypes.')

    if print_info:
        print(f'Strukturbericht {strukturbericht} detected')
    
    # Get gliding system using new architecture
    gliding_system = get_gliding_system(strukturbericht)
    
    # Use default plane if not provided
    if not gliding_plane:
        gliding_plane = gliding_system.default_plane
    
    plane_config = gliding_system.get_plane(gliding_plane)
    
    # Group layers from conventional structure
    layers_dict = group_by_layers(ase_atoms_conventional)
    
    if len(layers_dict) != plane_config.n_layers:
        raise ValueError(
            f'Layer count mismatch: found {len(layers_dict)} layers, but expected {plane_config.n_layers} for '
            f'{strukturbericht} with gliding plane {gliding_plane}. '
            'This may indicate wrong initial structure, incorrect structure type, or incorrect transformation matrix.'
        )
    
    return (strukturbericht, gliding_system, plane_config, layers_dict)

def get_unstable_faulted_structure(
        ase_atoms_uc,
        gliding_plane: ty.Optional[str] = None,
        P: ty.Optional[ty.Union[list, numpy.ndarray]] = None,
        n_unit_cells: int = 3,
        print_info: bool = False,
    ) -> tuple[str, AttributeDict]:
    """
    Generate faulted structures for a given unit cell structure.
    
    Args:
        ase_atoms_uc: ASE Atoms object representing the unit cell
        gliding_plane: Gliding plane direction (e.g., '111', '011'). Defaults to '111'
        P: Transformation matrix. If None, uses the default from gliding system
        n_unit_cells: Number of unit cells to repeat
        print_info: Whether to print debug information
        
    Returns:
        Tuple of (strukturbericht, structures_dict) where structures_dict contains:
            - 'conventional': Conventional structure
            - 'twinning': Twinning structure (if applicable)
            - 'cleavaged': Cleavaged surface structure
            - 'intrinsic': Intrinsic fault structure (if configured)
            - 'unstable': Unstable fault structure (if configured)
            - 'extrinsic': Extrinsic fault structure (if configured)
    """

    strukturbericht = get_strukturbericht(ase_atoms_uc)
    if not strukturbericht:
        raise ValueError('No match found in the provided list of prototypes.')

    if print_info:
        print(f'Strukturbericht {strukturbericht} detected')
    
    # Get gliding system using new architecture
    gliding_system = get_gliding_system(strukturbericht)
    
    # Use default plane if not provided
    if not gliding_plane:
        gliding_plane = gliding_system.default_plane
    
    plane_config = gliding_system.get_plane(gliding_plane)
    
    # Use provided transformation matrix or default from config
    if not P:
        P = plane_config.transformation_matrix
    else:
        P = numpy.array(P)

    ase_atoms_t = make_supercell(ase_atoms_uc, P)
    layers_dict = group_by_layers(ase_atoms_t)
    
    if len(layers_dict) != plane_config.n_layers:
        raise ValueError(
            f'Layer count mismatch: found {len(layers_dict)} layers, but expected {plane_config.n_layers} for '
            f'{strukturbericht} with gliding plane {gliding_plane}. '
            'This may indicate wrong initial structure, incorrect structure type, or incorrect transformation matrix.'
        )

    # Build base structures using unified function
    structures = AttributeDict({
        'conventional': _build_base_structure(
            'unfaulted', ase_atoms_t, n_unit_cells, layers_dict, plane_config, print_info
        ),
        'twinning': _build_base_structure(
            'twinning', ase_atoms_t, n_unit_cells, layers_dict, plane_config, print_info
        ),
        'cleavaged': _build_base_structure(
            'cleavaged', ase_atoms_t, n_unit_cells, layers_dict, plane_config, print_info
        ),
    })

    # Build faulted structures using new architecture
    intrinsic_fault = _build_faulted_structure(
        plane_config.intrinsic, ase_atoms_t, n_unit_cells, layers_dict, print_info
    )
    if intrinsic_fault is not None:
        structures['intrinsic'] = intrinsic_fault

    unstable_fault = _build_faulted_structure(
        plane_config.unstable, ase_atoms_t, n_unit_cells, layers_dict, print_info
    )
    if unstable_fault is not None:
        structures['unstable'] = unstable_fault

    extrinsic_fault = _build_faulted_structure(
        plane_config.extrinsic, ase_atoms_t, n_unit_cells, layers_dict, print_info
    )
    if extrinsic_fault is not None:
        structures['extrinsic'] = extrinsic_fault

    return (strukturbericht, structures)

def get_conventional_structure(
        ase_atoms_uc,
        gliding_plane: ty.Optional[str] = None,
        P: ty.Optional[ty.Union[list, numpy.ndarray]] = None,
        print_info: bool = False,
) -> tuple[str, Atoms]:
    """
    Generate conventional (unfaulted) structure from unit cell structure.
    This is the only function that converts unit cell to conventional cell.
    
    Args:
        ase_atoms_uc: ASE Atoms object representing the unit cell
        gliding_plane: Gliding plane direction (e.g., '111', '011'). 
                       If None, uses the default plane from gliding system
        P: Transformation matrix. If None, uses the default from gliding system
        print_info: Whether to print debug information
        
    Returns:
        Tuple of (strukturbericht, conventional_structure)
    """
    strukturbericht = get_strukturbericht(ase_atoms_uc)
    if not strukturbericht:
        raise ValueError('No match found in the provided list of prototypes.')

    if print_info:
        print(f'Strukturbericht {strukturbericht} detected')
    
    # Get gliding system using new architecture
    gliding_system = get_gliding_system(strukturbericht)
    
    # Use default plane if not provided
    if not gliding_plane:
        gliding_plane = gliding_system.default_plane
    
    plane_config = gliding_system.get_plane(gliding_plane)
    
    # Use provided transformation matrix or default from config
    if not P:
        P = plane_config.transformation_matrix
    else:
        P = numpy.array(P)

    ase_atoms_conventional = make_supercell(ase_atoms_uc, P)
    
    return (strukturbericht, ase_atoms_conventional)


def get_cleavaged_structure(
        ase_atoms_conventional,
        gliding_plane: ty.Optional[str] = None,
        n_unit_cells: int = 3,
        print_info: bool = False,
    ) -> tuple[str, Atoms]:
    """
    Generate cleavaged surface structure from conventional cell structure.
    
    Args:
        ase_atoms_conventional: ASE Atoms object representing the conventional cell
        gliding_plane: Gliding plane direction (e.g., '111', '011'). 
                       If None, uses the default plane from gliding system
        n_unit_cells: Number of unit cells to repeat
        print_info: Whether to print debug information
        
    Returns:
        Tuple of (strukturbericht, cleavaged_structure)
    """
    strukturbericht, _, _, layers_dict = _prepare_structure_data(
        ase_atoms_conventional, gliding_plane, print_info
    )
    
    cleavaged_structure = build_atoms_surface(
        ase_atoms_conventional, n_unit_cells, layers_dict, print_info=print_info,
    )
    
    return (strukturbericht, cleavaged_structure)


class FaultedStructureEntry(ty.TypedDict, total=False):
    """Container for a single faulted structure variant."""
    structure: Atoms
    layers: list[int]  # only for removal faults
    burger_vector: list[float]  # only for gliding faults


class FaultedStructureResult(ty.TypedDict):
    """Normalized return type for faulted structures."""
    mode: ty.Literal['removal', 'gliding']
    structures: list[FaultedStructureEntry]

def get_faulted_structure(
        ase_atoms_conventional,
        fault_mode: str,
        fault_type: str,
        additional_spacing: float = 0.0,
        gliding_plane: ty.Optional[str] = None,
        n_unit_cells: int = 3,
        vacuum_ratio: float = 0.0,
        print_info: bool = False,
        **kwargs,
    ) -> tuple[str, ty.Optional[FaultedStructureResult]]:
    """Generate faulted structure of a specific type from conventional cell structure."""

    from copy import deepcopy as dp
    
    if fault_mode not in ['removal', 'vacuum', 'general']:
        raise ValueError(
            f"fault_mode must be one of 'removal', 'vacuum', 'general', "
            f"got '{fault_mode}'"
        )

    if fault_mode == 'removal' and fault_type not in ['intrinsic', 'unstable', 'extrinsic']:
        raise ValueError(
            f"fault_type must be one of 'intrinsic', 'unstable', or 'extrinsic', "
            f"got '{fault_type}'"
        )

    strukturbericht, _, plane_config, layers_dict = _prepare_structure_data(
        ase_atoms_conventional, gliding_plane, print_info
    )

    fault_config = getattr(plane_config, fault_type)

    if not fault_config.possible:
        return None
    
    # Prefer removal mode if available and requested
    if fault_mode == 'removal' and fault_config.removal_layers is not None:
        if fault_config.removal_layers is not None:
            raise ValueError(
                f"Fault type {fault_type} is not available for removal mode."
            )
        structure = build_atoms_from_stacking_removal(
            ase_atoms_conventional,
            n_unit_cells,
            fault_config.removal_layers,
            layers_dict,
            additional_spacing=(fault_config.interface, additional_spacing),
            print_info=print_info
        )
        faulted_structure = {
            'mode': 'removal',
            'structures': [{
                'structure': structure,
                'layers': fault_config.removal_layers,
            }],
        }

    # Use burger vector (gliding/vacuum) mode if available
    if fault_mode == 'vacuum' and vacuum_ratio > 0.0 and fault_config.burger_vectors is not None:
        structures_list: list[FaultedStructureEntry] = []
        for burger_vector in fault_config.burger_vectors:
            structure = build_atoms_from_burger_vector_with_vacuum(
                ase_atoms_conventional,
                n_unit_cells,
                burger_vector,
                layers_dict,
                vacuum_ratio=vacuum_ratio,
                print_info=print_info
            )

            structures_list.append({
                'structure': structure,
                'burger_vector': burger_vector,
            })
        faulted_structure = {
            'mode': 'vacuum',
            'structures': structures_list,
        }
    
    if fault_mode == 'general' and fault_config.burger_vectors is not None:
        structures_list: list[FaultedStructureEntry] = []
        nsteps = kwargs.get('nsteps', fault_config.nsteps)
        stacking_order = ''.join(layers_dict.keys())
        if not isinstance(n_unit_cells, int) or n_unit_cells < 2:
            raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be an integer >= 2.")

        zs = [(value['z'] + layer)/n_unit_cells for layer in range(n_unit_cells) for value in layers_dict.values()]
        stacking_order_supercell = stacking_order * n_unit_cells

        new_cell = ase_atoms_conventional.cell.array.copy()
        new_cell[-1] *= (n_unit_cells)

        if isinstance(fault_config.burger_vectors, dict):
            for direction_name, path_points in fault_config.burger_vectors.items():
                # iterate over path points
                for segment in path_points:
                    burgers_vector_for_cell = numpy.zeros(3)
                    faults = numpy.zeros((len(stacking_order_supercell), 3))
                    structure = build_atoms_from_burger_vector_general(
                        new_cell,
                        deepcopy(zs),
                        layers_dict,
                        stacking_order_supercell,
                        burgers_vector_for_cell,
                        faults,
                        print_info=print_info
                    )
                    structures_list.append({
                        'structure': structure,
                        'burger_vector': burgers_vector_for_cell.tolist(),
                    })
                    for interface, burgers_vector in segment:
                        burgers_vector_step = numpy.array(burgers_vector) / nsteps
                        for _ in range(1, 1+nsteps):
                            # Global progress t in [0, 1]
                            # print(faults)
                            faults = update_faults(faults, interface, burgers_vector_step)
                            burgers_vector_for_cell += burgers_vector_step
                            structure = build_atoms_from_burger_vector_general(
                                new_cell,
                                deepcopy(zs),
                                layers_dict,
                                stacking_order_supercell,
                                burgers_vector_for_cell,
                                faults,
                                print_info=print_info
                            )
                            
                            structures_list.append({
                                'structure': structure,
                                'burger_vector': burgers_vector_for_cell.tolist(),
                            })

        else:
             # Legacy or unhandled format - skipping as per new requirement
             pass
             
        faulted_structure = {
            'mode': 'gliding',
            'structures': structures_list,
        }
    return (strukturbericht, faulted_structure)


def get_unstable_faulted_structure_and_kpoints(
    structure_uc: orm.StructureData,
    kpoints_uc: orm.KpointsData,
    n_layers: int,
    slipping_system: orm.List,
) -> tuple[orm.StructureData, orm.KpointsData]:
    """Get unstable faulted structure and corresponding kpoints for GSFE workflow.
    
    This is a convenience wrapper that extracts structure and calculates kpoints
    from get_unstable_faulted_structure.
    
    :param structure_uc: Unit cell structure
    :param kpoints_uc: Unit cell kpoints
    :param n_layers: Number of layers
    :param slipping_system: List containing [structure_type, gliding_plane, slipping_direction]
    :return: Tuple of (faulted_structure, kpoints)
    """
    structure_type, gliding_plane, _ = slipping_system.get_list()
    
    # Get unstable faulted structure
    _, structures_dict = get_unstable_faulted_structure(
        structure_uc.get_ase(),
        gliding_plane=gliding_plane if gliding_plane else None,
        n_unit_cells=n_layers,
    )
    
    # Extract unstable structure
    if 'unstable' not in structures_dict or structures_dict['unstable'] is None:
        raise ValueError('Unstable fault structure is not available for this gliding system.')
    
    unstable_data = structures_dict['unstable']
    if not unstable_data.get('structures'):
        raise ValueError('Unstable fault structure list is empty.')
    
    unstable_structure_ase = unstable_data['structures'][0].get('structure')
    if unstable_structure_ase is None:
        raise ValueError('Unstable fault structure is missing structure data.')
    
    # Convert to StructureData
    structure_sc = orm.StructureData(ase=unstable_structure_ase)
    
    # Calculate kpoints for supercell
    # Get z-ratio between supercell and unit cell
    z_ratio = unstable_structure_ase.cell.cellpar()[2] / structure_uc.cell.cellpar()[2]
    kpoints_mesh_uc = kpoints_uc.get_kpoints_mesh()[0]
    
    # Adjust kpoints mesh for supercell
    kpoints_mesh_sc = list(kpoints_mesh_uc)
    kpoints_mesh_sc[2] = ceil(kpoints_mesh_sc[2] / z_ratio)
    
    kpoints_sc = orm.KpointsData()
    kpoints_sc.set_kpoints_mesh(kpoints_mesh_sc)
    
    return (structure_sc, kpoints_sc)

def get_kpoints_mesh_for_supercell(
        kpoints_uc: orm.KpointsData,
        n_layers: int,
        n_stacking: int,
    ) -> orm.KpointsData:
    """
    Get the kpoints mesh for the supercell.
    Assume scaling by n_layers * n_stacking along Z?
    Or just roughly heuristic.
    """
    kpoints_mesh = list(kpoints_uc.get_kpoints_mesh()[0])
    # Heuristic: reduce z-sampling by factor of supercell expansion
    # Total expansion ~ n_layers (if n_layers means n_unit_cells)
    if n_layers > 0:
        kpoints_mesh[2] = max(1, int(ceil(kpoints_mesh[2] / n_layers)))
    
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh(kpoints_mesh)
    return kpoints