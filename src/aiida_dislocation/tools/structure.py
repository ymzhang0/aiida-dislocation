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

class AttributeDict(dict):
    """
    A dictionary that can be accessed like an attribute.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# logger = logging.getLogger('aiida.workflow.dislocation')


@dataclass
class FaultConfig:
    """Configuration for a fault type (intrinsic, unstable, or extrinsic)."""
    removal_layers: ty.Union[list[int], int] = None
    burger_vectors: ty.Optional[list[list[float]]] = None
    periodicity: bool = False
    possible: bool = True
    interface: int = 0
    nsteps: int = 1

@dataclass
class GlidingPlaneConfig:
    """Configuration for a specific gliding plane."""
    transformation_matrix: list[list[int]]
    transformation_matrix_c: ty.Optional[list[list[int]]] = None
    n_layers: int = 2
    intrinsic: FaultConfig = field(default_factory=FaultConfig)
    unstable: FaultConfig = field(default_factory=FaultConfig)
    extrinsic: FaultConfig = field(default_factory=FaultConfig)
    general: FaultConfig = field(default_factory=FaultConfig)

class GlidingSystem(ABC):
    """Base class for gliding system configurations."""
    
    default_plane: str = '111'  # Default gliding plane, can be overridden by subclasses
    
    def __init__(self, strukturbericht: str):
        self.strukturbericht = strukturbericht
        self._planes: dict[str, GlidingPlaneConfig] = {}
        self._register_planes()
    
    @abstractmethod
    def _register_planes(self):
        """Register all gliding planes for this system."""
        pass
    
    def get_plane(self, gliding_plane: str) -> GlidingPlaneConfig:
        """Get configuration for a specific gliding plane."""
        if gliding_plane not in self._planes:
            raise ValueError(
                f'Gliding plane {gliding_plane} is not supported for {self.strukturbericht}. '
                f'Supported planes: {list(self._planes.keys())}'
            )
        return self._planes[gliding_plane]
    
    def list_planes(self) -> list[str]:
        """List all supported gliding planes."""
        return list(self._planes.keys())

# Concrete implementations
class A1GlidingSystem(GlidingSystem):
    """A1 (FCC) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 0, 0], [0, -1, 1], [-1, 1, 1]],
            n_layers=2,
            intrinsic=FaultConfig(possible=False),
            extrinsic=FaultConfig(possible=False),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 0, 0],
                    ],
                interface=(2, ),
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            transformation_matrix_c=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            n_layers=2,
            intrinsic=FaultConfig(possible=False),
            extrinsic=FaultConfig(possible=False),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [0, 1, 0],
                    ],
                interface=(2, ),
                nsteps = 10
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            transformation_matrix_c=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
            n_layers=3,
            intrinsic=FaultConfig(
                possible=True,
                removal_layers=[3],
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
                interface=3,
            ),
            extrinsic=FaultConfig(
                possible=True,
                removal_layers=[3, 5],
                burger_vectors=[[2/3, 2/3, 0]],
                periodicity=False
            ),
            unstable=FaultConfig(
                possible=True,
                removal_layers=[-1],
                burger_vectors=[[1/6, 1/6, 0]],
                periodicity=False,
                interface=3,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1/3, 1/3, 0],
                    [1/3, 1/3, 0]
                    ],
                interface=(3, 4),
                nsteps = 8
            )
        )

class A2GlidingSystem(GlidingSystem):
    """A2 (BCC) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 1, 0], [1, 0, 1], [0, 1, 1]],
            n_layers=2,
            intrinsic=FaultConfig(
                removal_layers=[2]
                ),
            unstable=FaultConfig(
                removal_layers=[2]
                ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1, 0, 0]],
                interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, 0], [0, 0, 1], [2, 1, 1]],
            transformation_matrix_c=[[0, 1, -1], [0, 1, 1], [2, 1, 1]],
            n_layers=2,
            intrinsic=FaultConfig(
                removal_layers=[2]
                ),
            unstable=FaultConfig(
                removal_layers=[2]
                ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1, 0, 0]],
                interface=[2,],
                nsteps = 10
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[-1, 1, 0], [-1, 0, 1], [1, 1, 1]],
            transformation_matrix_c=[[-2, 1, 1], [0, -1, 1], [2, 2, 2]],
            n_layers=3,
            intrinsic=FaultConfig(
                removal_layers=[3], interface=3,
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
            ),
            extrinsic=FaultConfig(
                removal_layers=[3, 5],
                burger_vectors=[[2/3, 2/3, 0]],
                periodicity=False,
            ),
            unstable=FaultConfig(
                removal_layers=[3, 4], interface=3,
                burger_vectors=[[2/3, 2/3, 0]],
                periodicity=False,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1/3, 1/3, 0],
                    [1/3, 1/3, 0]
                    ],
                interface=(3, 4),
                nsteps = 8
            )
        )

class B1GlidingSystem(GlidingSystem):
    """B1 (NaCl) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 0, 0], [0, -1, 1], [-1, 1, 1]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1, 0, 0]],
                interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [0, 1, 0],
                    ],
                interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            transformation_matrix_c=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
            n_layers=6,
            intrinsic=FaultConfig(
                removal_layers=[6, 7, 8, 9], interface=6,
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1/3, 1/3, 0],
                    ],
                interface=(6, ),
                nsteps = 10
            )
        )


class B2GlidingSystem(GlidingSystem):
    """B2 (CsCl) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1, 0, 0]],
                interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [1, 0, 0], [0, 1, 1]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 1, 0],
                    ],
                interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            transformation_matrix_c=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
            n_layers=6,
            intrinsic=FaultConfig(
                removal_layers=[6, 7, 8, 9], interface=6,
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1/3, 1/3, 0],
                    ],
                interface=(6, ),
                nsteps = 10
            )
        )

class C1bGlidingSystem(GlidingSystem):
    """C1b (Half-Heusler) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 0, 0], [0, -1, 1], [-1, 1, 1]],
            n_layers=4,
            intrinsic=FaultConfig(
                removal_layers=[2],
                burger_vectors=[[1/2, 0, 0], [0, 1/2, 0], [1/2, 1/2, 0]],
                periodicity=False,
                interface=2,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 0, 0],
                    [0, 1, 0],
                    ],
                interface=(4, 4),
                nsteps = 8
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            n_layers=2,
            intrinsic=FaultConfig(
                removal_layers=[2],
                burger_vectors=[[1/2, 0, 0], [0, 1/2, 0], [1/2, 1/2, 0]],
                periodicity=False,
                interface=2,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 0, 0],
                    [0, 1, 0],
                    ],
                interface=(2, 2),
                nsteps = 8
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            transformation_matrix_c=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
            n_layers=9,
            intrinsic=FaultConfig(
                removal_layers=[9, 10, 11, 12, 13, 14],
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
                interface=9,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1/3, 1/3, 0]],
                interface=9,
                nsteps = 10
            )
        )


class L21GlidingSystem(GlidingSystem):
    """L21 (Heusler) gliding system."""
    
    def _register_planes(self):
        self._planes['100'] = GlidingPlaneConfig(
            transformation_matrix=[[1, 0, 0], [0, -1, 1], [-1, 1, 1]],
            n_layers=4,
            intrinsic=FaultConfig(
                removal_layers=[2],
                burger_vectors=[[1/2, 0, 0], [0, 1/2, 0], [1/2, 1/2, 0]],
                periodicity=False,
                interface=2,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 0, 0],
                    [0, 1, 0],
                    ],
                interface=(4, 4),
                nsteps = 6
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            n_layers=2,
            intrinsic=FaultConfig(
                removal_layers=[2],
                burger_vectors=[[1/2, 0, 0], [0, 1/2, 0], [1/2, 1/2, 0]],
                periodicity=False,
                interface=2,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[
                    [1, 0, 0],
                    [0, 1, 0],
                    ],
                interface=(2, 2),
                nsteps = 6
            )
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            transformation_matrix_c=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
            n_layers=9,
            intrinsic=FaultConfig(
                removal_layers=[9, 10, 11, 12, 13, 14],
                burger_vectors=[[1/3, 1/3, 0]],
                periodicity=False,
                interface=9,
            ),
            general=FaultConfig(
                possible=True,
                burger_vectors=[[1/3, 1/3, 0]],
                interface=9,
                nsteps = 10
            )
        )

class E21GlidingSystem(GlidingSystem):
    """E21 (Perovskite) gliding system."""
    
    def _register_planes(self):
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 0, 1], [-1, 1, 0], [1, 1, 0]],
            n_layers=4,
            intrinsic=FaultConfig(removal_layers=[4, 5]),
            extrinsic=FaultConfig(possible=False)
        )
        self._planes['111'] = GlidingPlaneConfig(
            transformation_matrix=[[1, -1, 0], [1, 0, -1], [1, 1, 1]],
            n_layers=6,
            intrinsic=FaultConfig(removal_layers=[6, 7, 8, 9]),
            unstable=FaultConfig(removal_layers=[6, 7])
        )

# Registry for gliding systems
_GLIDING_SYSTEM_REGISTRY: dict[str, type[GlidingSystem]] = {
    'A1': A1GlidingSystem,
    'A2': A2GlidingSystem,
    'B1': B1GlidingSystem,
    'B2': B2GlidingSystem,
    'C1_b': C1bGlidingSystem,
    'L2_1': L21GlidingSystem,
    'E_21': E21GlidingSystem,
}

# Cache for instantiated systems
_GLIDING_SYSTEM_CACHE: dict[str, GlidingSystem] = {}


def get_gliding_system(strukturbericht: str) -> GlidingSystem:
    """Get or create a gliding system instance."""
    if strukturbericht not in _GLIDING_SYSTEM_REGISTRY:
        raise ValueError(
            f'Strukturbericht {strukturbericht} is not supported. '
            f'Supported types: {list(_GLIDING_SYSTEM_REGISTRY.keys())}'
        )
    
    if strukturbericht not in _GLIDING_SYSTEM_CACHE:
        system_class = _GLIDING_SYSTEM_REGISTRY[strukturbericht]
        _GLIDING_SYSTEM_CACHE[strukturbericht] = system_class(strukturbericht)
    
    return _GLIDING_SYSTEM_CACHE[strukturbericht]


@deprecated(reason="This function is not used in any workflow. Use ASE's built-in methods instead.")
def check_bravais_lattice(ase_atoms):
    bl = ase_atoms.cell.get_bravais_lattice(eps=1e-6)
    return bl.name

def read_structure_from_file(
    filename: ty.Union[str, pathlib.Path],
    store: bool = False
    ) -> orm.StructureData:
    """Read a xsf/xyz/cif/.. file and return aiida ``StructureData``."""
    from ase.io import read as aseread

    if filename in [
        'Al',
        'V',
        'Cu',
        'NiTi',
        'TaRu3C',
        'Nb3Sn',
        'CrH',
        'AsTe',
        'NbCoSb',
        'ZrRuSb',
        'TiSbRu',
        'TiAlPt',
        'MoN',
        'HfAlPd2',
        'ZrGaNi2',
        'YSnPd2',
        'ZrAlNi2',
        'MgB2',
        'TaSe2',
        'VC'
        ]:
        import importlib.resources
        data_path = importlib.resources.files('aiida_dislocation.data')
        filename = data_path / f'structures/cif/{filename}.cif'

    struct = orm.StructureData(ase=aseread(filename))

    if store:
        struct.store()
        print(f"Read and stored structure {struct.get_formula()}<{struct.pk}>")

    return struct

def group_by_layers(
    ase_atoms,
    decimals=6,
    ):
    """
    Splits an ASE Atoms object into multiple layers based on z-coordinates.

    Args:
        atoms (ase.Atoms): The input Atoms object to be split.
        decimals (int): The number of decimal places to round the z-coordinates
                        to for grouping atoms into layers. This acts as a tolerance.

    Returns:
        dict: A dictionary where keys are the unique z-coordinates of the layers
              and values are new Atoms objects, each containing one layer.
    """
    import string
    from copy import deepcopy

    if not ase_atoms:
        return {}

    scaled_positions = ase_atoms.get_scaled_positions()

    z_coords = scaled_positions[:, 2]
    rounded_z = numpy.round(z_coords, decimals=decimals) % 1.0

    sorted_unique_z = sorted(numpy.unique(rounded_z))

    labels = string.ascii_uppercase
    if len(sorted_unique_z) > len(labels):
        print(f"Warning: Number of layers ({len(sorted_unique_z)}) exceeds number of labels ({len(labels)}).")
        labels = [f"Layer_{i+1}" for i in range(len(sorted_unique_z))]

    labeled_layers_dict = {}

    for i, z_val in enumerate(sorted_unique_z):
        layer_label = labels[i]
        indices = numpy.where(rounded_z == z_val)[0]
        layer_content = [deepcopy(ase_atoms[idx]) for idx in indices]
        labeled_layers_dict[layer_label] = {
            'atoms': layer_content,
            'z': z_val
            }

    return labeled_layers_dict

def build_atoms_surface(
    ase_atoms_uc,
    n_unit_cells,
    layers_dict,
    print_info = False,
    ):
    atoms = Atoms()

    if not isinstance(n_unit_cells, int) or n_unit_cells < 1:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be a positive integer.")

    stacking_order = n_unit_cells * ''.join(layers_dict.keys())

    zs = [(value['z'] + cell)/n_unit_cells/2 for cell in range(n_unit_cells) for value in layers_dict.values()]

    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= 2 * n_unit_cells
    atoms.set_cell(new_cell)
    for layer_label, z in zip(stacking_order, zs):
        for atom in layers_dict[layer_label]['atoms']:
            scaled_position = atom.scaled_position
            scaled_position[-1] = z
            atom.position = scaled_position @ new_cell
            atoms.append(atom)

    return atoms

def build_atoms_from_stacking_removal(
    ase_atoms_uc,
    n_unit_cells,
    removed_layers,
    layers_dict,
    additional_spacing = (0, 0.0),
    print_info = False,
    ):

    atoms = Atoms()

    stacking_order = n_unit_cells * ''.join(layers_dict.keys())
    if not isinstance(n_unit_cells, int) or n_unit_cells < 1:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be a positive integer.")
    if any(layer >= len(stacking_order) for layer in removed_layers):
        raise ValueError(
            f"Invalid removed layers {removed_layers}: layer indices must be < {len(stacking_order)} "
            f"(number of layers in stacking order)"
        )

    zs = numpy.array([value['z']/n_unit_cells + layer/n_unit_cells for layer in range(n_unit_cells) for value in layers_dict.values()])

    removed_layers_sorted = sorted(set(removed_layers))
    removed_spacing = 0.0
    faulted_stacking = "".join([char for i, char in enumerate(stacking_order) if i not in removed_layers_sorted])

    # Remove layers from the end to avoid index shifts while updating zs
    for removed_layer in reversed(removed_layers_sorted):
        spacing = zs[removed_layer] - zs[removed_layer - 1]
        if spacing + additional_spacing[1] < 0.0:
            raise ValueError(f"Spacing between removed layers is less than additional spacing: {spacing} < {additional_spacing}")
        removed_spacing += spacing
        zs[removed_layer:] -= spacing
        zs = numpy.delete(zs, removed_layer)

    # Apply additional spacing if requested
    if additional_spacing[0] >= len(zs):
        raise ValueError(f"additional_spacing layer index {additional_spacing[0]} is out of bounds for remaining layers {len(zs)}")
    if additional_spacing[1] != 0.0:
        zs[additional_spacing[0]:] += additional_spacing[1]
        removed_spacing -= additional_spacing[1]

    zs /= (1-removed_spacing)
    if print_info:
        print(zs)
        print(faulted_stacking)
    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= (1-removed_spacing) * n_unit_cells
    atoms.set_cell(new_cell)
    for layer_label, z in zip(faulted_stacking, zs):
        for atom in layers_dict[layer_label]['atoms']:
            new_atom = deepcopy(atom)
            scaled_position = new_atom.scaled_position
            scaled_position[-1] = z
            new_atom.position = scaled_position @ new_cell
            atoms.append(new_atom)
    return atoms

def build_atoms_from_stacking_mirror(
    ase_atoms_uc,
    n_unit_cells,
    layers_dict,
    print_info = False,
    ):

    atoms = Atoms()
    cell = ase_atoms_uc.cell.array.copy()
    z_norm = numpy.linalg.norm(cell[2])

    n_layers_uc = len(layers_dict)
    stacking_order_uc = ''.join(layers_dict.keys())
    stacking_order = n_unit_cells * stacking_order_uc
    stacking_order_uc_r = stacking_order_uc[::-1]
    if not isinstance(n_unit_cells, int) or n_unit_cells < 1:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be a positive integer.")

    # Taking 3 unit cells of 3-layer unit cell as an example
    # Firstly, we place an 'ABC' stacking as a substrate.

    spacings = [
        (layers_dict[label]['z'] - layers_dict[prev_label]['z'])*z_norm
        for label, prev_label in zip(stacking_order_uc[1:], stacking_order_uc[:-1])
        ]
    connection_to_next_cell = (1 + layers_dict[stacking_order[0]]['z'] - layers_dict[stacking_order[-1]]['z']) * z_norm
    if print_info:
        print(spacings)
    # Then we calculate the z coordinate of 3 stacked unit cells.
    # (ABC)ABCABCABC
    zs = [
        (value['z'] + layer) * z_norm
        for layer in range(n_unit_cells)
        for value in layers_dict.values()
        ]
    # And we calculate the spacing of (ABC)CBACBACBA and reverse it.
    # We calculate the spacing between the layers.
    # Note that the first spacing just link the substrate to the reversed layers.
    # It's convenient then we remove one C layer.
    # We pop the last spacing between B and A because
    # it will be calculated later when we do normal stacking.
    spacings += [
        z - prev_z
        for z, prev_z in zip(zs[1:], zs[:-1])
        ][::-1]

    # spacings.pop()
    if print_info:
        print('zs for reversed layers', zs)
        print('spacings for reversed layers', spacings)
    # Here we do the stacking of the rest (n_unit_cells-1) unit cells.
    # Because we already have one substrate unit cell.
    # (ABC)(BACBACBA)(BCABC)
    zs = [
        (value['z'] + layer + n_unit_cells+1) * z_norm
        for layer in range(n_unit_cells-1)
        for value in layers_dict.values()
        ]

    spacings += [
        z - prev_z
        for z, prev_z in zip(zs[1:], zs[:-1])
        ]

    if print_info:
        print(spacings)
    # spacings += [(layers_dict[stacking_order_uc[0]]['z']+1.0 - layers_dict[stacking_order_uc[-1]]['z']) / n_layers/2]

    zs = [0.0] + list(itertools.accumulate(spacings))
    if print_info:
        print(zs)

    new_thickness = zs[-1] + connection_to_next_cell

    faulted_stacking = stacking_order_uc[:-1] + stacking_order_uc_r * n_unit_cells + (stacking_order_uc * (n_unit_cells-1))[1:]
    if print_info:
        print(faulted_stacking)
    z_dialation = new_thickness / z_norm
    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= z_dialation
    atoms.set_cell(new_cell)
    for layer_label, z in zip(faulted_stacking, zs):
        for atom in layers_dict[layer_label]['atoms']:
            new_atom = deepcopy(atom)
            scaled_position = new_atom.scaled_position
            scaled_position[-1] = z / new_thickness
            new_atom.position = scaled_position @ new_cell
            atoms.append(new_atom)

    return atoms

def build_atoms_from_burger_vector(
    ase_atoms_uc,
    n_unit_cells,
    burger_vector,
    layers_dict,
    print_info = False,
    ):

    atoms = Atoms()

    stacking_order = ''.join(layers_dict.keys())
    if not isinstance(n_unit_cells, int) or n_unit_cells < 2:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be an integer >= 2.")

    zs = [(value['z'] + layer)/n_unit_cells/2 for layer in range(2*n_unit_cells) for value in layers_dict.values()][::-1]

    if print_info:
        print(zs)

    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= (n_unit_cells*2)
    atoms.set_cell(new_cell)

    for layer_label in stacking_order:
        z = zs.pop()
        for atom in layers_dict[layer_label]['atoms']:
            new_atom = deepcopy(atom)
            scaled_position = new_atom.scaled_position
            scaled_position[-1] = z
            new_atom.position = scaled_position @ new_cell
            atoms.append(new_atom)

    for layer in range(n_unit_cells):
        for layer_label in stacking_order:
            z = zs.pop()
            for atom in layers_dict[layer_label]['atoms']:
                new_atom = deepcopy(atom)
                scaled_position = new_atom.scaled_position
                scaled_position += numpy.array(burger_vector)
                scaled_position[-1] = z
                new_atom.position = scaled_position @ new_cell
                atoms.append(new_atom)

    for layer in range(n_unit_cells-1):
        for layer_label in stacking_order:
            z = zs.pop()
            for atom in layers_dict[layer_label]['atoms']:
                new_atom = deepcopy(atom)
                scaled_position = new_atom.scaled_position
                scaled_position[-1] = z
                new_atom.position = scaled_position @ new_cell
                atoms.append(new_atom)

    if zs:
        raise ValueError(f"zs is not empty: {zs}")

    return atoms

def update_faults_list(faults, interface, burger_vector):
    """
    Update faults list by adding burger_vector to layers at and after interface.
    
    Args:
        faults: numpy array of shape (n_layers, 3) containing burger_vectors for each layer
        interface: Layer index where fault starts (layers at and after this index will be updated)
        burger_vector: Burger vector to add to layers at/after interface
        
    Returns:
        Updated faults array
    """
    faults = faults.copy()
    faults[interface:] += burger_vector
    return faults

def build_atoms_from_burger_vector_general(
    new_cell,
    zs,
    layers_dict,
    stacking_order_supercell,
    burger_vector_for_cell,
    faults,
    print_info = False,
    ):
    """
    Build atoms structure with burger vector faults.
    
    Args:
        new_cell: Cell matrix
        zs: List of z coordinates for layers
        layers_dict: Dictionary of layers
        stacking_order_supercell: Stacking order for supercell
        faults: numpy array of shape (n_layers, 3) containing burger_vectors for each layer
        print_info: Whether to print debug info
        
    Returns:
        Atoms object with faults applied
    """
    atoms = Atoms()
    
    # Calculate cell tilt from total burger_vector in xy plane
    # Sum all faults to get total burger_vector for cell tilt
    burger_vector_cart = burger_vector_for_cell[:2] @ new_cell[:2]
    new_cell_tilted = deepcopy(new_cell)
    new_cell_tilted[-1] += burger_vector_cart
    
    atoms.set_cell(new_cell_tilted)

    for layer_label, fault in zip(stacking_order_supercell, faults):
        z = zs.pop(0)
        for atom in layers_dict[layer_label]['atoms']:
            new_atom = deepcopy(atom)
            scaled_position = new_atom.scaled_position + fault
            scaled_position[-1] = z
            # Calculate absolute position using original cell (without tilt)
            # This preserves the absolute spatial position of atoms
            absolute_position = scaled_position @ new_cell
            # Set the absolute position directly; ASE will handle fractional coordinate conversion
            # when we access scaled_position later with the tilted cell
            new_atom.position = absolute_position
            atoms.append(new_atom)

    if zs:
        raise ValueError(f"zs is not empty: {zs}")

    return atoms

@deprecated(reason="This function is not used in any workflow. Use build_atoms_from_burger_vector instead.")
def build_atoms_from_burger_vector_with_vacuum(
    ase_atoms_uc,
    n_unit_cells,
    burger_vector,
    layers_dict,
    vacuum_ratio = 0.0,
    print_info = False,
    ):

    atoms = Atoms()

    stacking_order = ''.join(layers_dict.keys())
    if not isinstance(n_unit_cells, int) or n_unit_cells < 2:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}. Must be an integer >= 2.")

    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= n_unit_cells
    new_cell[-1] *= (1 + vacuum_ratio)
    atoms.set_cell(new_cell)

    zs = [(value['z'] + layer)/n_unit_cells/2/(1 + vacuum_ratio) for layer in range(2*n_unit_cells) for value in layers_dict.values()][::-1]

    # if print_info:
    #     print(zs)

    for layer in range(n_unit_cells):
        for layer_label in stacking_order:
            # z = (layers_dict[layer_label]['z'] + layer)/n_unit_cells/2/(1 + vacuum_ratio)
            z = zs.pop()
            for atom in layers_dict[layer_label]['atoms']:
                new_atom = deepcopy(atom)
                scaled_position = new_atom.scaled_position
                scaled_position[-1] = z
                new_atom.position = scaled_position @ new_cell
                atoms.append(new_atom)

    for layer in range(n_unit_cells):
        for layer_label in stacking_order:
            # z = (layers_dict[layer_label]['z'] + layer)/n_unit_cells/2/(1 + vacuum_ratio)
            z = zs.pop()
            for atom in layers_dict[layer_label]['atoms']:
                new_atom = deepcopy(atom)
                scaled_position = new_atom.scaled_position
                scaled_position += numpy.array(burger_vector)
                scaled_position[-1] = z
                new_atom.position = scaled_position @ new_cell
                atoms.append(new_atom)

    return atoms

def get_strukturbericht(
    atoms_to_check,
    print_info = False,
    ):
    import pymatgen.core as mg
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    # This dictionary holds the names of common prototypes and their
    # corresponding Material IDs (mp-id) in the Materials Project database.
    # sga = SpacegroupAnalyzer(read_structure_from_file('AsTe').get_pymatgen())

    # 1. Load your local crystal structure from a file (e.g., a CIF or POSCAR)
    # For this example, let's create a simple NaCl structure in memory.
    # In your real code, you would use: struct_to_check = mg.Structure.from_file("your_file.cif")

    PROTOTYPES = {
        "A1": read_structure_from_file('Al').get_pymatgen(),          # Copper (Cu)
        'A2': read_structure_from_file('V').get_pymatgen(),      # Vandadium (V)
        "B1": read_structure_from_file('AsTe').get_pymatgen(),   # Arsenic Telluride (AsTe)
        "B2": read_structure_from_file('NiTi').get_pymatgen(),   # Arsenic Telluride (AsTe)
        "B_h": read_structure_from_file('MoN').get_pymatgen(),   # Arsenic Telluride (AsTe)
        "A15": read_structure_from_file('Nb3Sn').get_pymatgen(),        # Nb3Sn (Nb3Sn)
        "C1_b": read_structure_from_file('NbCoSb').get_pymatgen(),            # Gold-Copper (AuCu3)
        "L2_1": read_structure_from_file('HfAlPd2').get_pymatgen(),            # Gold-Copper (AuCu3)
        "C_7": read_structure_from_file('TaSe2').get_pymatgen(),            # Gold-Copper (AuCu3)
        "C_32": read_structure_from_file('MgB2').get_pymatgen(),            # Gold-Copper (AuCu3)
        "E_21": read_structure_from_file('TaRu3C').get_pymatgen(),            # Gold-Copper (AuCu3)
    }
    struct_to_check = AseAtomsAdaptor.get_structure(atoms_to_check)

    try:
        # 2. Initialize the StructureMatcher.
        # primitive_cell=True is crucial because it compares the fundamental building block
        # of the crystal, ignoring differences in conventional vs. primitive cell choices.
        matcher = StructureMatcher(primitive_cell=True, scale=True)

        # 3. Fetch prototypes from Materials Project and compare
        found_match = False
        if print_info:
            print("Comparing your structure against the database...")
        for name, prototype_struct in PROTOTYPES.items():
            # Fetch the standard prototype structure
            # prototype_struct = mpr.get_structure_by_material_id(mp_id)

            # Use the .fit() method to see if they match
            if matcher.fit_anonymous(struct_to_check, prototype_struct):
                if print_info:
                    print(f"✅ Your structure<{atoms_to_check.get_chemical_formula()}> is of the {name} type.")
                found_match = True
                return name

        if not found_match:
            if print_info:
                print(f"\n❌ No match found for structure<{atoms_to_check.get_chemical_formula()}> in the provided list of prototypes.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have a valid structure file or an API key for the Materials Project.")
        return None

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
    """Generate faulted structure of a specific type from conventional cell structure.
    
    Args:
        ase_atoms_conventional: ASE Atoms object representing the conventional cell
        fault_type: Type of fault to generate ('intrinsic', 'unstable', or 'extrinsic')
        additional_spacing: Additional spacing to add to the structure
        gliding_plane: Gliding plane direction (e.g., '111', '011'). 
                       If None, uses the default plane from gliding system
        n_unit_cells: Number of unit cells to repeat
        fault_mode: Preferred fault mode ('removal' or 'vacuum'). If None, uses default from config
        vacuum_ratio: Vacuum ratio when using vacuum mode
        print_info: Whether to print debug information
        
    Returns:
        Tuple of (strukturbericht, fault_structure_data) where fault_structure_data is:
            - FaultedStructureResult dict with keys:
                * mode: 'removal' or 'gliding'
                * structures: list of entries with structure plus metadata
            - None if the fault type is not available
    """
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

        # Initialize faults list: each layer starts with zero burger_vector
        faults = numpy.zeros((len(stacking_order_supercell), 3))
        burger_vector_for_cell = numpy.array([0.0, 0.0, 0.0])
        # Process each burger_vector and interface pair
        # print(faults)
        structure = build_atoms_from_burger_vector_general(
            new_cell,
            deepcopy(zs),
            layers_dict,
            stacking_order_supercell,
            burger_vector_for_cell,
            faults,
            print_info=print_info
        )
        
        structures_list.append({
            'structure': structure,
            'burger_vector': burger_vector_for_cell.tolist(),
        })
        for burger_idx, (_burger_vector, interface) in enumerate(zip(fault_config.burger_vectors, fault_config.interface)):
            burger_vector_step = numpy.array(_burger_vector) / nsteps
            burger_vector_for_cell += burger_vector_step
            # Update faults list: add burger_vector_step to layers at/after interface
            for step in range(
                int(burger_idx>0), nsteps
                ):
                burger_vector_for_cell += burger_vector_step
                faults = update_faults_list(faults, interface, burger_vector_step)

                # Build structure with current faults
                # print(faults)
                structure = build_atoms_from_burger_vector_general(
                    new_cell,
                    deepcopy(zs),
                    layers_dict,
                    stacking_order_supercell,
                    burger_vector_for_cell,
                    faults,
                    print_info=print_info
                )
                
                structures_list.append({
                    'structure': structure,
                    'burger_vector': burger_vector_for_cell.tolist(),
                })
                
            # Note: faults have already been updated step by step in the loop above
            # No need to update again here, as the final step already has the full burger_vector applied
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

def is_primitive_cell(structure: orm.StructureData) -> bool:
    """
    Check if the structure is a primitive cell
    """
    structure_pmg = structure.get_pymatgen()

    primivite_structure_pmg = structure_pmg.get_primitive_structure()

    return structure_pmg.composition == primivite_structure_pmg.composition

def get_elements_for_wyckoff_symbols(
        structure: orm.StructureData,
    ) -> dict:
    """
    Get the symbol of the atom at the given fractional coordinates
    """
    sga = SpacegroupAnalyzer(structure.get_pymatgen_structure(), symprec=1e-5)
    symmetrized_structure = sga.get_symmetrized_structure()


    return {wyckoff_letter: element.symbol
            for wyckoff_letter, element in zip(
                symmetrized_structure.wyckoff_letters,
                symmetrized_structure.elements
                )
            }

def get_kpoints_mesh_for_supercell(
        kpoints_uc: orm.KpointsData,
        n_layers: int,
        n_stacking: int,
    ) -> orm.KpointsData:
    """
    Get the kpoints mesh for the supercell
    """
    kpoints_mesh_sc = kpoints_uc.get_kpoints_mesh()[0]

    kz = kpoints_mesh_sc[2]

    if n_layers%n_stacking != 0:
        logger.warning(
            'Supercell is not integer multiple of unit cell. '
            'Ceiling the kpoints mesh.'
            'Might cause energy difference.'
            )

    if kz % (n_layers // n_stacking) != 0:
        logger.warning(
            'kpoints mesh is not compatible with supercell dimensions. '
            'Rouding the kpoints mesh.'
            'Might cause energy difference.'
            )

    kpoints_mesh_sc[2] = kz // (n_layers // n_stacking)


    kpoints_sc = orm.KpointsData()
    kpoints_sc.set_kpoints_mesh(kpoints_mesh_sc)

    return kpoints_sc

def calculate_surface_area(cell):
    """
    Calculate the surface area of the cell
    """
    return la.norm(numpy.cross(cell[0], cell[1]))