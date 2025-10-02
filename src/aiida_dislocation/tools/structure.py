from tkinter.constants import NONE
from aiida import orm
from math import sqrt, acos, pi, ceil
import numpy
import logging
from ase import Atoms
from ase.spacegroup import get_spacegroup
from ase.build import make_supercell
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pathlib
import typing as ty
import copy
import itertools

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
_GLIDING_SYSTEMS = {
    'A1': {
        '011':{
            'transformation_matrix': [
                [0, 1, -1],
                [-1, 1, 1],
                [1, 0, 0]
            ],
            'transformation_matrix_c': [
                [0, 1, -1],
                [-1, 1, 1],
                [1, 0, 0]
            ],
            'n_layers': 2,
            'intrinsic_removal': [2],
            'extrinsic_removal': None,
            'unstable_removal': None,
        },
        '111':{
            'transformation_matrix': [
                [1, -1, 0],
                [1, 0, -1],
                [1, 1, 1]
            ],
            'transformation_matrix_c': [
                [1, -1, 0],
                [1, 1, -2],
                [1, 1, 1]
            ],
            'n_layers': 3,
            'intrinsic_removal': [3],
            'extrinsic_removal': [3, 5],
        }
    },
    'A2': {
        '011':{
            'transformation_matrix': [
                [0, 1, 0],
                [0, 0, 1],
                [2, 1, 1]
            ],
            'transformation_matrix_c': [
                [0, 1, -1],
                [0, 1, 1],
                [2, 1, 1]
            ],
            'n_layers': 2,
            'intrinsic_removal': [2],
            'extrinsic_removal': None,
        },
        '111':{
            'transformation_matrix': [
                [-1, 1, 0],
                [-1, 0, 1],
                [2, 2, 2]
        ],
            'transformation_matrix_c': [
                [-2, 1, 1],
                [0, -1, 1],
                [2, 2, 2]
        ],
            'n_layers': 6,
            'intrinsic_removal': [6, 7],
            'extrinsic_removal': [6, 7, 10, 11],
        },
    },
    'B1': {
        '011':{
            'transformation_matrix': [
                [0, 1, -1],
                [-1, 1, 1],
                [1, 0, 0]
            ],
            'n_layers': 2,
            'intrinsic_removal': [2],
            'extrinsic_removal': None,
        },
        '111':{
            'transformation_matrix': [
                [1, -1, 0],
                [1, 0, -1],
                [1, 1, 1]
            ],
            'transformation_matrix_c': [
                [1, -1, 0],
                [1, 1, -2],
                [1, 1, 1]
            ],
            'n_layers': 6,
            'intrinsic_removal': [3],
            'extrinsic_removal': [3, 5],
        }
    },
    'C1_b':{
        '011':{
            'transformation_matrix': [
                [0, 1, -1],
                [-1, 1, 1],
                [1, 0, 0]
            ],
            'n_layers': 6,
            'intrinsic_removal': [2],
            'extrinsic_removal': None,
        },
        '111':{
            'transformation_matrix': [
                [1, -1, 0],
                [1, 0, -1],
                [1, 1, 1]
            ],
            'transformation_matrix_c': [
                [1, -1, 0],
                [1, 1, -2],
                [1, 1, 1]
            ],
            'n_layers': 9,
            'intrinsic_removal': [3],
            'extrinsic_removal': [3, 5],
        }
    },
    'E_21':{
        '011':{
            'transformation_matrix': [
                [1, 1, 0],
                [-1, 1, 0],
                [0, 0, 2]
            ],
            'n_layers': 4,
            'intrinsic_removal': [2],
            'extrinsic_removal': None,
        },
        '111':{
            'transformation_matrix': [
                [1, -1, 0],
                [1, 0, -1],
                [1, 1, 1]
            ],
            'n_layers': 6,
            'intrinsic_removal': [6, 7],
            'extrinsic_removal': [6, 7, 10, 11],
        }
    }
}

_IMPLEMENTED_SLIPPING_SYSTEMS = {
    'A1': {
        'info': 'FCC element crystal <space group #225, prototype Cu>. '
                'Usually, the gliding plane is 111.',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[010]',
                    'faulting_possible': True,
                    },
            '110': {'stacking': 'AB',
                    'slipping_direction': '1/2[112]',
                    'faulting_possible': True,
                    },
            '111': {'stacking': 'ABC',
                    'slipping_direction': '1/2[110]',
                    'faulting_possible': True,
                    },
        }
    },
    'A2': {
        'info': 'FCC element crystal <space group #227, prototype V>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[110]',
                    'faulting_possible': True,
                    },
            '110': {'stacking': 'AB',
                    'slipping_direction': '1/2[001]',
                    'faulting_possible': True,
                    },
            '111': {'stacking': 'ABC',
                    'slipping_direction': '1/2[011]',
                    'faulting_possible': True,
                    },
        }
    },
    'A15': {
        'info': 'A3B crystal <space group #223, prototype Nb3Sn>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[110]',
                    'faulting_possible': True,
                    },
            '110': {'stacking': 'AB',
                    'slipping_direction': '1/2[001]',
                    'faulting_possible': True,
                    },
            '111': {'stacking': 'ABC',
                    'slipping_direction': '1/2[011]',
                    'faulting_possible': True,
                    },
        }
    },
    'B1': {
        'info': 'FCC element crystal <space group #225, prototype NaCl>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[010]',
                    'faulting_possible': True,
                    },
            '110': {'stacking': 'AB',
                    'slipping_direction': '1/2[112]',
                    'faulting_possible': True,
                    },
        }
    },
    'B2': {
        'info': 'FCC element crystal <space group #229, prototype CsCl>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[010]',
                    'faulting_possible': True,
                    },
        }
    },
    'C1': {
        'info': 'We are doing pyrite-type structure. <space group #205, prototype FeS2>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'ABCD',
                    'slipping_direction': '1/2[100]',
                    'faulting_possible': True,
                    },
        }
    },
    'C1b': {
        'info': 'We are doing half-heusler-type structure. <space group #216, prototype MgSiAl>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'ABCD',
                    'slipping_direction': '1/2[100]',
                    'faulting_possible': True,
                    },
            '110': {'stacking': 'AB',
                    'slipping_direction': '1/2[110]',
                    'faulting_possible': True,
                    },
            '111': {'stacking': 'ABC',
                    'slipping_direction': '1/2[111]',
                    'faulting_possible': True,
                    },
        }
    },
    'E21': {
        'info': 'We are doing perovskite-type structure. <space group #221, prototype BaTiO3>. '
                'I don\'t know the usual gliding plane. ',
        'possible_gliding_planes': {
            '100': {'stacking': 'AB',
                    'slipping_direction': '1/2[010]',
                    'faulting_possible': True,
                    },
        }
    },
}

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
        'TaRu3C',
        'Nb3Sn',
        'AsTe',
        'NbCoSb',
        'MoN',
        'MgB2',
        'TaSe2'
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

    if n_unit_cells < 1 or type(n_unit_cells) != int:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}")
    
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
    print_info = False,
    ):

    atoms = Atoms()
    
    stacking_order = n_unit_cells * ''.join(layers_dict.keys())
    if n_unit_cells < 1 or type(n_unit_cells) != int:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}")
    if any(layer > len(stacking_order) for layer in removed_layers):
        raise ValueError(f"Removed layers {removed_layers} is greater than the number of layers {len(layers_dict)}")

    zs = [value['z']/n_unit_cells + layer/n_unit_cells for layer in range(n_unit_cells) for value in layers_dict.values()]

    removed_spacing = 0.0
    faulted_stacking = "".join([char for i, char in enumerate(stacking_order) if i not in removed_layers])
    
    for removed_layer in removed_layers:
        spacing = zs[removed_layer] - zs[removed_layer - 1]
        removed_spacing += spacing
        for z in zs[removed_layer:]:
            z -= spacing

    for _ in range(len(removed_layers)):
        zs.pop()

    zs = [z / (1-removed_spacing) for z in zs]
    if print_info:
        print(zs)
        print(faulted_stacking)
    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= (1-removed_spacing) * n_unit_cells
    atoms.set_cell(new_cell)
    for layer_label, z in zip(faulted_stacking, zs):
        for atom in layers_dict[layer_label]['atoms']:
            scaled_position = atom.scaled_position
            scaled_position[-1] = z
            atom.position = scaled_position @ new_cell
            atoms.append(atom)
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
    if n_unit_cells < 1 or type(n_unit_cells) != int:
        raise ValueError(f"Invalid number of unit cells {n_unit_cells}")

    # Taking 3 unit cells of 3-layer unit cell as an example
    # Firstly, we place an 'ABC' stacking as a substrate.
    
    spacings = [
        (layers_dict[label]['z'] - layers_dict[prev_label]['z'])*z_norm
        for label, prev_label in zip(stacking_order_uc[1:], stacking_order_uc[:-1])
        ]
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
    
    spacings.pop()
    if print_info:
        print(spacings)
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


    faulted_stacking = stacking_order_uc[:-1] + stacking_order_uc_r * n_unit_cells + (stacking_order_uc * (n_unit_cells-1))[1:]
    if print_info:
        print(faulted_stacking)
    z_dialation = len(stacking_order) / len(layers_dict)
    new_cell = ase_atoms_uc.cell.array.copy()
    new_cell[-1] *= z_dialation
    atoms.set_cell(new_cell)
    for layer_label, z in zip(faulted_stacking, zs):
        for atom in layers_dict[layer_label]['atoms']:
            atom.scaled_position[-1] = z
            atoms.append(atom)

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
        "B_h": read_structure_from_file('MoN').get_pymatgen(),   # Arsenic Telluride (AsTe)
        "A15": read_structure_from_file('Nb3Sn').get_pymatgen(),        # Nb3Sn (Nb3Sn)
        "C1_b": read_structure_from_file('NbCoSb').get_pymatgen(),            # Gold-Copper (AuCu3)
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
        gliding_plane=None,
        P = None,
        n_unit_cells = 3,
        burger_vector = None,
        vacuum_ratio = 0,
        print_info = False,
    ):

    strukturbericht = get_strukturbericht(ase_atoms_uc)
    if not strukturbericht:
        raise ValueError('No match found in the provided list of prototypes.')
        
    if print_info:
        print(f'Strukturbericht {strukturbericht} detected')
    if not gliding_plane:
        gliding_plane = '111'

    gliding_system = _GLIDING_SYSTEMS[strukturbericht][gliding_plane]
    if not P:
        P = gliding_system['transformation_matrix']

    ase_atoms_t = make_supercell(ase_atoms_uc, P)
    layers_dict = group_by_layers(ase_atoms_t)
    if len(layers_dict) != gliding_system.get('n_layers'):
        raise ValueError(
            f'We found {len(layers_dict)} layers.'
            'This either comes from the wrong initial structure, or wrong indication of structure type, or wrong transformation.')

    structures = AttributeDict({
        'unfaulted': ase_atoms_t,
    # ...ABC*(A)BCABC...
        'intrinsic': build_atoms_from_stacking_removal(
        ase_atoms_t, n_unit_cells, gliding_system['intrinsic_removal'], layers_dict, print_info = print_info,
        ),
    # ...ABC*(A)B(C)ABC...
        'extrinsic': build_atoms_from_stacking_removal(
        ase_atoms_t, n_unit_cells, gliding_system['extrinsic_removal'], layers_dict, print_info = print_info,
        ) if 'extrinsic' in gliding_system else NONE,
    # ...ABC*BACBA*BCABC
        'unstable': build_atoms_from_stacking_removal(
        ase_atoms_t, n_unit_cells, gliding_system['unstable_removal'], layers_dict, print_info = print_info,
        ) if 'unstable' in gliding_system else NONE,
        'twinning': build_atoms_from_stacking_mirror(
        ase_atoms_t, n_unit_cells, layers_dict, print_info = print_info,
        ) if gliding_system.get('n_layers') > 2 else NONE,
        'cleavaged': build_atoms_surface(
        ase_atoms_t, n_unit_cells, layers_dict, print_info = print_info,
        ),
    })

    # if strukturbericht == 'A2':
    #     if print_info:
    #         print('Strukturbericht A2 detected')
    #     if not gliding_plane:
    #         gliding_plane = '111'
    #     if not P:
    #         P = _GLIDING_SYSTEMS[strukturbericht][gliding_plane]['transformation_matrix']
    #     n_layers = _GLIDING_SYSTEMS[strukturbericht][gliding_plane]['n_layers']
    #     ase_atoms_t = make_supercell(ase_atoms_uc, P)
    #     layers_dict = group_by_layers(ase_atoms_t)
    #     if len(layers_dict) != n_layers:
    #         raise ValueError(
    #             f'We found {len(layers_dict)} layers.'
    #             'This either comes from the wrong initial structure, or wrong indication of structure type, or wrong transformation.')
    #     gliding_system = _GLIDING_SYSTEMS[strukturbericht][gliding_plane]

    #     structures = AttributeDict(
    #         {
    #             'unfaulted': ase_atoms_t,
    #         # ...ABCDEF*(AB)CDEF...
    #             'intrinsic': build_atoms_from_stacking_removal(
    #             ase_atoms_t, n_unit_cells, 
    #             gliding_system['intrinsic_removal'], 
    #             layers_dict, print_info = print_info,
    #             ),
    #         # ...ABCDEF*(AB)CD(EF)...
    #             'extrinsic': build_atoms_from_stacking_removal(
    #             ase_atoms_t, n_unit_cells, 
    #             gliding_system['extrinsic_removal'], 
    #             layers_dict, print_info = print_info,
    #             ) if 'extrinsic' in gliding_system else NONE,
    #         # ...ABCDEF*EDCAB*ABCDEF
    #             'twinning': build_atoms_from_stacking_mirror(
    #             ase_atoms_t, n_unitcf_cells, layers_dict, print_info = print_info,
    #             ) if n_layers > 2 else None,
    #             'cleavaged': build_atoms_surface(
    #             ase_atoms_t, n_unit_cells, layers_dict, print_info = print_info,
    #             ),
    #         }
    #     )
    return (strukturbericht, structures)

def get_unstable_faulted_structure_and_kpoints_old(
        structure_uc: orm.StructureData,
        kpoints_uc: orm.KpointsData,
        n_layers: int,
        slipping_system: orm.List,
    ) -> tuple[orm.StructureData, orm.KpointsData, int]:

    """
    Get a supercell of the structure
    """

    structure_type, gliding_plane, slipping_direction = slipping_system

    structure_sc = orm.StructureData()
    kpoints_sc = orm.KpointsData()

    structure_cl = orm.StructureData()
    kpoints_cl = orm.KpointsData()

    multiplicity = 1
    multiplicity_cl = 1
    surface_area = None

    if structure_type == 'A1':

        # A, B, C are the length from [0 0 0] to [1/2, 1/2, 0]
        # The lattice constant for conventional cell is A * sqrt(2)

        A, B, C = structure_uc.cell_lengths
        alpha, beta, gamma = structure_uc.cell_angles

        if (
            (max(A, B, C) - min(A, B, C)) > 1e-5
            or
            any(abs(angle - 60.0) > 1e-5 for angle in (alpha, beta, gamma))
            ):
            logger.info('Cell length or angles differ more than 1e-5.')

        ATOM = structure_uc.get_kind_names()[0]

        if gliding_plane == '100':

            # A / sqrt(2) is half of the lattice constant for conventional cell

            A = A
            C = C * sqrt(2) * n_layers / 2

            supercell = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 0.5, 0.5],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM, 0.0, 0.5],]
                planer_config['D'] = [[ATOM, 0.5, 0.0],]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )


            cleavaged_stacking = 'AB' * int(n_layers/2)

            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )

            multiplicity = n_layers
            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

        elif gliding_plane == '110':
            A = A
            C = C * n_layers / 2

            supercell = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A*sqrt(2),   0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A*sqrt(2),   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)
            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 0.5, 0.5],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM, 0.0, 0.5],]
                planer_config['D'] = [[ATOM, 0.5, 0.0],]

            faulted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)
            cleavaged_stacking = 'AB' * int(n_layers/2)
            for idx, st in enumerate(faulted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )

            multiplicity = n_layers
            multiplicity_cl = n_layers
            surface_area = numpy.linalg.norm(numpy.cross(supercell[0], supercell[1]))

        elif gliding_plane == '111':

            if n_layers % 3 != 0:
                raise ValueError('n_layers must be a multiple of 3 for 111 gliding plane')

            C = C * sqrt(2) * sqrt(3) * (n_layers - 1) / 3
            supercell = numpy.array([
                [A,   0.0,           0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0,           C],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A,   0.0, 0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/3, 1/3],],
                'C': [[ATOM, 2/3, 2/3],],
            }

            faulted_stacking = 'ABC'  + 'DEF' * int(n_layers/3-1)

            faulted_stacking = faulted_stacking[:-1]

            if slipping_direction:
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['E'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
                planer_config['F'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['C']
                ]
            else:
                planer_config['D'] = [[ATOM, 1/3, 1/3],]
                planer_config['E'] = [[ATOM, 2/3, 2/3],]
                planer_config['F'] = [[ATOM, 0.0, 0.0],]

            for idx, st in enumerate(faulted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/(n_layers - 1)])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers - 1,
                3
            )

            multiplicity = n_layers - 1

            cleavaged_stacking = 'ABC' * int(n_layers/3)

            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                3
            )

            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

    elif structure_type == 'A2':

        ## A is the length from [0 0 0] to [1/2, 1/2, 1/2]
        ## Lattice constant for conventional cell is A * 2 / sqrt(3)
        A, B, C = structure_uc.cell_lengths
        alpha, beta, gamma = structure_uc.cell_angles

        if (
            (max(A, B, C) - min(A, B, C)) > 1e-5
            or
            any(abs(angle - acos(-1/3) * 180/pi) > 1e-5 for angle in (alpha, beta, gamma))
            ):
            logger.info('Cell length or angles differ more than 1e-5.')

        ATOM = structure_uc.get_kind_names()[0]


        if gliding_plane == '100':
            if n_layers % 2 != 0:
                raise ValueError('n_layers must be a multiple of 2 for 100 gliding plane')
            A = A * 2 / sqrt(3)
            C = A * n_layers / 2

            supercell = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A,   0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/2, 1/2],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM, 0.0, 1/2],]
                planer_config['D'] = [[ATOM, 1/2, 0],]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #


            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
                )

            cleavaged_stacking = 'AB' * int(n_layers/2)
            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )

            multiplicity = n_layers
            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

        if gliding_plane == '110':
            if n_layers % 2 != 0:
                raise ValueError('n_layers must be a multiple of 2 for 110 gliding plane')
            A = A * 2 / sqrt(3)
            C = A * sqrt(2) * n_layers / 2

            supercell = numpy.array([
                [ A / 2, A / sqrt(2), 0.0],
                [-A / 2, A / sqrt(2), 0.0],
                [0.0,    0.0,         C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A / 2, A / sqrt(2), 0.0],
                [-A / 2, A / sqrt(2),   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/2, 1/2],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM, 0.0, 1/2],]
                planer_config['D'] = [[ATOM, 1/2, 0.0],]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #
            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            cleavaged_stacking = 'AB' * int(n_layers/2)
            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #
            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )

            multiplicity = n_layers
            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

        elif gliding_plane == '111':
            if n_layers % 3 != 0:
                raise ValueError('n_layers must be a multiple of 3 for 111 gliding plane')

            A = A * 2 * sqrt(2) / sqrt(3)
            C = C * 2 * (n_layers - 1) / 4

            supercell = numpy.array([
                [A,   0.0, 0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A,   0.0, 0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 2/3, 2/3],],
                'C': [[ATOM, 1/3, 1/3],],
            }

            if slipping_direction:
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['E'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
                planer_config['F'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['C']
                ]
            else:
                planer_config['D'] = [[ATOM, 1/3, 1/3],]
                planer_config['E'] = [[ATOM, 0.0, 0.0],]
                planer_config['F'] = [[ATOM, 2/3, 2/3],]

            falted_stacking = 'ABC'  + 'DEF' * int(n_layers/3-1)

            falted_stacking = falted_stacking[:-1]

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/(n_layers - 1)])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                3
            )

            cleavaged_stacking = 'ABC' * int(n_layers/3)
            for idx, st in enumerate(cleavaged_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])  #
            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                3
            )

            multiplicity = n_layers - 1
            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

    elif structure_type == 'A15':
        pass
    elif structure_type == 'B1':

        ## A is the length from [0 0 0] to [1/2, 1/2, 0]
        ## Lattice constant for conventional cell is A * sqrt(2)

        A, B, C = structure_uc.cell_lengths
        alpha, beta, gamma = structure_uc.cell_angles

        if (
            (max(A, B, C) - min(A, B, C)) > 1e-5
            or
            any(abs(angle - 60.0) > 1e-5 for angle in (alpha, beta, gamma))
            ):
            logger.info('Cell length or angles differ more than 1e-5.')

        elements_wyckoff_symbols = get_elements_for_wyckoff_symbols(structure_uc)

        ATOM_1, ATOM_2 = (elements_wyckoff_symbols[k] for k in ['a', 'b'])

        if gliding_plane == '100':

            A = A
            C = A * sqrt(2) * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.5, 0.5]],
                'B': [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.0, 0.0]],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM_1, 0.5, 0.0], [ATOM_2, 0.0, 0.5]]
                planer_config['D'] = [[ATOM_1, 0.0, 0.5], [ATOM_2, 0.5, 0.0]]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            falted_stacking_cl = 'AB' * int(n_layers/2)

            for idx, st in enumerate(falted_stacking_cl):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])
                    #
            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            multiplicity = n_layers

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )
            multiplicity_cl = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))

        elif gliding_plane == '110':

            A = A
            C = A * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0      , 0.0],
                [0.0, sqrt(2)*A, 0.0],
                [0.0, 0.0      , C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A * sqrt(2),   0.0],
                [0.0, 0.0, C*2 ],
            ])

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.0, 0.5],],
                'B': [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.5, 0.0],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM_1, 0.5, 0.0], [ATOM_2, 0.5, 0.5]]
                planer_config['D'] = [[ATOM_1, 0.0, 0.5], [ATOM_2, 0.0, 0.0]]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            falted_stacking_cl = 'AB' * int(n_layers/2)

            for idx, st in enumerate(falted_stacking_cl):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            multiplicity = n_layers

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )
            multiplicity_cl = n_layers

            surface_area = A * A * sqrt(2)


        elif gliding_plane == '111':

            if n_layers % 12 != 0:
                raise ValueError('n_layers must be a multiple of 12 for 111 gliding plane')

            A = A
            C = A * n_layers / sqrt(6)

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)


            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0]],
                'c': [[ATOM_2, 2/3, 2/3]],
                'B': [[ATOM_1, 1/3, 1/3]],
                'a': [[ATOM_1, 0.0, 0.0]],
                'C': [[ATOM_1, 2/3, 2/3]],
                'b': [[ATOM_1, 1/3, 1/3]],
            }

            falted_stacking = 'AcBaCb' * int(n_layers/6)
            falted_stacking = falted_stacking[:6] + falted_stacking[7:]
            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                6
            )

            multiplicity = n_layers

            surface_area = A * A * sqrt(3) / 2

    elif structure_type == 'C1b':
        A, B, C = structure_uc.cell_lengths
        alpha, beta, gamma = structure_uc.cell_angles

        if (
            (max(A, B, C) - min(A, B, C)) > 1e-5
            or
            any(abs(angle - 60.0) > 1e-5 for angle in (alpha, beta, gamma))
            ):
            logger.info('Cell length or angles differ more than 1e-5.')


        ## TODO: Check the order of the atoms

        ## ATOM_1 is the atom at [0, 0, 0]
        ## ATOM_2 is the atom at [0.5, 0.5, 0.5]
        ## ATOM_3 is the atom at [0.25, 0.25, 0.25]

        elements_wyckoff_symbols = get_elements_for_wyckoff_symbols(structure_uc)

        ATOM_1, ATOM_2, ATOM_3 = (elements_wyckoff_symbols[k] for k in ['a', 'b', 'c'])


        if gliding_plane == '100':

            ## The length of the conventional cell is A * sqrt(2)
            if n_layers % 8 != 0:
                raise ValueError('n_layers must be a multiple of 8 for 100 gliding plane')

            A = A
            C = A * sqrt(2) * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C  ],
            ], dtype=numpy.float64)

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C*2 ],
            ], dtype=numpy.float64)

            structure_cl.set_cell(supercell_cl)


            planer_config = {
                'A':    [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.5, 0.5],],
                'B':    [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.0, 0.0] ],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM_1, 0.5, 0.0], [ATOM_2, 0.0, 0.5]]
                planer_config['D'] = [[ATOM_1, 0.0, 0.5], [ATOM_2, 0.5, 0.0]]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            falted_stacking_cl = 'AB' * int(n_layers/2)

            for idx, st in enumerate(falted_stacking_cl):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])
            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            multiplicity = n_layers

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )
            multiplicity_cl = n_layers

            surface_area = A * A

        elif gliding_plane == '110':

            A = A
            C = C * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A * sqrt(2),   0.0],
                [0.0, 0.0, C  ],
            ])

            structure_sc.set_cell(supercell)

            supercell_cl = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A * sqrt(2),   0.0],
                [0.0, 0.0, C*2 ],
            ])

            structure_cl.set_cell(supercell_cl)

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.0, 0.5],],
                'B': [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.5, 0.0],],
            }

            if slipping_direction:
                planer_config['C'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['A']
                ]
                planer_config['D'] = [
                    [ATOM, x + slipping_direction[0], y + slipping_direction[1]]
                    for ATOM, x, y in planer_config['B']
                ]
            else:
                planer_config['C'] = [[ATOM_1, 0.5, 0.0], [ATOM_2, 0.5, 0.5]]
                planer_config['D'] = [[ATOM_1, 0.0, 0.5], [ATOM_2, 0.05, 0.0]]

            falted_stacking = 'AB' * int(n_layers/4) + 'CD' * int(n_layers/4)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #

            falted_stacking_cl = 'AB' * int(n_layers/2)

            for idx, st in enumerate(falted_stacking_cl):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers/2])
                    position_cart = position_frac @ supercell_cl
                    structure_cl.append_atom(position=position_cart, symbols=value[0])

            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                2
            )

            multiplicity = n_layers

            kpoints_cl = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers*2,
                2
            )
            multiplicity_cl = n_layers

            surface_area = A * A * sqrt(2)

        elif gliding_plane == '111':

            A = A
            C = C * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0, C  ],
            ])

            structure_sc.set_cell(supercell)

    elif structure_type == 'E21':
        A, B, C = structure_uc.cell_lengths
        alpha, beta, gamma = structure_uc.cell_angles

        if (
            (max(A, B, C) - min(A, B, C)) > 1e-5
            or
            any(abs(angle - 60.0) > 1e-5 for angle in (alpha, beta, gamma))
            ):
            logger.info('Cell length or angles differ more than 1e-5.')

        ATOM_1, ATOM_2, ATOM_3 = structure_uc.get_kind_names()


        if gliding_plane == '110':

            C = A * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0      , 0.0],
                [0.0, sqrt(2)*A, 0.0],
                [0.0, 0.0      , C  ],
            ])

            structure_sc.set_cell(supercell)

            ## TODO: Check the order of the atoms
            elements_wyckoff_symbols = elements_wyckoff_symbols(structure_uc)
            ATOM_1, ATOM_2, ATOM_3 = structure_uc.get_kind_names()

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.0, 1/2], [ATOM_3, 1/2, 1/2]],
                'B': [[ATOM_2, 1/4, 0.0], [ATOM_2, 3/4, 0.0]],
                'C': [[ATOM_2, 0.5, 0.5], [ATOM_3, 0.0, 1/2], [ATOM_1, 1/2, 0.0]],
                }

            falted_stacking = 'ABCB' * int(n_layers/8) + 'CBAB' * int(n_layers/8)

            for idx, st in enumerate(falted_stacking):
                for value in planer_config[st]:
                    position_frac = numpy.array([*value[1:], idx/n_layers])
                    position_cart = position_frac @ supercell
                    structure_sc.append_atom(position=position_cart, symbols=value[0])  #
            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc,
                n_layers,
                4
            )


    return (structure_sc, kpoints_sc, multiplicity, surface_area, structure_cl, kpoints_cl, multiplicity_cl)

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