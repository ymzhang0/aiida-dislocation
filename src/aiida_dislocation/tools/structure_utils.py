import numpy
import typing as ty
import pathlib
from aiida import orm
from copy import deepcopy
from deprecated import deprecated
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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

def calculate_surface_area(ase_atoms) -> float:
    """
    Calculate the surface area of the structure (XY plane area).
    
    Args:
        ase_atoms: ASE Atoms object
        
    Returns:
        float: Surface area in Angstrom^2
    """
    cell = ase_atoms.get_cell()
    # Assuming surface is defined by vector 0 and 1 (standard for this package)
    return numpy.linalg.norm(numpy.cross(cell[0], cell[1]))
