from ase import Atoms
from copy import deepcopy
import typing as ty
import numpy
from deprecated import deprecated
import itertools

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

def update_faults(faults, interface, burger_vector):
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
