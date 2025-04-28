from aiida import orm
from ase import Atoms
from ase.build import make_supercell
import numpy

def group_by_plane(
    ase_atoms: Atoms,
    P,
    cut_plane: list[float],
    tol: float = 1e-6
    ):

    grouped_by_plane = {}
    
    supercell = make_supercell(ase_atoms, P)

    cut_plane = numpy.dot(cut_plane, ase_atoms.cell)

    c = numpy.linalg.norm(cut_plane)
    
    for atom in supercell:
        z_position = numpy.dot(atom.position, cut_plane) / c
        match = next(
            (k for k in grouped_by_plane.keys() 
            if numpy.isclose(k, z_position, atol=tol)),
            None
        )
        if match is None:
            grouped_by_plane[z_position] = [atom]
        else:
            grouped_by_plane[match].append(atom)

    return grouped_by_plane

def plot_layer(
    ase_atoms: Atoms,
    grouped_layer,
    ax, 
    x,
    y,
    color,
    symbol,
    ):
    
    ax.set_xlabel(f"[{x[0]}{x[1]}{x[2]}] ($\AA$)")
    ax.set_ylabel(f"[{y[0]}{y[1]}{y[2]}] ($\AA$)")
    
    x = numpy.dot(x, ase_atoms.cell)
    y = numpy.dot(y, ase_atoms.cell)
    
    
    xs = []
    ys = []
    for atom in grouped_layer:
        if atom.symbol == symbol:
            positions = atom.position
            xs.append(numpy.dot(positions, x) / numpy.linalg.norm(x))
            ys.append(numpy.dot(positions, y) / numpy.linalg.norm(y))
            
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.scatter(xs, ys, s=50, color=color)
    
    ax.set_aspect('equal')

