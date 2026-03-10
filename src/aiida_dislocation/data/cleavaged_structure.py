from __future__ import annotations

import typing as ty

import numpy
from aiida.common.exceptions import ModificationNotAllowed
from aiida.orm import Data
from ase import Atoms
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from aiida_dislocation.data.gliding_systems import (
    GlidingPlaneConfig,
    GlidingSystem,
    get_gliding_system,
)
from aiida_dislocation.tools.structure_builder import build_atoms_surface
from aiida_dislocation.tools.structure_utils import get_strukturbericht, group_by_layers


class CleavagedStructure:
    """Helper for conventional and cleavaged structure generation."""

    def __init__(
        self,
        ase_atoms: Atoms,
        n_unit_cells: int,
        gliding_plane: ty.Optional[str] = None,
    ) -> None:
        self._ase_atoms = ase_atoms
        self._n_unit_cells = n_unit_cells
        if not gliding_plane:
            self._gliding_plane = self.gliding_system.default_plane
        else:
            self._gliding_plane = gliding_plane

    @property
    def unit_cell(self) -> Atoms:
        """Get the original unit-cell structure."""
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
        strukturbericht = get_strukturbericht(self.unit_cell)
        if not strukturbericht:
            raise ValueError('No match found in the provided list of prototypes.')
        return strukturbericht

    @property
    def gliding_system(self) -> GlidingSystem:
        """Get the gliding system instance."""
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
        """Calculate the surface area of the conventional cell."""
        cell = self.get_conventional_structure().cell
        return float(numpy.linalg.norm(numpy.cross(cell[0], cell[1])))

    def _get_effective_gliding_plane(self) -> str:
        """Return the configured gliding plane or the default detected one."""
        return self.gliding_plane or self.gliding_system.default_plane

    def _prepare_plane_data(self) -> GlidingPlaneConfig:
        """Get the plane configuration for the effective gliding plane."""
        return self.gliding_system.get_plane(self._get_effective_gliding_plane())

    def get_conventional_structure(
        self,
        P: ty.Optional[ty.Union[list[ty.Any], 'numpy.ndarray']] = None,
        print_info: bool = False,
    ) -> Atoms:
        """Generate the conventional structure."""
        if print_info:
            print(f'Strukturbericht {self.strukturbericht} detected')

        plane_config = self._prepare_plane_data()
        if P is None:
            P = plane_config.transformation_matrix
        else:
            P = numpy.array(P)

        return make_supercell(self.unit_cell, P)

    def get_cleavaged_structure(
        self,
        vacuum_spacing: float = 1.0,
        print_info: bool = False,
    ) -> Atoms:
        """Generate the cleavaged surface structure from the conventional cell."""
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

        return build_atoms_surface(
            conventional_structure,
            self.n_unit_cells,
            layers_dict,
            print_info=print_info,
            vacuum_spacing=vacuum_spacing,
        )


class CleavagedStructureData(Data):
    """Pure configuration node for cleavaged slab generation."""

    N_UNIT_CELLS_KEY = 'n_unit_cells'
    GLIDING_PLANE_KEY = 'gliding_plane'
    VACUUM_SPACINGS_KEY = 'vacuum_spacings'

    def __init__(
        self,
        n_unit_cells: ty.Optional[int] = None,
        gliding_plane: ty.Optional[str] = None,
        vacuum_spacings: ty.Optional[ty.Sequence[float]] = None,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_unit_cells is None and gliding_plane is None and vacuum_spacings is None:
            return

        resolved_n_unit_cells = 4 if n_unit_cells is None else int(n_unit_cells)
        resolved_gliding_plane = '' if gliding_plane is None else str(gliding_plane)
        resolved_vacuum_spacings = [float(value) for value in (vacuum_spacings or [1.0])]

        self._set_attribute(self.N_UNIT_CELLS_KEY, resolved_n_unit_cells)
        self._set_attribute(self.GLIDING_PLANE_KEY, resolved_gliding_plane)
        self._set_attribute(self.VACUUM_SPACINGS_KEY, resolved_vacuum_spacings)

    def _set_attribute(self, key: str, value: ty.Any) -> None:
        """Set an attribute before storing the node."""
        if self.is_stored:
            raise ModificationNotAllowed('`CleavagedStructureData` attributes cannot be modified after storing.')
        self.base.attributes.set(key, value)

    @property
    def n_unit_cells(self) -> int:
        """Return the configured number of repeated unit cells."""
        return int(self.base.attributes.get(self.N_UNIT_CELLS_KEY))

    @property
    def gliding_plane(self) -> str:
        """Return the configured gliding plane."""
        return str(self.base.attributes.get(self.GLIDING_PLANE_KEY, ''))

    @property
    def vacuum_spacings(self) -> list[float]:
        """Return the configured vacuum spacings."""
        return [float(value) for value in self.base.attributes.get(self.VACUUM_SPACINGS_KEY, [1.0])]

    def get_structure_builder(self, structure: 'orm.StructureData | Atoms') -> CleavagedStructure:
        """Return a helper bound to a specific structure."""
        from aiida import orm

        ase_atoms = structure.get_ase() if isinstance(structure, orm.StructureData) else structure
        effective_gliding_plane = self.gliding_plane or None
        return CleavagedStructure(
            ase_atoms=ase_atoms,
            n_unit_cells=self.n_unit_cells,
            gliding_plane=effective_gliding_plane,
        )
