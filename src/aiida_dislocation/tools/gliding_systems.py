import typing as ty
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                        ),
                        )
                    },
                # interface=(2, ),
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
                burger_vectors={
                    '010': (
                        (
                            (2, [0, 1, 0]),
                        ),
                        )
                    },
                # interface=(2, ),
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
            general=FaultConfig(
                possible=True,
                burger_vectors={
                    '110': (
                        (
                            (3, [1/3, 1/3, 0]),
                            (4, [1/3, 1/3, 0])
                        ),
                    ),
                },
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                        ),
                        )
                    },
                # interface=[2, ],
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                        ),
                        )
                    },
                # interface=[2,],
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
            general=FaultConfig(
                possible=True,
                burger_vectors={
                    '110': (
                        (
                             (3, [1/3, 1/3, 0]),
                            (4, [1/3, 1/3, 0])
                            )
                        )
                    },
                # interface=(3, 4),
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                        ),
                        )
                    },
                # interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [-1, 1, 1], [1, 0, 0]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors={
                    '010': (
                        (
                            (2, [0, 1, 0]),
                        ),
                        )
                    },
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
                burger_vectors={
                    '110': (
                        (
                            (6, [1/3, 1/3, 0]),
                            (7, [1/3, 1/3, 0])
                            ),
                        )
                    },
                # interface=(6, ),
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                        ),
                        )
                    },
                # interface=[2, ],
                nsteps = 10
            )
        )
        self._planes['011'] = GlidingPlaneConfig(
            transformation_matrix=[[0, 1, -1], [1, 0, 0], [0, 1, 1]],
            n_layers=2,
            general=FaultConfig(
                possible=True,
                burger_vectors={
                    '110': (
                        (
                            (2, [1, 1, 0]),
                        ),
                        ),
                    },
                # interface=[2, ],
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
                burger_vectors={
                    '110': (
                        (
                            (6, [1/3, 1/3, 0]),
                            ),
                        )
                    },
                # interface=(6, ),
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
                burger_vectors={
                    '100': (
                        (
                            (4, [1, 0, 0]),
                            ),
                        ),
                    '010': (
                        (
                            (4, [0, 1, 0]),
                            ),
                        )
                    },
                # interface=(4, 4),
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                            ),
                        ),
                    '010': (
                        (
                            (2, [0, 1, 0]),
                            ),
                        )
                    },
                # interface=(2, 2),
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
                burger_vectors={
                    '110': (
                        (
                            (9, [1/3, 1/3, 0]),
                            ),
                        )
                    },
                # interface=9,
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
                burger_vectors={
                    '110': (
                        (
                            (4, [1, 1, 0]),
                            ),
                        )
                    },
                # interface=(4,),
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
                burger_vectors={
                    '100': (
                        (
                            (2, [1, 0, 0]),
                            ),
                        ),
                    '010': (
                        (
                            (2, [0, 1, 0]),
                            ),
                        ),
                    '210': (
                        (
                            (2, [2, 1, 0]),
                            ),
                        )
                    },
                # interface=(2, ),
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
                burger_vectors={
                    '110': (
                        (
                            (9, [1/3, 1/3, 0]),
                            ),
                        )
                    },
                # interface=9,
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
