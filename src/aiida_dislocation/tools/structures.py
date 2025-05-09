from aiida import orm
from math import sqrt, acos, pi, ceil
import numpy
import logging
from ase.spacegroup import get_spacegroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger('aiida.workflow.dislocation')


def get_unstable_faulted_structure_and_kpoints(
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

    multiplicity = 1
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
            ])

            structure_sc.set_cell(supercell)


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

            multiplicity = n_layers

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

            multiplicity = n_layers

            surface_area = numpy.linalg.norm(numpy.cross(supercell[0], supercell[1]))

        elif gliding_plane == '111':

            if n_layers % 3 != 0:
                raise ValueError('n_layers must be a multiple of 3 for 111 gliding plane')

            C = C * sqrt(2) * sqrt(3) * (n_layers - 1) / 3
            supercell = numpy.array([
                [A,   0.0,           0.0],
                [A * 1/2, A * sqrt(3)/2, 0.0],
                [0.0, 0.0,           C],
            ])

            structure_sc.set_cell(supercell)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/3, 1/3],],
                'C': [[ATOM, 2/3, 2/3],],
            }

            falted_stacking = 'ABC' * int(n_layers/3)

            falted_stacking = falted_stacking[:3] + falted_stacking[4:]

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

            multiplicity = n_layers - 1

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
            ])

            structure_sc.set_cell(supercell)

            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/2, 1/2],],
                'C': [[ATOM, 0.0, 1/2],],
                'D': [[ATOM, 1/2, 1/2],],
            }

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

            multiplicity = n_layers

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
            ])

            structure_sc.set_cell(supercell)

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

            multiplicity = n_layers

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
            ])
            
            structure_sc.set_cell(supercell)
            
            planer_config = {
                'A': [[ATOM, 0.0, 0.0],],
                'B': [[ATOM, 1/3, 1/3],],
                'C': [[ATOM, 2/3, 2/3],],
            }
            
            falted_stacking = 'ABC' * int(n_layers/3)

            falted_stacking = falted_stacking[:3] + falted_stacking[4:]
            
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
            
            multiplicity = n_layers - 1

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
            ])

            structure_sc.set_cell(supercell)

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
            kpoints_sc = get_kpoints_mesh_for_supercell(
                kpoints_uc, 
                n_layers, 
                2
            )

            multiplicity = n_layers

            surface_area = numpy.abs(numpy.linalg.norm(numpy.cross(supercell[0], supercell[1])))
            
        elif gliding_plane == '110':

            A = A
            C = A * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0      , 0.0],
                [0.0, sqrt(2)*A, 0.0],
                [0.0, 0.0      , C  ],
            ])
            
            structure_sc.set_cell(supercell)

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.0, 0.5]], 
                'B': [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.5, 0.0]],
            }

            falted_stacking = 'AB' * int(n_layers/4) + 'BA' * int(n_layers/4)

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

            multiplicity = n_layers

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
            ])

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
            C = A * sqrt(2) * n_layers / 4

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A,   0.0],
                [0.0, 0.0, C  ],
            ])

            structure_sc.set_cell(supercell)

            planer_config = {
                'A':    [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.5, 0.5],], 
                'B':    [[ATOM_3, 0.5, 0.0], ], 
                'C':    [[ATOM_1, 0.5, 0.5], [ATOM_2, 0.0, 0.0]], 
                'D':    [[ATOM_3, 0.0, 0.5]], 
            }

            falted_stacking = 'ABCD' * int(n_layers/8) + 'CDAB' * int(n_layers/8)

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

            multiplicity = n_layers / 2

            surface_area = A * A
                  
        elif gliding_plane == '110':
            
            A = A
            C = A * n_layers / 2

            supercell = numpy.array([
                [A  , 0.0, 0.0],
                [0.0, A * sqrt(2),   0.0],
                [0.0, 0.0, C  ],
            ])

            structure_sc.set_cell(supercell)

            planer_config = {
                'A': [[ATOM_1, 0.0, 0.0], [ATOM_2, 0.0, 0.5],], 
                'B': [[ATOM_3, 0.5, 1/4], ], 
                'C': [[ATOM_1, 0.0, 0.5], [ATOM_2, 0.0, 0.0]], 
                'D': [[ATOM_3, 0.5, 3/4]], 
            }

            falted_stacking = 'ABCD' * int(n_layers/8) + 'CDAB' * int(n_layers/8)

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
            
            multiplicity = n_layers / 2

            surface_area = A * A * sqrt(2)
            
            
            
            
            
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


    return (structure_sc, kpoints_sc, multiplicity, surface_area)
    


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

    if kpoints_mesh_sc[2] % n_stacking != 0:
        logger.warning(
            'kpoints mesh is not compatible with supercell dimensions. '
            'Ceiling the kpoints mesh.'
            'Might cause energy difference.'
            )
        
        kpoints_mesh_sc[2] = ceil(kpoints_mesh_sc[2])

    kpoints_mesh_sc[2] /= (n_layers/n_stacking)

    kpoints_sc = orm.KpointsData()
    kpoints_sc.set_kpoints_mesh(kpoints_mesh_sc)

    return kpoints_sc