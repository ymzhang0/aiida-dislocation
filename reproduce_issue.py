import sys
import numpy as np
from aiida_dislocation.tools.structure import read_structure_from_file, get_conventional_structure, get_faulted_structure

def reproduce():
    print("Loading Al structure...")
    try:
        # read_structure_from_file returns orm.StructureData
        al_struct_data = read_structure_from_file('Al')
        al_atoms = al_struct_data.get_ase()
    except Exception as e:
        print(f"Failed to load Al structure: {e}")
        return

    print("Getting conventional structure...")
    try:
        _, conventional_al = get_conventional_structure(al_atoms, gliding_plane='111')
    except Exception as e:
        print(f"Failed to get conventional structure: {e}")
        return
        
    print(f"Conventional structure layers: {len(conventional_al)}")

    print("Generating faulted structures...")
    try:
        # A1 111 general fault has 2 burger vectors.
        # We use nsteps=2 to minimize output but show progression.
        # n_unit_cells=2 ensures enough layers for interface (3, 4).
        _, result = get_faulted_structure(
            conventional_al,
            fault_mode='general',
            fault_type='general',
            gliding_plane='111',
            n_unit_cells=2, 
            nsteps=2
        )
    except Exception as e:
        print(f"Failed to generate faulted structures: {e}")
        import traceback
        traceback.print_exc()
        return

    if not result or 'structures' not in result:
        print("No structures returned.")
        return

    structures = result['structures']
    print(f"Generated {len(structures)} structures.")

    duplicates_found = False
    for i in range(len(structures) - 1):
        s1 = structures[i]['structure']
        s2 = structures[i+1]['structure']
        
        pos1 = s1.get_positions()
        pos2 = s2.get_positions()
        
        # Check if identical (allowing small float errors)
        if np.allclose(pos1, pos2, atol=1e-5):
            print(f"!!! DUPLICATE FOUND at index {i} and {i+1} !!!")
            print(f"Index {i} burger vector: {structures[i].get('burger_vector')}")
            print(f"Index {i+1} burger vector: {structures[i+1].get('burger_vector')}")
            duplicates_found = True
        else:
            diff = np.linalg.norm(pos1 - pos2)
            # print(f"Index {i} and {i+1} are different. Norm diff: {diff}")

    if duplicates_found:
        print("\nIssue Reproduced: Duplicate structures found.")
    else:
        print("\nIssue NOT Reproduced: No strict duplicates found.")

if __name__ == "__main__":
    reproduce()
