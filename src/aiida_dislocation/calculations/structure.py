"""Calcfunctions for provenance-aware structure generation."""

from __future__ import annotations

import typing as ty

from ase import Atoms
from aiida import orm
from aiida.engine import calcfunction

from aiida_dislocation.data.faulted_structure import FaultedStructureData, GeneralFaultStructurePoint


class FaultedStructureMetadata(ty.TypedDict):
    """Metadata stored for each generated faulted structure."""

    point_index: int
    structure_uuid: str
    direction_name: str
    path_index: int
    step_index: int
    burger_vector: list[float]


def _normalize_faulted_structure_points(
    generated: ty.Any,
    fault_type: str,
) -> list[GeneralFaultStructurePoint]:
    """Normalize outputs from ``FaultedStructure.get_faulted_structure`` to a point list."""
    if generated is None:
        return []

    if isinstance(generated, Atoms):
        return [{
            'structure': generated,
            'burger_vector': [],
            'direction_name': fault_type,
            'path_index': 0,
            'step_index': 0,
        }]

    if isinstance(generated, list):
        normalized: list[GeneralFaultStructurePoint] = []
        for index, item in enumerate(generated):
            if isinstance(item, Atoms):
                normalized.append({
                    'structure': item,
                    'burger_vector': [],
                    'direction_name': fault_type,
                    'path_index': 0,
                    'step_index': index,
                })
                continue

            if not isinstance(item, dict) or 'structure' not in item:
                raise TypeError('Unsupported faulted structure payload returned by `FaultedStructureData`.')

            normalized.append({
                'structure': item['structure'],
                'burger_vector': [float(value) for value in item.get('burger_vector', [])],
                'direction_name': item.get('direction_name', fault_type),
                'path_index': int(item.get('path_index', 0)),
                'step_index': int(item.get('step_index', index)),
            })

        return normalized

    raise TypeError('Unsupported faulted structure payload returned by `FaultedStructureData`.')


@calcfunction
def generate_faulted_structures(
    faulted_data: FaultedStructureData,
    fault_mode: orm.Str,
    fault_type: orm.Str,
) -> dict[str, orm.Data]:
    """Generate provenance-tracked faulted structures from ``FaultedStructureData``."""
    generated = faulted_data.get_faulted_structure(
        fault_mode=fault_mode.value,
        fault_type=fault_type.value,
    )

    normalized_points = _normalize_faulted_structure_points(generated, fault_type.value)
    if not normalized_points:
        raise ValueError('No faulted structures could be generated for the requested configuration.')

    metadata: dict[str, FaultedStructureMetadata] = {}
    outputs: dict[str, orm.Data] = {
        'structure_map': orm.Dict(dict={}),
        'conventional_structure': orm.StructureData(ase=faulted_data.get_conventional_structure()),
        'cleavaged_structure': orm.StructureData(ase=faulted_data.get_cleavaged_structure()),
        'surface_area': orm.Float(float(faulted_data.surface_area)),
    }

    for index, point in enumerate(normalized_points, start=1):
        key = f'sfe_idx_{index:03d}'
        structure_node = orm.StructureData(ase=point['structure'])
        outputs[key] = structure_node
        metadata[key] = {
            'point_index': index,
            'structure_uuid': structure_node.uuid,
            'direction_name': point['direction_name'],
            'path_index': point['path_index'],
            'step_index': point['step_index'],
            'burger_vector': [float(value) for value in point['burger_vector']],
        }

    outputs['structure_map'] = orm.Dict(dict=metadata)
    return outputs
