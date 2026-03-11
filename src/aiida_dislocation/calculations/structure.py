"""Calcfunctions for provenance-aware structure generation."""

from __future__ import annotations

import typing as ty

from ase import Atoms
from aiida import orm
from aiida.engine import calcfunction

from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.data.faulted_structure import FaultedStructureData, GeneralFaultStructurePoint


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


def _format_spacing_key(vacuum_spacing: float) -> str:
    """Return a Dict-safe key for a vacuum spacing."""
    return f'{vacuum_spacing:.6f}'.replace('.', '_')


def _format_fault_key(index: int) -> str:
    """Return the standard output key for a generated faulted structure."""
    return f'sfe_idx_{index:03d}'


@calcfunction
def generate_faulted_structures(
    structure: orm.StructureData,
    faulted_data: FaultedStructureData,
    fault_mode: orm.Str,
    fault_type: orm.Str,
) -> dict[str, orm.Data]:
    """Generate provenance-tracked faulted structures from structure and faulted configuration."""
    builder = faulted_data.get_structure_builder(structure)
    generated = builder.get_faulted_structure(
        fault_mode=fault_mode.value,
        fault_type=fault_type.value,
    )

    normalized_points = _normalize_faulted_structure_points(generated, fault_type.value)
    if not normalized_points:
        raise ValueError('No faulted structures could be generated for the requested configuration.')

    outputs: dict[str, orm.Data] = {
        'conventional_structure': orm.StructureData(ase=builder.get_conventional_structure()),
        'cleavaged_structure': orm.StructureData(ase=builder.get_cleavaged_structure()),
        'surface_area': orm.Float(float(builder.surface_area)),
    }

    for index, point in enumerate(normalized_points, start=1):
        key = _format_fault_key(index)
        outputs[key] = orm.StructureData(ase=point['structure'])
        outputs[f'burger_vector_{key}'] = orm.List(list=[float(value) for value in point['burger_vector']])
        outputs[f'direction_name_{key}'] = orm.Str(point['direction_name'])
        outputs[f'path_index_{key}'] = orm.Int(int(point['path_index']))
        outputs[f'step_index_{key}'] = orm.Int(int(point['step_index']))

    return outputs


@calcfunction
def generate_cleavaged_structures(
    structure: orm.StructureData,
    cleavaged_data: CleavagedStructureData,
) -> dict[str, orm.Data]:
    """Generate provenance-tracked slab structures from primitive structure and cleavaged configuration."""
    builder = cleavaged_data.get_structure_builder(structure)
    vacuum_spacings = cleavaged_data.vacuum_spacings

    if not vacuum_spacings:
        raise ValueError('No vacuum spacings configured for cleavaged structure generation.')

    outputs: dict[str, orm.Data] = {
        'conventional_structure': orm.StructureData(ase=builder.get_conventional_structure()),
        'surface_area': orm.Float(float(builder.surface_area)),
    }

    for vacuum_spacing in vacuum_spacings:
        spacing_key = _format_spacing_key(float(vacuum_spacing))
        slab_key = f'slab_{spacing_key}'
        spacing_output_key = f'vacuum_spacing_{spacing_key}'

        if slab_key in outputs or spacing_output_key in outputs:
            raise ValueError(f'Duplicate vacuum spacing key generated for {vacuum_spacing}.')

        outputs[spacing_output_key] = orm.Float(float(vacuum_spacing))
        outputs[slab_key] = orm.StructureData(
            ase=builder.get_cleavaged_structure(vacuum_spacing=vacuum_spacing)
        )

    return outputs
