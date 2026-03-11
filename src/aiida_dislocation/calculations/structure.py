"""Calcfunctions for provenance-aware structure generation."""

from __future__ import annotations

import typing as ty

from ase import Atoms
from aiida import orm
from aiida.engine import calcfunction

from aiida_dislocation.data.cleavaged_structure import CleavagedStructureData
from aiida_dislocation.data.faulted_structure import (
    FaultedStructureData,
    GeneralFaultStructurePoint,
    GeneralFaultStructureResult,
)


def _normalize_faulted_structure_points(
    generated: ty.Any,
    fault_type: str,
) -> list[GeneralFaultStructurePoint]:
    """Normalize outputs from ``FaultedStructure.get_faulted_structure`` to a point list."""
    if generated is None:
        return []

    if isinstance(generated, Atoms):
        return [{
            'label': f'sfe_{fault_type}_000',
            'structure': generated,
            'burger_vector': [],
            'total_cell_shift': [],
            'interface_slips': {},
            'direction_name': fault_type,
            'step_index': 0,
        }]

    if isinstance(generated, list):
        normalized: list[GeneralFaultStructurePoint] = []
        for index, item in enumerate(generated):
            if isinstance(item, Atoms):
                normalized.append({
                    'label': f'sfe_{fault_type}_{index:03d}',
                    'structure': item,
                    'burger_vector': [],
                    'total_cell_shift': [],
                    'interface_slips': {},
                    'direction_name': fault_type,
                    'step_index': index,
                })
                continue

            if not isinstance(item, dict) or 'structure' not in item:
                raise TypeError('Unsupported faulted structure payload returned by `FaultedStructureData`.')

            normalized.append({
                'label': str(item.get('label', f'sfe_{item.get("direction_name", fault_type)}_{index:03d}')),
                'structure': item['structure'],
                'burger_vector': [float(value) for value in item.get('burger_vector', [])],
                'total_cell_shift': [float(value) for value in item.get('total_cell_shift', item.get('burger_vector', []))],
                'interface_slips': {
                    int(interface): [float(value) for value in interface_shift]
                    for interface, interface_shift in item.get('interface_slips', {}).items()
                },
                'direction_name': item.get('direction_name', fault_type),
                'step_index': int(item.get('step_index', index)),
            })

        return normalized

    if isinstance(generated, dict):
        normalized = []
        general_result = ty.cast(GeneralFaultStructureResult, generated)

        for direction_name, steps in general_result.items():
            for step_index, entry in sorted(steps.items()):
                metadata = entry['metadata']
                normalized.append({
                    'label': metadata['label'],
                    'structure': entry['structure'],
                    'burger_vector': [float(value) for value in metadata['burger_vector']],
                    'total_cell_shift': [float(value) for value in metadata['total_cell_shift']],
                    'interface_slips': {
                        int(interface): [float(value) for value in interface_shift]
                        for interface, interface_shift in metadata['interface_slips'].items()
                    },
                    'direction_name': direction_name,
                    'step_index': int(step_index),
                })

        return normalized

    raise TypeError('Unsupported faulted structure payload returned by `FaultedStructureData`.')


def _format_spacing_key(vacuum_spacing: float) -> str:
    """Return a Dict-safe key for a vacuum spacing."""
    return f'{vacuum_spacing:.6f}'.replace('.', '_')


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
        'surface_area': orm.Float(float(builder.surface_area)),
    }

    for point in normalized_points:
        key = point['label']
        structure_node = orm.StructureData(ase=point['structure'])
        structure_node.label = key
        structure_node.base.extras.set_many({
            'label': key,
            'direction_name': point['direction_name'],
            'step_index': int(point['step_index']),
            'burger_vector': [float(value) for value in point['burger_vector']],
            'total_cell_shift': [float(value) for value in point['total_cell_shift']],
            'interface_slips': {
                str(interface): [float(value) for value in interface_shift]
                for interface, interface_shift in point['interface_slips'].items()
            },
        })
        outputs[key] = structure_node

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
