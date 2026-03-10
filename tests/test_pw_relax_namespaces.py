"""Regression tests for ``PwRelaxWorkChain`` namespace compatibility."""

from __future__ import annotations

import pytest
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from ase.build import bulk

from aiida_dislocation.workflows.gsfe import GSFEWorkChain
from aiida_dislocation.workflows.layer_relax import RigidLayerRelaxWorkChain
from aiida_dislocation.workflows.surface import SurfaceEnergyWorkChain
from aiida_dislocation.workflows.usfe import USFEWorkChain


def _mock_pw_relax_builder_from_protocol(cls, *args, **kwargs):
    """Return a minimal ``PwRelaxWorkChain`` builder using the new namespaces."""
    return PwRelaxWorkChain.get_builder()


def _mock_pw_base_builder_from_protocol(cls, *args, **kwargs):
    """Return a minimal ``PwBaseWorkChain`` builder."""
    return PwBaseWorkChain.get_builder()


def _mock_layer_relax_builder_from_protocol(cls, *args, **kwargs):
    """Return a minimal ``RigidLayerRelaxWorkChain`` builder."""
    return RigidLayerRelaxWorkChain.get_builder()


@pytest.fixture
def aluminum_structure(aiida_profile_clean) -> orm.StructureData:
    """Return a simple aluminum FCC structure."""
    return orm.StructureData(ase=bulk('Al', 'fcc', a=4.05))


def _patch_sub_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch nested protocol builders so tests exercise only namespace wiring."""
    monkeypatch.setattr(
        PwRelaxWorkChain,
        'get_builder_from_protocol',
        classmethod(_mock_pw_relax_builder_from_protocol),
    )
    monkeypatch.setattr(
        PwBaseWorkChain,
        'get_builder_from_protocol',
        classmethod(_mock_pw_base_builder_from_protocol),
    )
    monkeypatch.setattr(
        RigidLayerRelaxWorkChain,
        'get_builder_from_protocol',
        classmethod(_mock_layer_relax_builder_from_protocol),
    )


def test_usfe_builder_uses_new_pw_relax_namespaces(monkeypatch, aluminum_structure) -> None:
    """Top-level SFE builders should target ``base`` and drop ``base_final_scf``."""
    _patch_sub_builders(monkeypatch)

    builder = USFEWorkChain.get_builder_from_protocol(object(), aluminum_structure)

    assert 'base' in builder.relax
    assert 'base_final_scf' not in builder.relax


def test_gsfe_builder_uses_new_pw_relax_namespaces(monkeypatch, aluminum_structure) -> None:
    """GSFE builder should consume the new ``PwRelaxWorkChain`` namespace layout."""
    _patch_sub_builders(monkeypatch)

    builder = GSFEWorkChain.get_builder_from_protocol(object(), aluminum_structure)

    assert 'base' in builder.relax
    assert 'base_final_scf' not in builder.relax


def test_surface_builder_uses_new_pw_relax_namespaces(monkeypatch, aluminum_structure) -> None:
    """Surface builder should consume the new ``PwRelaxWorkChain`` namespace layout."""
    _patch_sub_builders(monkeypatch)

    builder = SurfaceEnergyWorkChain.get_builder_from_protocol(object(), aluminum_structure)

    assert 'base' in builder.relax
    assert 'base_final_scf' not in builder.relax


def test_layer_relax_builder_uses_new_pw_relax_namespaces(monkeypatch, aluminum_structure) -> None:
    """Rigid layer relax builder should consume the new ``PwRelaxWorkChain`` namespace layout."""
    monkeypatch.setattr(
        PwRelaxWorkChain,
        'get_builder_from_protocol',
        classmethod(_mock_pw_relax_builder_from_protocol),
    )

    builder = RigidLayerRelaxWorkChain.get_builder_from_protocol(object(), aluminum_structure)

    assert 'base' in builder.relax
    assert 'base_final_scf' not in builder.relax
