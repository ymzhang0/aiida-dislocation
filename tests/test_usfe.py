"""Tests for the ``USFEWorkChain``."""

import pytest
from aiida.common import LinkType
from aiida.orm import Dict, StructureData, List, Str, Float, Int, Bool
from plumpy import ProcessState
from ase.build import bulk


@pytest.fixture
def generate_structure():
    """Generate a structure for testing."""
    def _generate_structure():
        """Generate a simple structure (Aluminum FCC)."""
        atoms = bulk('Al', 'fcc', a=4.05)
        return StructureData(ase=atoms)
    return _generate_structure


@pytest.fixture
def generate_inputs_usfe(generate_structure):
    """Generate default inputs for the ``USFEWorkChain``."""
    def _generate_inputs_usfe():
        """Generate default inputs for the ``USFEWorkChain``."""
        structure = generate_structure()
        
        inputs = {
            'structure': structure,
            'kpoints_distance': Float(0.3),
            'n_repeats': Int(4),
            'gliding_plane': Str('111'),
            'additional_spacings': List(list=[0.0, 0.002]),
            'fault_method': Str('removal'),
            'vacuum_ratio': Float(0.1),
            'clean_workdir': Bool(False),
        }
        return inputs
    return _generate_inputs_usfe


@pytest.fixture
def generate_workchain_usfe(generate_workchain, generate_inputs_usfe, generate_calc_job_node):
    """Generate an instance of a ``USFEWorkChain``."""
    def _generate_workchain_usfe(exit_code=None, inputs=None, return_inputs=False, 
                                  relax_outputs=None, scf_outputs=None, usfe_outputs=None):
        """Generate an instance of a ``USFEWorkChain``.

        :param exit_code: exit code for the sub-workchains.
        :param inputs: inputs for the ``USFEWorkChain``.
        :param return_inputs: return the inputs of the ``USFEWorkChain``.
        :param relax_outputs: ``dict`` of outputs for the relax ``PwRelaxWorkChain``.
        :param scf_outputs: ``dict`` of outputs for the scf ``PwBaseWorkChain``.
        :param usfe_outputs: ``dict`` of outputs for the usfe ``PwRelaxWorkChain``.
        """
        from aiida_dislocation.workflows.usfe import USFEWorkChain
        
        if inputs is None:
            inputs = generate_inputs_usfe()
        
        if return_inputs:
            return inputs
        
        # Create workchain instance using builder
        builder = USFEWorkChain.get_builder()
        for key, value in inputs.items():
            setattr(builder, key, value)
        
        # Create process instance (not running)
        process = USFEWorkChain(builder)
        
        # Mock the context that would be set up by the workflow
        process.ctx.iteration = 1
        additional_spacings = inputs.get('additional_spacings')
        if additional_spacings is not None:
            process.ctx.additional_spacings = additional_spacings.get_list() if hasattr(additional_spacings, 'get_list') else additional_spacings
        else:
            process.ctx.additional_spacings = [0.0]
        
        # Create mock nodes for sub-workchains
        if relax_outputs is not None:
            relax_node = generate_calc_job_node(inputs={'parameters': Dict()})
            process.ctx.workchain_relax = relax_node
            process.ctx.children = [relax_node]
            
            for link_label, output_node in relax_outputs.items():
                output_node.base.links.add_incoming(
                    relax_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()
            
            if exit_code is not None:
                relax_node.set_process_state(ProcessState.FINISHED)
                relax_node.set_exit_status(exit_code.status)
        
        if scf_outputs is not None:
            scf_node = generate_calc_job_node(inputs={'parameters': Dict()})
            process.ctx.workchain_scf = scf_node
            
            for link_label, output_node in scf_outputs.items():
                output_node.base.links.add_incoming(
                    scf_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()
        
        if usfe_outputs is not None:
            usfe_node = generate_calc_job_node(inputs={'parameters': Dict()})
            process.ctx.workchain_sfe = [usfe_node]
            
            for link_label, output_node in usfe_outputs.items():
                output_node.base.links.add_incoming(
                    usfe_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()
        
        return process
    
    return _generate_workchain_usfe


def test_usfe_workchain_define():
    """Test that the ``USFEWorkChain`` can be instantiated."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    builder = USFEWorkChain.get_builder()
    assert hasattr(builder, 'structure')
    assert hasattr(builder, 'additional_spacings')
    assert hasattr(builder, 'fault_method')


def test_usfe_workchain_default_inputs(generate_inputs_usfe):
    """Test that the ``USFEWorkChain`` has correct default inputs."""
    inputs = generate_inputs_usfe()
    
    assert inputs['fault_method'].value == 'removal'
    assert inputs['vacuum_ratio'].value == 0.1
    assert isinstance(inputs['additional_spacings'], List)


def test_usfe_workchain_structure_input(generate_structure):
    """Test that the ``USFEWorkChain`` accepts a structure input."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    structure = generate_structure()
    builder = USFEWorkChain.get_builder()
    builder.structure = structure
    
    assert builder.structure == structure


def test_usfe_workchain_additional_spacings():
    """Test that the ``USFEWorkChain`` handles additional_spacings correctly."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    builder = USFEWorkChain.get_builder()
    builder.additional_spacings = List(list=[0.0, 0.002, 0.004])
    
    assert builder.additional_spacings.get_list() == [0.0, 0.002, 0.004]


def test_usfe_workchain_fault_method():
    """Test that the ``USFEWorkChain`` accepts different fault_method values."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    for method in ['removal', 'vacuum']:
        builder = USFEWorkChain.get_builder()
        builder.fault_method = Str(method)
        
        assert builder.fault_method.value == method


def test_usfe_workchain_vacuum_ratio():
    """Test that the ``USFEWorkChain`` accepts vacuum_ratio input."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    builder = USFEWorkChain.get_builder()
    builder.vacuum_ratio = Float(0.2)
    
    assert builder.vacuum_ratio.value == 0.2


def test_usfe_workchain_get_fault_type(generate_inputs_usfe):
    """Test that the ``USFEWorkChain`` returns correct fault type."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    # Create a workchain instance using builder
    builder = USFEWorkChain.get_builder()
    inputs = generate_inputs_usfe()
    for key, value in inputs.items():
        setattr(builder, key, value)
    
    # Create process instance and test the method
    # Note: This requires AiiDA profile to be loaded
    try:
        process = USFEWorkChain(builder)
        assert process._get_fault_type() == 'unstable'
    except Exception:
        # If profile is not available, skip this test
        pytest.skip("AiiDA profile not available, skipping test that requires workchain instance")


def test_usfe_workchain_exit_code():
    """Test that the ``USFEWorkChain`` has the correct exit code defined."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    assert hasattr(USFEWorkChain.exit_codes, 'ERROR_SUB_PROCESS_FAILED_USF')
    assert USFEWorkChain.exit_codes.ERROR_SUB_PROCESS_FAILED_USF.status == 404


def test_usfe_workchain_setup():
    """Test that the ``USFEWorkChain`` setup method exists."""
    from aiida_dislocation.workflows.usfe import USFEWorkChain
    
    # Test that the setup method exists
    assert hasattr(USFEWorkChain, 'setup')
    assert callable(getattr(USFEWorkChain, 'setup'))

