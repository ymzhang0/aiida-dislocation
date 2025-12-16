"""Project-wide pytest fixtures & hooks.

Docs: https://docs.pytest.org/en/stable/how-to/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pytest

# Load AiiDA test fixtures
pytest_plugins = ['aiida.manage.tests.pytest_fixtures']

from aiida.orm import CalcJobNode, Dict


@pytest.fixture
def readability_counts() -> bool:
    return True


@pytest.fixture
def generate_workchain():
    """Generate a workchain instance for testing."""
    def _generate_workchain(entry_point, inputs):
        """Generate a workchain instance.
        
        :param entry_point: Entry point of the workchain.
        :param inputs: Inputs for the workchain.
        """
        from aiida.plugins import WorkflowFactory
        
        WorkChain = WorkflowFactory(entry_point)
        builder = WorkChain.get_builder()
        
        # Set inputs
        for key, value in inputs.items():
            setattr(builder, key, value)
        
        # Create a process instance (not running)
        process = WorkChain(builder)
        return process
    
    return _generate_workchain


@pytest.fixture
def generate_calc_job_node():
    """Generate a CalcJobNode for testing."""
    def _generate_calc_job_node(inputs=None):
        """Generate a CalcJobNode.
        
        :param inputs: Inputs for the calculation.
        """
        from aiida.plugins import CalculationFactory
        
        # Use a simple calculation type for testing
        PwCalculation = CalculationFactory('quantumespresso.pw')
        
        # Create a node (not stored)
        node = CalcJobNode(process_type=PwCalculation.build_process_type())
        if inputs:
            for key, value in inputs.items():
                node.set_attribute(f'input_{key}', value)
        
        return node
    
    return _generate_calc_job_node
