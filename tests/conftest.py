"""Project-wide pytest fixtures & hooks.

Docs: https://docs.pytest.org/en/stable/how-to/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

# Try to load AiiDA test fixtures, but don't fail if pgtest is not available
try:
    import pgtest  # noqa: F401
    pytest_plugins = ['aiida.manage.tests.pytest_fixtures']
except ImportError:
    # If pgtest is not available, we'll skip tests that require a database
    # For simple builder tests, we don't need the full AiiDA test environment
    pass


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
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """
    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory
        
        if inputs is None:
            inputs = {}
        
        manager = get_manager()
        runner = manager.get_runner()
        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)
        return process.prepare_for_submission(folder)
    
    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(fixture_localhost, tmp_path_factory):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""
    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name='base', computer=None, test_name=None, inputs=None, attributes=None, retrieve_temporary=None
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string
        
        if computer is None:
            computer = fixture_localhost
        
        filepath_folder = None
        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(entry_point_name[len('quantumespresso.') :], test_name)
            filepath_folder = os.path.join(basepath, 'parsers', 'fixtures', filename)
            filepath_input = os.path.join(filepath_folder, 'aiida.in')
        
        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)
        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set('input_filename', 'aiida.in')
        node.base.attributes.set('output_filename', 'aiida.out')
        node.base.attributes.set('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)
        
        if attributes:
            node.base.attributes.set_many(attributes)
        
        if filepath_folder and inputs is None:
            from qe_tools.exceptions import ParsingError
            from aiida_quantumespresso.tools.pwinputparser import PwInputFile
            
            inputs = {}
            try:
                with open(filepath_input, encoding='utf-8') as input_file:
                    parsed_input = PwInputFile(input_file.read())
            except (ParsingError, FileNotFoundError):
                pass
            else:
                inputs['structure'] = parsed_input.get_structuredata()
                inputs['parameters'] = orm.Dict(parsed_input.namelists)
        
        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})
            for name, option in options.items():
                node.set_option(name, option)
            
            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(input_node, link_type=LinkType.INPUT_CALC, link_label=link_label)
        
        node.store()
        
        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            dirpath = Path(dirpath)
            filepaths = []
            for filename in filenames:
                filepaths.extend(Path(filepath_folder).glob(filename))
            for filepath in filepaths:
                shutil.copy(filepath, dirpath / filepath.name)
        
        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.base.repository.put_object_from_tree(filepath_folder)
            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filepath in filepaths:
                    retrieved.delete_object(filepath.name)
            retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
            retrieved.store()
            
            remote_folder = orm.RemoteData(computer=computer, remote_path=tmp_path_factory.mktemp('cj-tmp').as_posix())
            remote_folder.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='remote_folder')
            remote_folder.store()
        
        return node
    
    return _generate_calc_job_node
