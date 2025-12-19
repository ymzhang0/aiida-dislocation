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
def generate_structure():
    """Return a ``StructureData`` representing either bulk silicon or a water molecule."""

    def _generate_structure(structure_id='silicon'):
        """Return a ``StructureData`` representing bulk silicon or a snapshot of a single water molecule dynamics.

        :param structure_id: identifies the ``StructureData`` you want to generate. Either 'silicon' or 'water'.
        """
        from aiida.orm import StructureData

        if structure_id.startswith('silicon'):
            name1 = 'Si0' if structure_id.endswith('kinds') else 'Si'
            name2 = 'Si1' if structure_id.endswith('kinds') else 'Si'
            param = 5.43
            cell = [[param / 2.0, param / 2.0, 0], [param / 2.0, 0, param / 2.0], [0, param / 2.0, param / 2.0]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si', name=name1)
            structure.append_atom(position=(param / 4.0, param / 4.0, param / 4.0), symbols='Si', name=name2)
        elif structure_id == 'cobalt-prim':
            cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Co', name='Co')
        elif structure_id == 'water':
            structure = StructureData(cell=[[5.29177209, 0.0, 0.0], [0.0, 5.29177209, 0.0], [0.0, 0.0, 5.29177209]])
            structure.append_atom(position=[12.73464656, 16.7741411, 24.35076238], symbols='H', name='H')
            structure.append_atom(position=[-29.3865565, 9.51707929, -4.02515904], symbols='H', name='H')
            structure.append_atom(position=[1.04074437, -1.64320127, -1.27035021], symbols='O', name='O')
        elif structure_id == 'uranium':
            param = 5.43
            cell = [[param / 2.0, param / 2.0, 0], [param / 2.0, 0, param / 2.0], [0, param / 2.0, param / 2.0]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='U', name='U')
            structure.append_atom(position=(param / 4.0, param / 4.0, param / 4.0), symbols='U', name='U')
        elif structure_id == '2D-xy-arsenic':
            cell = [[3.61, 0, 0], [-1.80, 3.13, 0], [0, 0, 21.3]]
            structure = StructureData(cell=cell, pbc=(True, True, False))
            structure.append_atom(position=(1.804, 1.042, 11.352), symbols='As', name='As')
            structure.append_atom(position=(0, 2.083, 9.960), symbols='As', name='As')
        elif structure_id == '1D-x-carbon':
            cell = [[4.2, 0, 0], [0, 20, 0], [0, 0, 20]]
            structure = StructureData(cell=cell, pbc=(True, False, False))
            structure.append_atom(position=(0, 0, 0), symbols='C', name='C')
        elif structure_id == '1D-y-carbon':
            cell = [[20, 0, 0], [0, 4.2, 0], [0, 0, 20]]
            structure = StructureData(cell=cell, pbc=(False, True, False))
            structure.append_atom(position=(0, 0, 0), symbols='C', name='C')
        elif structure_id == '1D-z-carbon':
            cell = [[20, 0, 0], [0, 20, 0], [0, 0, 4.2]]
            structure = StructureData(cell=cell, pbc=(False, False, True))
            structure.append_atom(position=(0, 0, 0), symbols='C', name='C')
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return structure

    return _generate_structure


@pytest.fixture
def generate_kpoints():
    """Return a `KpointsData` node."""

    def _generate_kpoints(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints


@pytest.fixture
def generate_inputs_usfe(generate_structure):
    """Generate default inputs for a `USFEWorkChain`."""

    def _generate_inputs_usfe():
        """Generate default inputs for a `USFEWorkChain`."""
        from aiida.orm import Float, Int, Str, List, Bool

        structure = generate_structure()
        
        return {
            'structure': structure,
            'kpoints_distance': Float(0.3),
            'n_repeats': Int(4),
            'gliding_plane': Str('111'),
            'additional_spacings': List(list=[0.0, 0.002]),
            'fault_method': Str('removal'),
            'vacuum_ratio': Float(0.1),
            'clean_workdir': Bool(False),
        }

    return _generate_inputs_usfe


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        return instantiate_process(runner, process_class, **inputs)

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
        from aiida.common import LinkType
        from aiida.orm import Dict
        from aiida_dislocation.workflows.usfe import USFEWorkChain
        from plumpy import ProcessState

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
