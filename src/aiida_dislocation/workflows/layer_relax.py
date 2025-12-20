"""SFE Spacing WorkChain - handles looping over additional_spacings for SFE calculations."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, while_, append_
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_dislocation.tools import get_faulted_structure
from math import ceil
import numpy
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

class RigidLayerRelaxWorkChain(ProtocolMixin, WorkChain):
    """WorkChain for looping over additional_spacings and performing SFE calculations.
    
    This workchain handles:
    - Looping over additional_spacings list
    - For each spacing: generating faulted structure, setting up kpoints, running calculation
    - Collecting results for all spacings
    
    It is designed to be called as a sub-workchain from RigidLayerWorkChain
    or other workflows that need to perform SFE calculations for multiple spacings.
    """
    
    _NAMESPACE = 'layer_relax'
    _RELAX_NAMESPACE = 'relax'
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.input('structure', valid_type=orm.StructureData, required=True,
                   help='The conventional structure for generating faulted structures.')
        spec.input('layer_spacings', valid_type=orm.List, required=True,
                   help='List of layer spacings to evaluate.')
        spec.input('fault_type', valid_type=orm.Str, required=True,
                   help="Fault type: 'intrinsic', 'unstable', or 'extrinsic'.")
        spec.input('fault_method', valid_type=orm.Str, required=False,
                   default=lambda: orm.Str('removal'),
                   help="Fault method: 'removal' or 'vacuum'.")
        spec.input('vacuum_ratio', valid_type=orm.Float, required=False,
                   default=lambda: orm.Float(0.1),
                   help='Vacuum ratio when using vacuum method.')
        spec.input('gliding_plane', valid_type=orm.Str, required=False,
                   help='Gliding plane direction.')
        spec.input('n_repeats', valid_type=orm.Int, required=True,
                   help='Number of unit cells to repeat.')
        spec.input('kpoints', valid_type=orm.KpointsData, required=True,
                   help='The kpoints mesh for the relaxation calculation.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._RELAX_NAMESPACE,
            exclude=('structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )
        
        
        spec.outline(
            cls.setup,
            while_(cls.should_run_relax)(
                cls.setup_supercell_kpoints,
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.results,
        )
        
        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace=cls._RELAX_NAMESPACE,
            namespace_options={'required': False}
        )
        
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message='The sub-process failed.',
        )


    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml
        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(
            cls,
            code,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (code, structure, protocol)

        builder = cls.get_builder()

        # Set up the sub-workchains

        sub_builder = PwRelaxWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._RELAX_NAMESPACE, {}),
        )
        # sub_builder.pop('structure', None)
        sub_builder.pop('clean_workdir', None)
        sub_builder.pop('kpoints', None)
        sub_builder.pop('kpoints_distance', None)

        sub_builder['base_relax'].pop('kpoints', None)
        sub_builder['base_relax'].pop('kpoints_distance', None)
        sub_builder.pop('base_init_relax', None)

        builder[cls._RELAX_NAMESPACE]._data = sub_builder._data

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        return builder

    def setup(self):
        """Initialize context for spacing loop."""
        self.ctx.iteration = 1
        self.ctx.layer_spacings = self.inputs.layer_spacings.get_list().copy()
    
    def should_run_relax(self):
        """Check if there are more spacings to process."""
        if self.ctx.layer_spacings == []:
            return False

        # Get current spacing
        current_spacing = self.ctx.layer_spacings.pop(0)
        self.ctx.current_spacing = current_spacing
        
        # Generate faulted structure for this spacing
        fault_type = self.inputs.fault_type.value
        fault_method = self.inputs.fault_method.value.lower() if self.inputs.fault_method.value else 'removal'
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        
        if fault_method == 'removal':
            _, faulted_structure_data = get_faulted_structure(
                self.inputs.structure.get_ase(),
                fault_type=fault_type,
                additional_spacing=current_spacing,
                gliding_plane=gliding_plane,
                n_unit_cells=self.inputs.n_repeats.value,
                fault_mode='removal',
            )
        elif fault_method == 'vacuum':
            vacuum_ratio = float(self.inputs.vacuum_ratio.value)
            _, faulted_structure_data = get_faulted_structure(
                self.inputs.structure.get_ase(),
                fault_type=fault_type,
                additional_spacing=current_spacing,
                gliding_plane=gliding_plane,
                n_unit_cells=self.inputs.n_repeats.value,
                fault_mode='vacuum',
                vacuum_ratio=vacuum_ratio,
            )
        else:
            raise ValueError(f"Unsupported fault method: {fault_method}")
        
        # Validate structure
        if faulted_structure_data is None or not faulted_structure_data.get('structures'):
            self.report(f'Faulted structure not available for spacing {current_spacing}. Skipping.')
            return False
        
        # Extract structure
        actual_structure = faulted_structure_data['structures'][0].get('structure')
        if actual_structure is None:
            self.report(f'Faulted structure is missing for spacing {current_spacing}. Skipping.')
            return False
        
        # Store structure and calculate multiplier
        self.ctx.current_structure_ase = actual_structure
        self.ctx.current_structure = orm.StructureData(ase=actual_structure)
        
        # Calculate multiplier
        from ase.formula import Formula
        formula = Formula(actual_structure.get_chemical_formula())
        _, multiplier = formula.reduce()
        self.ctx.current_multiplier = multiplier
        
        return True
    
    def setup_supercell_kpoints(self):
        """Setup kpoints for the current rigid layer structure."""
        # Calculate kpoints based on z-ratio between faulted and conventional structures
        
        z_ratio = self.ctx.current_structure.get_ase().cell.cellpar()[2] / self.inputs.structure.get_ase().cell.cellpar()[2]
        kpoints_mesh = self.inputs.kpoints.get_kpoints_mesh()[0]
        
        kpoints_relax = orm.KpointsData()
        kpoints_relax.set_kpoints_mesh(kpoints_mesh[:2] + [ceil(kpoints_mesh[2] / z_ratio)])
        
        self.ctx.kpoints_relax = kpoints_relax
        self.report(f'Kpoints mesh for rigid layer relaxation (spacing {self.ctx.current_spacing}): {kpoints_relax.get_kpoints_mesh()[0]}')
    
    def run_relax(self):
        """Run the rigid layer relaxation calculation for current spacing."""
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._RELAX_NAMESPACE
            )
        )
        
        inputs.structure = self.ctx.current_structure
        inputs.base_relax.kpoints = self.ctx.kpoints_relax
        inputs.metadata.call_link_label = f'relax_{self.ctx.iteration}'
        
        # Apply fault_method specific settings
        fault_method = self.inputs.fault_method.value.lower() if self.inputs.fault_method.value else 'removal'
        parameters = inputs.base_relax.pw.parameters.get_dict()
        
        if fault_method == 'vacuum':
            parameters['CELL']['cell_dofree'] = 'fixc'
        
        if hasattr(self.ctx, 'nbnd') and self.ctx.nbnd:
            parameters['SYSTEM']['nbnd'] = int(self.ctx.nbnd)
        
        inputs.base_relax.pw.parameters = orm.Dict(parameters)
        
        # Apply fixed coordinates for relaxation
        settings = inputs.base_relax.pw.settings.get_dict()
        settings['USE_FRACTIONAL'] = True
        
        FIXED_COORDS = numpy.full_like(
            self.ctx.current_structure.get_ase().get_positions(),
            fill_value=True,
            dtype=bool
        )
        settings['FIXED_COORDS'] = FIXED_COORDS.tolist()
        inputs.base_relax.pw.settings = orm.Dict(settings)
        
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for spacing: {self.ctx.current_spacing}.')
        
        return {f"workchain_relax": append_(running)}
    
    def inspect_relax(self):
        """Inspect the rigid layer relaxation calculation results for current spacing."""
        workchain = self.ctx.workchain_relax[-1]
        self.ctx.iteration += 1
        
        if not workchain.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{workchain.pk}> for spacing {self.ctx.current_spacing} "
                f"failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        
        self.report(f'PwRelaxWorkChain<{workchain.pk}> for spacing {self.ctx.current_spacing} finished successfully.')
        
        # Extract number of bands for next iteration
        if 'output_parameters' in workchain.outputs:
            self.ctx.nbnd = workchain.outputs.output_parameters.get('number_of_bands')
        


    def results(self):
        """Output collected results."""
        pass