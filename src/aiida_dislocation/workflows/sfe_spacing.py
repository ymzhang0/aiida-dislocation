"""SFE Spacing WorkChain - handles looping over additional_spacings for SFE calculations."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, append_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_dislocation.tools import get_faulted_structure
from math import ceil
import numpy


class SfeSpacingWorkChain(WorkChain):
    """WorkChain for looping over additional_spacings and performing SFE calculations.
    
    This workchain handles:
    - Looping over additional_spacings list
    - For each spacing: generating faulted structure, setting up kpoints, running calculation
    - Collecting results for all spacings
    
    It is designed to be called as a sub-workchain from SFEBaseWorkChain
    or other workflows that need to perform SFE calculations for multiple spacings.
    """
    
    _NAMESPACE = 'sfe_spacing'
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.input('conventional_structure', valid_type=orm.StructureData, required=True,
                   help='The conventional structure for generating faulted structures.')
        spec.input('additional_spacings', valid_type=orm.List, required=True,
                   help='List of additional spacings to evaluate.')
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
        spec.input('kpoints_scf_mesh', valid_type=orm.List, required=True,
                   help='The kpoints mesh from SCF calculation.')
        spec.input('workchain_type', valid_type=orm.Str, required=False,
                   default=lambda: orm.Str('PwRelaxWorkChain'),
                   help='Type of workchain to use: PwRelaxWorkChain or PwBaseWorkChain.')
        
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._NAMESPACE,
            exclude=('structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )
        
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=f'{cls._NAMESPACE}_base',
            exclude=('pw.structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` (used when workchain_type is PwBaseWorkChain).'
            }
        )
        
        spec.outline(
            cls.setup,
            while_(cls.should_run_sfe)(
                cls.setup_supercell_kpoints,
                cls.run_sfe,
                cls.inspect_sfe,
            ),
            cls.results,
        )
        
        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace=cls._NAMESPACE,
            namespace_options={'required': False}
        )
        
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=f'{cls._NAMESPACE}_base',
            namespace_options={'required': False}
        )
        
        spec.output('results', valid_type=orm.Dict, required=False,
                   help='Collected SFE results for all spacings.')
        
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message='The sub-process failed.',
        )
    
    def setup(self):
        """Initialize context for spacing loop."""
        self.ctx.iteration = 1
        self.ctx.additional_spacings = self.inputs.additional_spacings.get_list().copy()
        self.ctx.sfe_data = []
        self.ctx.kpoints_scf_mesh = self.inputs.kpoints_scf_mesh.get_list()
    
    def should_run_sfe(self):
        """Check if there are more spacings to process."""
        if not self.ctx.additional_spacings:
            return False
        
        # Get current spacing
        current_spacing = self.ctx.additional_spacings.pop(0)
        self.ctx.current_spacing = current_spacing
        
        # Generate faulted structure for this spacing
        fault_type = self.inputs.fault_type.value
        fault_method = self.inputs.fault_method.value.lower() if self.inputs.fault_method.value else 'removal'
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        
        if fault_method == 'removal':
            _, faulted_structure_data = get_faulted_structure(
                self.inputs.conventional_structure.get_ase(),
                fault_type=fault_type,
                additional_spacing=current_spacing,
                gliding_plane=gliding_plane,
                n_unit_cells=self.inputs.n_repeats.value,
                fault_mode='removal',
            )
        elif fault_method == 'vacuum':
            vacuum_ratio = float(self.inputs.vacuum_ratio.value)
            _, faulted_structure_data = get_faulted_structure(
                self.inputs.conventional_structure.get_ase(),
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
        """Setup kpoints for the current faulted structure."""
        # Calculate kpoints based on z-ratio between faulted and conventional structures
        faulted_structure_ase = self.ctx.current_structure.get_ase()
        conventional_structure_ase = self.inputs.conventional_structure.get_ase()
        
        z_ratio = faulted_structure_ase.cell.cellpar()[2] / conventional_structure_ase.cell.cellpar()[2]
        kpoints_scf_mesh = self.ctx.kpoints_scf_mesh
        
        kpoints_sfe = orm.KpointsData()
        kpoints_sfe.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / z_ratio)])
        
        self.ctx.kpoints_sfe = kpoints_sfe
        self.report(f'Kpoints mesh for SFE (spacing {self.ctx.current_spacing}): {kpoints_sfe.get_kpoints_mesh()[0]}')
    
    def run_sfe(self):
        """Run the SFE calculation for current spacing."""
        workchain_type = self.inputs.workchain_type.value
        
        if workchain_type == 'PwRelaxWorkChain':
            return self._run_relax_calculation()
        elif workchain_type == 'PwBaseWorkChain':
            return self._run_base_calculation()
        else:
            raise ValueError(f"Unsupported workchain_type: {workchain_type}")
    
    def _run_relax_calculation(self):
        """Run PwRelaxWorkChain calculation."""
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._NAMESPACE
            )
        )
        
        inputs.structure = self.ctx.current_structure
        inputs.base_relax.kpoints = self.ctx.kpoints_sfe
        inputs.metadata.call_link_label = f'{self._NAMESPACE}_{self.ctx.iteration}'
        
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
        
        return {f"workchain_sfe": append_(running)}
    
    def _run_base_calculation(self):
        """Run PwBaseWorkChain calculation."""
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=f'{self._NAMESPACE}_base'
            )
        )
        
        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints_sfe
        inputs.metadata.call_link_label = f'{self._NAMESPACE}_base_{self.ctx.iteration}'
        
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for spacing: {self.ctx.current_spacing}.')
        
        return {f"workchain_sfe": append_(running)}
    
    def inspect_sfe(self):
        """Inspect the SFE calculation results for current spacing."""
        workchain = self.ctx.workchain_sfe[-1]
        self.ctx.iteration += 1
        
        if not workchain.is_finished_ok:
            self.report(
                f"Sub-workchain<{workchain.pk}> for spacing {self.ctx.current_spacing} "
                f"failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        
        self.report(f'Sub-workchain<{workchain.pk}> for spacing {self.ctx.current_spacing} finished successfully.')
        
        # Extract number of bands for next iteration
        if 'output_parameters' in workchain.outputs:
            self.ctx.nbnd = workchain.outputs.output_parameters.get('number_of_bands')
        
        # Extract energy
        total_energy = workchain.outputs.output_parameters.get('energy')
        
        # Store results
        self.ctx.sfe_data.append({
            'spacing': self.ctx.current_spacing,
            'iteration': self.ctx.iteration - 1,
            'energy_ry': float(total_energy) if total_energy else None,
            'multiplier': self.ctx.current_multiplier,
        })
        
        # Expose outputs for this iteration
        workchain_type = self.inputs.workchain_type.value
        if workchain_type == 'PwRelaxWorkChain':
            self.out_many(
                self.exposed_outputs(
                    workchain,
                    PwRelaxWorkChain,
                    namespace=self._NAMESPACE
                )
            )
        else:
            self.out_many(
                self.exposed_outputs(
                    workchain,
                    PwBaseWorkChain,
                    namespace=f'{self._NAMESPACE}_base'
                )
            )
    
    def results(self):
        """Output collected results."""
        if self.ctx.sfe_data:
            self.out('results', orm.Dict(dict={'sfe': self.ctx.sfe_data}))

