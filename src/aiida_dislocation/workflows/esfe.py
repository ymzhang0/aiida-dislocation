from .sfebase import SFEBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida import orm
from aiida.engine import append_
from aiida_dislocation.tools import get_faulted_structure

class ESFEWorkChain(SFEBaseWorkChain):
    """ESFE WorkChain"""
    
    _SFE_NAMESPACE = "esfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.input('additional_spacings', valid_type=orm.List, required=False, default=lambda: orm.List(list=[0.0]),
                    help='The additional spacing to add to the structure.')
        spec.input('fault_method', valid_type=orm.Str, required=False, default=lambda: orm.Str('removal'),
                    help="How to generate faulted structures: 'removal', or 'vacuum'.")
        spec.input('vacuum_ratio', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.1),
                    help='Vacuum ratio added along the fault normal when using vacuum gliding.')
        spec.output('results', valid_type=orm.Dict, required=False, help='Collected ESFE energies for each evaluated spacing.')
        
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_ESF",
            message='The `PwBaseWorkChain` for the ESF run failed.',
        )

    @classmethod
    def get_builder_from_protocol(
            cls,
            code,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = super().get_builder_from_protocol(
            code, structure, protocol, overrides, **kwargs)
        builder.additional_spacings = orm.List(list=inputs.get('additional_spacings', [0.0]))
        builder.fault_method = orm.Str(inputs.get('fault_method', 'removal'))
        builder.vacuum_ratio = orm.Float(inputs.get('vacuum_ratio', 0.1))
        return builder

    def setup(self):
        super().setup()
        self.ctx.iteration = 1
        self.ctx.additional_spacings = self.inputs.additional_spacings.get_list()
        self.ctx.esfe_data = []

    def _get_fault_type(self):
        """Return the fault type for ESFE workchain."""
        return 'extrinsic'

    def setup_supercell_kpoints(self):
        """
        Setup kpoints for ESFE.
        Note: current_structure and multiplier are set in should_run_sfe for each iteration.
        """
        if not hasattr(self.ctx, 'current_structure'):
            raise ValueError('Current structure not found in context.')

        # Get kpoints_scf
        _, kpoints_scf_mesh = self._get_kpoints_scf()

        # Calculate kpoints for SFE using helper method
        current_structure_ase = self.ctx.current_structure.get_ase()
        self.ctx.kpoints_sfe = self._calculate_kpoints_for_structure(
            current_structure_ase,
            kpoints_scf_mesh
        )

        # Calculate kpoints for surface energy using helper method
        self.ctx.kpoints_surface_energy = self._setup_surface_energy_kpoints(kpoints_scf_mesh)

    def should_run_sfe(self):

        if not self._SFE_NAMESPACE in self.inputs:
            return False

        fault_method = self.inputs.fault_method.value.lower() if self.inputs.fault_method.value else 'removal'

        fault_type = self._get_fault_type()
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None

        if self.ctx.additional_spacings == []:
            return False

        current_spacing = self.ctx.additional_spacings.pop(0)
        self.ctx.current_spacing = current_spacing

        if fault_method == 'removal':
            # Get faulted structure (based on conventional cell)
            _, faulted_structure_data = get_faulted_structure(
                self.ctx.conventional_structure,
                fault_type=fault_type,
                additional_spacing=current_spacing,
                gliding_plane=gliding_plane,
                n_unit_cells=self.inputs.n_repeats.value,
                fault_mode='removal',
            )
        elif fault_method == 'vacuum':
            vacuum_ratio = float(self.inputs.vacuum_ratio.value)
            _, faulted_structure_data = get_faulted_structure(
                self.ctx.conventional_structure,
                fault_type=fault_type,
                additional_spacing=current_spacing,
                gliding_plane=gliding_plane,
                n_unit_cells=self.inputs.n_repeats.value,
                fault_mode='vacuum',
                vacuum_ratio=vacuum_ratio,
            )
        else:
            raise ValueError(f"Unsupported fault method: {fault_method}")
        
        # Validate using helper method
        is_valid, _ = self._validate_faulted_structure(faulted_structure_data, fault_type)
        if not is_valid:
            return False

        # Extract structure
        actual_structure = faulted_structure_data['structures'][0].get('structure')

        # Store the ASE atoms object for later use
        self.ctx.current_structure_ase = actual_structure
        # Create StructureData from the ASE atoms object
        self.ctx.current_structure = orm.StructureData(ase=actual_structure)

        # Calculate multiplier using helper method
        self.ctx.extrinsic_multiplier = self._calculate_structure_multiplier(actual_structure)

        return True

    def run_sfe(self):

        inputs = super().run_sfe()

        inputs.structure = self.ctx.current_structure
        inputs.base_relax.kpoints = self.ctx.kpoints_sfe
        inputs.metadata.call_link_label = f'esfe_{self.ctx.iteration}'
        fault_method = self.inputs.fault_method.value.lower() if self.inputs.fault_method.value else 'removal'
        parameters = inputs.base_relax.pw.parameters.get_dict()

        if fault_method == 'vacuum':
            parameters['CELL']['cell_dofree'] = 'fixc'
        
        if 'nbnd' in self.ctx:
            parameters['SYSTEM']['nbnd'] = int(self.ctx.nbnd)
        
        inputs.base_relax.pw.parameters = orm.Dict(parameters)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for additional spacing: {self.ctx.current_spacing}.')

        return {f"workchain_sfe": append_(running)}

    def inspect_sfe(self):
        """Inspect the SFE calculation for extrinsic stacking fault."""
        workchain = self.ctx.workchain_sfe[-1]
        self.ctx.iteration += 1
        
        # Use helper method for inspection
        exit_code = self._inspect_workchain(
            workchain,
            'PwRelaxWorkChain',
            'extrinsic faulted geometry',
            self.exit_codes.ERROR_SUB_PROCESS_FAILED_ESF,
            namespace=self._SFE_NAMESPACE,
            workchain_class=PwRelaxWorkChain
        )
        if exit_code:
            return exit_code

        # Extract number of bands for next iteration
        self.ctx.nbnd = workchain.outputs.output_parameters.get('number_of_bands')
        
        # Extract energy
        total_energy_esf_geometry = self._get_workchain_energy(workchain)
        
        # Report energy
        self._report_energy(
            total_energy_esf_geometry,
            self.ctx.extrinsic_multiplier,
            'extrinsic faulted geometry',
            'unit cells'
        )
        
        # Calculate stacking fault energy using helper method
        extrinsic_stacking_fault_energy = self._calculate_stacking_fault_energy(
            total_energy_esf_geometry,
            self.ctx.extrinsic_multiplier,
            'extrinsic'
        )

        # Collect per-spacing results for the final output node
        if not hasattr(self.ctx, 'esfe_data'):
            self.ctx.esfe_data = []
        
        self.ctx.esfe_data.append({
            'spacing': self.ctx.current_spacing,
            'iteration': self.ctx.iteration - 1,
            'energy_ry': float(total_energy_esf_geometry),
            'extrinsic_multiplier': self.ctx.extrinsic_multiplier,
            'esfe_j_m2': (
                float(extrinsic_stacking_fault_energy) 
                if extrinsic_stacking_fault_energy is not None 
                else None
            ),
        })

    def results(self):
        """Expose collected ESFE data to the caller."""
        if getattr(self.ctx, 'esfe_data', None):
            self.out('results', orm.Dict(dict={'esfe': self.ctx.esfe_data}))
