from .sfebase import SFEBaseWorkChain
from .layer_relax import RigidLayerRelaxWorkChain
from aiida import orm

class ESFEWorkChain(SFEBaseWorkChain):
    """ESFE WorkChain"""
    
    _SFE_NAMESPACE = "esfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)

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

    def _get_fault_type(self):
        """Return the fault type for ESFE workchain."""
        return 'extrinsic'

    def inspect_layer_relax(self):
        """Inspect the RigidLayerRelaxWorkChain results and calculate ESFE values."""
        workchain = self.ctx.workchain_layer_relax
        
        if not workchain.is_finished_ok:
            self.report(
                f"RigidLayerRelaxWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ESF
        
        self.report(f'RigidLayerRelaxWorkChain<{workchain.pk}> finished successfully.')
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                RigidLayerRelaxWorkChain,
                namespace=self._RIGID_LAYER_RELAX_NAMESPACE
            )
        )
        
        # Extract results from RigidLayerRelaxWorkChain and calculate ESFE
        if 'results' in workchain.outputs:
            relax_results = workchain.outputs.results.get_dict().get('rigid_layer_relax', [])
            self.ctx.esfe_data = []
            
            for result in relax_results:
                spacing = result['spacing']
                energy_ry = result['energy_ry']
                multiplier = result['multiplier']
                
                if energy_ry is None:
                    continue
                
                # Calculate stacking fault energy
                extrinsic_stacking_fault_energy = self._calculate_stacking_fault_energy(
                    energy_ry,
                    multiplier,
                    'extrinsic'
                )
                
                self.ctx.esfe_data.append({
                    'spacing': spacing,
                    'energy_ry': energy_ry,
                    'extrinsic_multiplier': multiplier,
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
