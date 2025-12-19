from .sfebase import SFEBaseWorkChain
from .layer_relax import RigidLayerRelaxWorkChain
from aiida import orm

class ISFEWorkChain(SFEBaseWorkChain):
    """ISFE WorkChain"""

    _SFE_NAMESPACE = "isfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output('results', valid_type=orm.Dict, required=False, help='Collected ISFE energies for each evaluated spacing.')
        
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_ISF",
            message='The `PwBaseWorkChain` for the ISF run failed.',
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
        """Return the fault type for ISFE workchain."""
        return 'intrinsic'

    def inspect_layer_relax(self):
        """Inspect the RigidLayerRelaxWorkChain results and calculate ISFE values."""
        workchain = self.ctx.workchain_layer_relax
        
        if not workchain.is_finished_ok:
            self.report(
                f"RigidLayerRelaxWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ISF
        
        self.report(f'RigidLayerRelaxWorkChain<{workchain.pk}> finished successfully.')
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                RigidLayerRelaxWorkChain,
                namespace=self._RIGID_LAYER_RELAX_NAMESPACE
            )
        )
        
        # Extract results from RigidLayerRelaxWorkChain and calculate ISFE
        if 'results' in workchain.outputs:
            relax_results = workchain.outputs.results.get_dict().get('rigid_layer_relax', [])
            self.ctx.isfe_data = []
            
            for result in relax_results:
                spacing = result['spacing']
                energy_ry = result['energy_ry']
                multiplier = result['multiplier']
                
                if energy_ry is None:
                    continue
                
                # Calculate stacking fault energy
                intrinsic_stacking_fault_energy = self._calculate_stacking_fault_energy(
                    energy_ry,
                    multiplier,
                    'intrinsic'
                )
                
                self.ctx.isfe_data.append({
                    'spacing': spacing,
                    'energy_ry': energy_ry,
                    'intrinsic_multiplier': multiplier,
                    'isfe_j_m2': (
                        float(intrinsic_stacking_fault_energy) 
                        if intrinsic_stacking_fault_energy is not None 
                        else None
                    ),
                })

    def results(self):
        """Expose collected ISFE data to the caller."""
        if getattr(self.ctx, 'isfe_data', None):
            self.out('results', orm.Dict(dict={'isfe': self.ctx.isfe_data}))
