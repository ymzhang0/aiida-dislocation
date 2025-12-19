from .sfebase import SFEBaseWorkChain
from .layer_relax import RigidLayerRelaxWorkChain
from aiida import orm

class USFEWorkChain(SFEBaseWorkChain):
    """USFE WorkChain"""

    _SFE_NAMESPACE = "usfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
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
        return builder

    def _get_fault_type(self):
        """Return the fault type for USFE workchain."""
        return 'unstable'

    def inspect_layer_relax(self):
        """Inspect the RigidLayerRelaxWorkChain results and calculate USFE values."""
        workchain = self.ctx.workchain_layer_relax
        
        if not workchain.is_finished_ok:
            self.report(
                f"RigidLayerRelaxWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF
        
        self.report(f'RigidLayerRelaxWorkChain<{workchain.pk}> finished successfully.')
        
        
        # Extract results from RigidLayerRelaxWorkChain and calculate USFE
        if 'results' in workchain.outputs:
            relax_results = workchain.outputs.results.get_dict().get('rigid_layer_relax', [])
            self.ctx.usfe_data = []
            
            for result in relax_results:
                spacing = result['spacing']
                energy_ry = result['energy_ry']
                multiplier = result['multiplier']
                
                if energy_ry is None:
                    continue
                
                # Calculate stacking fault energy
                unstable_stacking_fault_energy = self._calculate_stacking_fault_energy(
                    energy_ry,
                    multiplier,
                    'unstable'
                )
                
                self.ctx.usfe_data.append({
                    'spacing': spacing,
                    'energy_ry': energy_ry,
                    'unstable_multiplier': multiplier,
                    'usfe_j_m2': float(unstable_stacking_fault_energy) if unstable_stacking_fault_energy is not None else None,
                })

    def results(self):
        """Expose collected USFE data to the caller."""
        if getattr(self.ctx, 'usfe_data', None):
            self.out('results', orm.Dict(dict={'usfe': self.ctx.usfe_data}))