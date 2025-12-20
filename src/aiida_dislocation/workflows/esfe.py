from .sfebase import SFEBaseWorkChain
from .layer_relax import RigidLayerRelaxWorkChain
from aiida import orm

class ESFEWorkChain(SFEBaseWorkChain):
    """ESFE WorkChain"""
    
    _SFE_NAMESPACE = "esfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_ESF",
            message='The `PwBaseWorkChain` for the ESF run failed.',
        )


    def _get_fault_type(self):
        """Return the fault type for ESFE workchain."""
        return 'extrinsic'

    def results(self):
        """Expose collected ESFE data to the caller."""
        pass
