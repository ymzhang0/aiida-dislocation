from .sfebase import SFEBaseWorkChain
from .layer_relax import RigidLayerRelaxWorkChain
from aiida import orm

class ISFEWorkChain(SFEBaseWorkChain):
    """ISFE WorkChain"""

    _SFE_NAMESPACE = "isfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_ISF",
            message='The `PwBaseWorkChain` for the ISF run failed.',
        )

    def _get_fault_type(self):
        """Return the fault type for ISFE workchain."""
        return 'intrinsic'

    def results(self):
        """Expose collected ISFE data to the caller."""
        pass
