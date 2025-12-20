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


    def results(self):
        """Expose collected USFE data to the caller."""
        pass