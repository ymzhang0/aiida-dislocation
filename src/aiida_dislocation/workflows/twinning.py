from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from ..tools import get_unstable_faulted_structure, is_primitive_cell

class TwinningWorkChain(SFEBaseWorkChain):
    """Twinning WorkChain"""

    _NAMESPACE = 'twinning'
    _PW_SFE_NAMESPACE = "pw_twinning"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        

    def generate_kpoints_for_faulted_structure(self):

        structures = get_unstable_faulted_structure(self.ctx.current_structure.get_ase(), 'A1', '111')

        try:
            kpoints = self.inputs.kpoints
        except AttributeError:
            inputs = {
                'structure': structures['twinning'],
                'distance': self.inputs.kpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.kpoints = kpoints
        self.ctx.current_structure = structures['twinning']
    
    def run_sfe(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SFE_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_SFE_NAMESPACE


        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints


        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for twinning stacking fault.')

        self.to_context(workchain_scf_faulted_geometry = running)

    def inspect_sfe(self):
        super().inspect_sfe()