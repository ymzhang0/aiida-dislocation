from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from ..tools import get_unstable_faulted_structure, is_primitive_cell
from aiida import orm

class ISFEWorkChain(SFEBaseWorkChain):
    """ISFE WorkChain"""

    _NAMESPACE = 'isfe'
    _PW_SFE_NAMESPACE = "pw_isfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
    def generate_faulted_structure(self):
        structures = get_unstable_faulted_structure(
            self.ctx.current_structure.get_ase(),
            )
        self.ctx.structure_scf = orm.StructureData(ase=structures['unfaulted'])
        self.ctx.structure_sf = orm.StructureData(ase=structures['intrinsic'])


    def run_sfe(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SFE_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_SFE_NAMESPACE


        inputs.pw.structure = self.ctx.structure_sf
        # inputs.kpoints = self.ctx.kpoint_sf


        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for intrinsic stacking fault.')

        self.to_context(workchain_scf_faulted_geometry = running)

    def inspect_sfe(self):
        super().inspect_sfe()