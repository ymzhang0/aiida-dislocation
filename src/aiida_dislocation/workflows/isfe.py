from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

class ISFEWorkChain(SFEBaseWorkChain):
    """ISFE WorkChain"""

    _SFE_NAMESPACE = "isfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        # spec.expose_outputs(
        #     PwBaseWorkChain,
        #     namespace=cls._SFE_NAMESPACE,
        #     namespace_options={
        #         'required': False,
        #     }
        # )
        
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_ISF",
            message='The `PwBaseWorkChain` for the ISF run failed.',
        )

    def _get_fault_type(self):
        """Return the fault type for ISFE workchain."""
        return 'intrinsic'
    
    def run_sfe(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._SFE_NAMESPACE

        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints_sfe

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for intrinsic stacking fault.')

        self.to_context(workchain_sfe = running)

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{workchain.pk}> for intrinsic faulted geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ISF

        self.report(
            f'PwBaseWorkChain<{workchain.pk}> for intrinsic faulted geometry finished OK'
            )
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE,
            ),
        )

        self.ctx.total_energy_isf_geometry = self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of intrinsic faulted geometry [{self.ctx.intrinsic_multiplier} unit cells]: {self.ctx.total_energy_isf_geometry / self._RY2eV} Ry')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = self.ctx.total_energy_isf_geometry - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.intrinsic_multiplier
            intrinsic_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'intrinsic stacking fault energy of evaluated from conventional geometry: {intrinsic_stacking_fault_energy} J/m^2')
                    # energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        # isfe = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        # self.ctx.isfe = isfe
        # self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')
