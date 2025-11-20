from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

class ESFEWorkChain(SFEBaseWorkChain):
    """ESFE WorkChain"""
    
    _SFE_NAMESPACE = "esfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._SFE_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_ESF",
            message='The `PwBaseWorkChain` for the ESF run failed.',
        )

    def _get_fault_type(self):
        """Return the fault type for ESFE workchain."""
        return 'extrinsic'

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
        self.report(f'launching PwBaseWorkChain<{running.pk}> for extrinsic stacking fault.')

        self.to_context(workchain_sfe = running)

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{workchain.pk}> for extrinsic faulted geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ESF

        self.report(
            f'PwBaseWorkChain<{workchain.pk}> for extrinsic faulted geometry finished OK'
            )
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE,
            ),
        )

        self.ctx.total_energy_esf_geometry = self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of extrinsic faulted geometry [{self.ctx.extrinsic_multiplier} unit cells]: {self.ctx.total_energy_esf_geometry / self._RY2eV} Ry')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = self.ctx.total_energy_esf_geometry - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.extrinsic_multiplier
            extrinsic_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'extrinsic stacking fault energy of evaluated from conventional geometry: {extrinsic_stacking_fault_energy} J/m^2')
        # energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        # isfe = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        # self.ctx.isfe = isfe
        # self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')
