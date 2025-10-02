from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida import orm
from ase.formula import Formula

class ISFEWorkChain(SFEBaseWorkChain):
    """ISFE WorkChain"""

    _NAMESPACE = 'isfe'
    _PW_SFE_NAMESPACE = "pw_isfe"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_ISF",
            message='The `PwBaseWorkChain` for the ISF run failed.',
        )

    def setup_supercell_kpoints(self):
        ## todo: I don't know why but when running sfe the PwBaseWorkChain
        ## can't correctly generate the kpoints according to the kpoints_distance.
        ## I explicitly generate the kpoints here.
        current_structure = orm.StructureData(
            ase=self.ctx.structures.intrinsic
            )
        
        intrinsic_formula = Formula(self.ctx.structures.intrinsic.get_chemical_formula())
        _, intrinsic_multiplier = intrinsic_formula.reduce()
        
        if 'kpoints_scf' in self.ctx:
            kpoints_scf = self.ctx.kpoints_scf
        else:
            inputs = {
                'structure': orm.StructureData(
                    ase=self.ctx.structures.unfaulted
                    ),
                'distance': self.inputs.kpoints_distance,
                'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints_scf = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg
        
        kpoints_scf_mesh = kpoints_scf.get_kpoints_mesh()[0]

        kpoints_sfe = orm.KpointsData()
        sfe_z_ratio = self.ctx.structures.intrinsic.cell.cellpar()[2] / self.ctx.structures.unfaulted.cell.cellpar()[2]
        kpoints_sfe.set_kpoints_mesh(kpoints_scf_mesh[:2] + [kpoints_scf_mesh[2] / sfe_z_ratio])
        
        kpoints_surface_energy = orm.KpointsData()
        surface_energy_z_ratio = self.ctx.structures.cleavaged.cell.cellpar()[2] / self.ctx.structures.unfaulted.cell.cellpar()[2]
        kpoints_surface_energy.set_kpoints_mesh(
            kpoints_scf_mesh[:2] + [kpoints_scf_mesh[2] / surface_energy_z_ratio])
        
        self.ctx.kpoints_sfe = kpoints_sfe
        self.ctx.kpoints_surface_energy = kpoints_surface_energy
        self.ctx.current_structure = current_structure
        self.ctx.intrinsic_multiplier = intrinsic_multiplier
    
    def run_sfe(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SFE_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_SFE_NAMESPACE

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
                namespace=self._PW_SFE_NAMESPACE,
            ),
        )

        self.ctx.total_energy_isf_geometry = self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of intrinsic faulted geometry [{self.ctx.intrinsic_multiplier} unit cells]: {self.ctx.total_energy_isf_geometry / self._RY2eV} Ry')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = self.ctx.total_energy_isf_geometry - self.ctx.total_energy_conventional_geometry / self.ctx.unfaulted_multiplier * self.ctx.intrinsic_multiplier
            intrinsic_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'intrinsic stacking fault energy of evaluated from conventional geometry: {intrinsic_stacking_fault_energy} J/m^2')
                    # energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        # isfe = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        # self.ctx.isfe = isfe
        # self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')
