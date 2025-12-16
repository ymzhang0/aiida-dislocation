from .sfebase import SFEBaseWorkChain
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida import orm
from ase.formula import Formula
from aiida_dislocation.tools import (
    get_unstable_faulted_structure,
    calculate_surface_area,
)

class TwinningWorkChain(SFEBaseWorkChain):
    """Twinning WorkChain"""

    _SFE_NAMESPACE = "twinning"

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
        spec.output('results', valid_type=orm.Dict, required=False, help='Collected twinning energy and metadata.')
        
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_TWINNING",
            message='The `PwBaseWorkChain` for the twinning run failed.',
        )

    def setup(self):
        super().setup()
        self.ctx.twinning_done = False
        self.ctx.twinning_data = []

    def _get_fault_type(self):
        """Return the fault type for Twinning workchain.
        Twinning is not a fault type, but we need to implement this for the base class.
        """
        return None  # Twinning is a base structure, not a fault type

    def generate_structures(self):
        """Generate all structures including conventional, cleavaged, and twinning."""
        # First call base to generate conventional and cleavaged
        result = super().generate_structures()
        if result:
            return result
        
        gliding_plane = self.inputs.gliding_plane.value if self.inputs.gliding_plane.value else None
        
        # Get twinning structure using get_unstable_faulted_structure
        strukturbericht, structures_dict = get_unstable_faulted_structure(
            self.ctx.current_structure.get_ase(),
            gliding_plane=gliding_plane,
            n_unit_cells=self.inputs.n_repeats.value,
        )

        # Verify that twinning structure was generated
        if 'twinning' not in structures_dict or structures_dict['twinning'] is None:
            self.report('Twinning structure is not available for this gliding system.')
            return self.exit_codes.ERROR_NO_STRUCTURE_TYPE_DETECTED

        # Store twinning structure directly in context
        self.ctx.twinning_structure = structures_dict['twinning']
        self.ctx.unfaulted_structure = self.ctx.conventional_structure  # unfaulted is the same as conventional
        self.ctx.unfaulted_multiplier = self.ctx.conventional_multiplier

    def setup_supercell_kpoints(self):
        ## todo: I don't know why but when running sfe the PwBaseWorkChain
        ## can't correctly generate the kpoints according to the kpoints_distance.
        ## I explicitly generate the kpoints here.
        if not hasattr(self.ctx, 'twinning_structure'):
            raise ValueError('Twinning structure not found in context.')
        
        current_structure = orm.StructureData(ase=self.ctx.twinning_structure)
        
        twinning_formula = Formula(self.ctx.twinning_structure.get_chemical_formula())
        _, twinning_multiplier = twinning_formula.reduce()
        
        # Get kpoints_scf
        _, kpoints_scf_mesh = self._get_kpoints_scf()

        from math import ceil

        kpoints_sfe = orm.KpointsData()
        sfe_z_ratio = self.ctx.twinning_structure.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        kpoints_sfe.set_kpoints_mesh(kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / sfe_z_ratio)])
        
        kpoints_surface_energy = orm.KpointsData()
        surface_energy_z_ratio = self.ctx.cleavaged_structure.cell.cellpar()[2] / self.ctx.conventional_structure.cell.cellpar()[2]
        kpoints_surface_energy.set_kpoints_mesh(
            kpoints_scf_mesh[:2] + [ceil(kpoints_scf_mesh[2] / surface_energy_z_ratio)])
        
        self.ctx.kpoints_sfe = kpoints_sfe
        self.ctx.kpoints_surface_energy = kpoints_surface_energy
        self.ctx.current_structure = current_structure
        self.ctx.twinning_multiplier = twinning_multiplier

    def should_run_sfe(self):
        if not self._SFE_NAMESPACE in self.inputs:
            return False
        if getattr(self.ctx, 'twinning_done', False):
            return False
        return True

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
        self.report(f'launching PwBaseWorkChain<{running.pk}> for twinning stacking fault.')

        self.to_context(workchain_sfe = running)

    def inspect_sfe(self):
        workchain = self.ctx.workchain_sfe

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{workchain.pk}> for twinning faulted geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_TWINNING

        self.report(
            f'PwBaseWorkChain<{workchain.pk}> for twinning faulted geometry finished OK'
            )
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._SFE_NAMESPACE,
            ),
        )

        total_energy_twinning_geometry = workchain.outputs.output_parameters.get('energy')
        self.ctx.total_energy_twinning_geometry = total_energy_twinning_geometry
        self.ctx.total_energy_faulted_geometry = total_energy_twinning_geometry
        self.report(f'Total energy of twinning faulted geometry [{self.ctx.twinning_multiplier} unit cells]: {total_energy_twinning_geometry / self._RY2eV} Ry')
        if 'total_energy_conventional_geometry' in self.ctx:
            energy_difference = (
                total_energy_twinning_geometry
                - self.ctx.total_energy_conventional_geometry / self.ctx.conventional_multiplier * self.ctx.twinning_multiplier
            )
            twinning_stacking_fault_energy = energy_difference / self.ctx.surface_area * self._eVA22Jm2
            self.report(f'twinning stacking fault energy of evaluated from conventional geometry: {twinning_stacking_fault_energy} J/m^2')
        else:
            twinning_stacking_fault_energy = None

        self.ctx.twinning_data.append({
            'energy_ry': float(total_energy_twinning_geometry),
            'twinning_multiplier': self.ctx.twinning_multiplier,
            'twinning_j_m2': float(twinning_stacking_fault_energy) if twinning_stacking_fault_energy is not None else None,
        })
        self.ctx.twinning_done = True

    def results(self):
        if getattr(self.ctx, 'twinning_data', None):
            self.out('results', orm.Dict(dict={'twinning': self.ctx.twinning_data}))
