from tkinter import W
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida.engine import CalcJob, BaseRestartWorkChain, ExitCode, process_handler, while_, if_
from aiida_quantumespresso.calculations.pw import PwCalculation

import logging

from ..tools import get_unstable_faulted_structure_and_kpoints


class GSFEWorkChain(ProtocolMixin, WorkChain):
    """GSFE WorkChain"""

    _PW_SCF_NAMESPACE = "pw_scf"
    _PW_USF_NAMESPACE = "pw_usf"
    _PW_SURFACE_ENERGY_NAMESPACE = "pw_surface_energy"
    _PW_CURVE_NAMESPACE = "pw_curve"

    _workflow_name = 'gsfe'
    _workflow_description = 'GSFE WorkChain'

    # Strukturtyp, gliding plane, slipping direction

    _IMPLEMENTED_SLIPPING_SYSTEMS = (
        ('A1', '100', None),
        ('B1', '100', None),
        ('B1', '110', None),
        ('B1', '111', None),
        ('C2', '100', None),
        ('A15', '100', None),
        ('A15', '110', None),
        ('C1b', '100', None),
        ('E21', '100', None),
        ('E21', '110', None),
    )

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('n_layers', valid_type=orm.Int, required=True, default=lambda: orm.Int(4),
                   help='The number of layers in the supercell')
        spec.input('nsteps', valid_type=orm.Int, required=False, default=lambda: orm.Int(10),
                   help='The number of stacking faults ')
        spec.input('slipping_system', valid_type=orm.List, required=True,
                   help="""
                   Currently only supported:
                   A1, 100, 100
                   B1, 100, 100
                   B1, 110, 110
                   """)

        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints', valid_type=orm.KpointsData, required=True,)

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SCF_NAMESPACE,
            exclude=('structure', 'kpoints')
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_USF_NAMESPACE,
            exclude=('structure', 'kpoints')
            )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SURFACE_ENERGY_NAMESPACE,
            exclude=('structure', 'kpoints')
            )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_CURVE_NAMESPACE,
            exclude=('structure', 'kpoints')
            )

        spec.outline(
            cls.validate_slipping_system,
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_usf)(
                cls.run_usf,
                cls.inspect_usf,
            ),
            if_(cls.should_run_curve)(
                cls.run_curve,
                cls.inspect_curve,
            ),
            if_(cls.should_run_surface_energy)(
                cls.run_surface_energy,
                cls.inspect_surface_energy,
            ),
            cls.results,
        )

    def validate_slipping_system(self):
        structure_type, gliding_plane, slipping_direction = slipping_system = self.inputs.slipping_system
        if (structure_type, gliding_plane, slipping_direction) not in self._IMPLEMENTED_SLIPPING_SYSTEMS:
            raise NotImplementedError(f'Slipping system {self.inputs.slipping_system} not implemented')

        structure_type, gliding_plane, slipping_direction = slipping_system = self.inputs.slipping_system

        if structure_type == 'A1':
            if gliding_plane == '100':
                self.report(f'AB stacking for 100 gliding plane. Faulting possible.')
            if gliding_plane == '110':
                self.report(f'AB stacking for 100 gliding plane. Faulting possible.')
            if gliding_plane == '111':
                self.report(f'ABC stacking for 111 gliding plane.Faulting possible.')

        if structure_type == 'A2':
            if gliding_plane == '100':
                raise ValueError(f'Only 100 gliding plane for A2 structure')
            if gliding_plane == '110':
                self.report(f'Not implemented yet. But faulting possible.')
            if gliding_plane == '111':
                self.report(f'ABC stacking for 111 gliding plane. Faulting possible.')

        if structure_type == 'B1':
            self.report(f'We are doing NaCl-type structure. space group (225).')
            if gliding_plane == '100':
                raise ValueError(f'Only 1 type of stacking for 100 gliding plane. No faulting possible.')
            if gliding_plane == '110':
                self.report(f'AB stacking for 110 gliding plane. Faulting possible.')
            if gliding_plane == '111':
                raise NotImplementedError(f'Not implemented yet. But faulting possible.')

        if structure_type == 'B2':
            self.report(f'We are doing CsCl-type structure. space group (229).')
            if gliding_plane == '100':
                raise ValueError(f'Only 1 type of stacking for 100 gliding plane. No faulting possible.')
            if gliding_plane == '110':
                self.report(f'AB stacking for 110 gliding plane. Faulting possible.')
            if gliding_plane == '111':
                raise NotImplementedError(f'Not implemented yet. But faulting possible.')

        if structure_type == 'C1':
            self.report(f'We are doing pyrite-type structure.')
            if gliding_plane == '100':
                raise NotImplementedError(f'Not implemented yet.')
            if gliding_plane == '110':
                raise NotImplementedError(f'Not implemented yet.')
            if gliding_plane == '111':
                raise NotImplementedError(f'Not implemented yet.')

        if structure_type == 'C1b':
            self.report(f'We are doing half-heusler-type structure, space group (216).')
            if gliding_plane == '100':
                self.report(f'ABCB stacking for 100 gliding plane. Faulting possible.')
            if gliding_plane == '110':
                self.report(f'AB stacking for 110 gliding plane. Faulting possible.')
            if gliding_plane == '111':
                raise NotImplementedError(f'Not implemented yet. But faulting possible.')

        if structure_type == 'E21':
            self.report(f'We are doing perovskite-type structure, space group (221).')
            if gliding_plane == '100':
                raise ValueError(f'AB stacking for for 100 gliding plane. Faulting possible.')
            if gliding_plane == '110':
                self.report(f'ABCB stacking for 110 gliding plane. Faulting possible.')
            if gliding_plane == '111':
                raise NotImplementedError(f'Not implemented yet. But faulting possible.')

    def setup(self):
        self.ctx.current_structure = self.inputs.structure
        self.ctx.kpoints = self.inputs.kpoints
        self.ctx.sf_count = 0
        self.ctx.sf_energy = []
        self.ctx.sf_stress = []
        self.ctx.sf_strain = []
        self.ctx.sf_displacement = []
        self.ctx.sf_displacement = []

    def should_run_relax(self):
        return self._PW_RELAX_NAMESPACE in self.inputs

    def run_relax(self):
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='pw'))
        inputs.metadata.call_link_label = f'relax_uc'
        inputs.structure = self.inputs.structure
        inputs.kpoints = self.inputs.kpoints
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for unitcell {self.inputs.structure.get_formula()}')
        self.to_context(workchain_relax_uc = running)

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax_uc
        if not workchain.is_finished_ok:
            self.report(f'Relax calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'Relax calculation<{self.ctx.workchain_relax_uc.pk}> finished')

        self.ctx.current_structure = workchain.outputs.structure

    def should_run_scf(self):
        return self._PW_SCF_NAMESPACE in self.inputs

    def run_scf(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SCF_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._PW_SCF_NAMESPACE

        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for unitcell {self.inputs.structure.get_formula()}')

        self.to_context(workchain_scf_uc = running)

    def inspect_scf(self):
        workchain = self.ctx.workchain_scf_uc
        if not workchain.is_finished_ok:
            self.report(f'SCF calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f'SCF calculation<{self.ctx.workchain_scf_uc.pk}> finished')

    def should_run_usf(self):

        return self.inputs.only_USF

    def run_usf(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_USF_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_USF_NAMESPACE
        structure_uc = self.ctx.current_structure
        kpoints_uc = self.ctx.kpoints
        structure_sc, kpoints_sc = get_unstable_faulted_structure_and_kpoints(
            structure_uc, kpoints_uc, self.inputs.n_layers, self.inputs.slipping_system
            )
        inputs.structure = structure_sc
        inputs.kpoints = kpoints_sc

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for faulted supercell.')

        self.to_context(workchain_scf_faulted_supercell = running)

    def inspect_usf(self):
        workchain = self.ctx.workchain_scf_faulted_supercell
        if not workchain.is_finished_ok:
            self.report(f'USF calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(f'USF calculation<{self.ctx.workchain_scf_faulted_supercell.pk}> finished')
        self.ctx.current_structure = workchain.outputs.structure
        self.ctx.kpoints = workchain.outputs.kpoints

    def should_run_curve(self):
        return self._PW_CURVE_NAMESPACE in self.inputs

    def run_curve(self):
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace=self._PW_CURVE_NAMESPACE))
        inputs.metadata.call_link_label = self._PW_CURVE_NAMESPACE
        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for curve calculation.')
        self.to_context(workchain_curve = running)

    def inspect_curve(self):
        workchain = self.ctx.workchain_curve
        if not workchain.is_finished_ok:
            self.report(f'Curve calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CURVE

        self.report(f'Curve calculation<{self.ctx.workchain_curve.pk}> finished')
        self.ctx.current_structure = workchain.outputs.structure
        self.ctx.kpoints = workchain.outputs.kpoints

    def should_run_surface_energy(self):
        return self._PW_SURFACE_ENERGY_NAMESPACE in self.inputs

    def run_surface_energy(self):
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace=self._PW_SURFACE_ENERGY_NAMESPACE))
        inputs.metadata.call_link_label = self._PW_SURFACE_ENERGY_NAMESPACE
        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')
        self.to_context(workchain_surface_energy = running)

    def inspect_surface_energy(self):
        workchain = self.ctx.workchain_surface_energy
        if not workchain.is_finished_ok:
            self.report(f'Surface energy calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

        self.report(f'Surface energy calculation<{self.ctx.workchain_surface_energy.pk}> finished')
        self.ctx.current_structure = workchain.outputs.structure
        self.ctx.kpoints = workchain.outputs.kpoints

    def results(self):
        self.report(f'Results')
