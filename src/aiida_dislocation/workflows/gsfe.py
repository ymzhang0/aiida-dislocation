from aiida import orm
from aiida.common import AttributeDict, datastructures, exceptions
from aiida.common.lang import classproperty
from aiida.plugins import factories

from aiida.engine import CalcJob, BaseRestartWorkChain, ExitCode, process_handler, while_, if_
from aiida_quantumespresso.calculations.pw import PwCalculation

import logging

from ..tools.structures import get_supercell_structure_and_kpoints


class GSFEWorkChain(BaseRestartWorkChain):
    """GSFE WorkChain"""

    _process_class = PwCalculation

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

        # spec.input('gliding_plane', val   id_type=orm.List, required=True, 
        #            help='The gliding plane of the dislocation')
        spec.input('skip_uc', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                   help='Whether to skip the unitcell calculation')
        spec.input('n_layers', valid_type=orm.Int, required=True, default=lambda: orm.Int(4),
                   help='The number of layers in the supercell')
        # spec.input('slipping_direction', valid_type=orm.List, required=True, 
        #            help='The slipping direction of the dislocation')
        spec.input('only_USF', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(True),
                   help='Whether to do a series of stacking faults')
        spec.input('do_surface', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                   help='Whether to do a surface energy calculation')
        spec.input('do_curve', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                   help='Whether to do a curve calculation')
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

        spec.expose_inputs(PwCalculation, namespace='pw', exclude=('structure', 'kpoints'))

        spec.outline(
            cls.validate_slipping_system,
            cls.setup,
            if_(cls.should_run_uc)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_USF)(
                cls.run_USF,
                cls.inspect_USF,
            ),
            if_(cls.should_run_curve)(
            while_(cls.should_continue_curve)(
                cls.run_scf,
                cls.inspect_scf,
                ),
            ),
            if_(cls.should_run_surface)(
                cls.run_surface,
                cls.inspect_surface,
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

        self.ctx.sf_count = 0
        self.ctx.sf_energy = []
        self.ctx.sf_stress = []
        self.ctx.sf_strain = []
        self.ctx.sf_displacement = []
        self.ctx.sf_displacement = []

    def should_run_uc(self):
        if self.inputs.skip_uc:
            self.logger.warning(
                "You are skipping the unitcell calculation."
                "Hopefully you know what you are doing!"
            )   

        return not self.inputs.skip_uc
    def run_scf(self):
        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace='pw'))

        inputs.metadata.call_link_label = f'scf_uc'

        inputs.structure = self.inputs.structure
        inputs.kpoints = self.inputs.kpoints

        running = self.submit(PwCalculation, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for unitcell {self.inputs.structure.get_formula()}')

        self.to_context(workchain_scf_uc = running)

    def inspect_scf(self):
        self.report(f'SCF calculation<{self.ctx.workchain_scf_uc.pk}> finished')

    def should_run_USF(self):

        return self.inputs.only_USF

    def run_USF(self):

        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace='pw'))
        inputs.metadata.call_link_label = f'scf_faulted_supercell'
        structure_uc = self.inputs.structure
        kpoints_uc = self.inputs.kpoints
        structure_sc, kpoints_sc = get_supercell_structure_and_kpoints(
            structure_uc, kpoints_uc, self.inputs.n_layers, self.inputs.slipping_system
            )
        inputs.structure = structure_sc
        inputs.kpoints = kpoints_sc

        running = self.submit(PwCalculation, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for faulted supercell.')

        self.to_context(workchain_scf_faulted_supercell = running)

    def inspect_USF(self):
        self.report(f'USF calculation<{self.ctx.workchain_scf_faulted_supercell.pk}> finished')


    def should_run_curve(self):
        return self.inputs.do_curve

    def run_curve(self):
        pass

    def should_continue_curve(self):
        return True
    
    def should_run_surface(self):
        return self.inputs.do_surface

    def run_surface(self):
        pass

    def inspect_surface(self):
        self.report(f'Surface calculation<{self.ctx.workchain_surface.pk}> finished')

    def results(self):
        self.report(f'Results')
