from aiida import orm
from aiida.common import AttributeDict, datastructures, exceptions
from aiida.common.lang import classproperty
from aiida.plugins import factories

from aiida.engine import CalcJob, BaseRestartWorkChain, ExitCode, process_handler, while_, if_
from aiida_quantumespresso.calculations.pw import PwCalculation

import logging

from ..tools.structures import get_unstable_faulted_structure_and_kpoints, is_primitive_cell


class USFWorkChain(BaseRestartWorkChain):
    """GSFE WorkChain"""

    _process_class = PwCalculation

    _workflow_name = 'gsfe'
    _workflow_description = 'GSFE WorkChain'
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20 
    _eVA22Jm2 = 1.602176634E-19 * 1E+20 
    # Strukturtyp, gliding plane, slipping direction

    _IMPLEMENTED_SLIPPING_SYSTEMS = {
        'A1': {
            'info': 'FCC element crystal <space group #225, prototype Cu>. '
                    'Usually, the gliding plane is 111.',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[010]',
                        'faulting_possible': True,
                        },
                '110': {'stacking': 'AB', 
                        'slipping_direction': '1/2[112]',
                        'faulting_possible': True,
                        },
                '111': {'stacking': 'ABC', 
                        'slipping_direction': '1/2[110]',
                        'faulting_possible': True,
                        },
            }
        },
        'A2': {
            'info': 'FCC element crystal <space group #227, prototype V>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[110]',
                        'faulting_possible': True,
                        },
                '110': {'stacking': 'AB', 
                        'slipping_direction': '1/2[001]',
                        'faulting_possible': True,
                        },
                '111': {'stacking': 'ABC', 
                        'slipping_direction': '1/2[011]',
                        'faulting_possible': True,
                        },
            }
        },
        'A15': {
            'info': 'A3B crystal <space group #223, prototype Nb3Sn>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[110]',
                        'faulting_possible': True,
                        },
                '110': {'stacking': 'AB', 
                        'slipping_direction': '1/2[001]',
                        'faulting_possible': True,
                        },
                '111': {'stacking': 'ABC', 
                        'slipping_direction': '1/2[011]',
                        'faulting_possible': True,
                        },
            }
        },
        'B1': {
            'info': 'FCC element crystal <space group #225, prototype NaCl>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[010]',
                        'faulting_possible': True,
                        },
                '110': {'stacking': 'AB', 
                        'slipping_direction': '1/2[112]',
                        'faulting_possible': True,
                        },
            }
        },
        'B2': {
            'info': 'FCC element crystal <space group #229, prototype CsCl>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[010]',
                        'faulting_possible': True,
                        },
            }
        },
        'C1': {
            'info': 'We are doing pyrite-type structure. <space group #205, prototype FeS2>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'ABCD', 
                        'slipping_direction': '1/2[100]',
                        'faulting_possible': True,
                        },
            }
        },       
        'C1b': {
            'info': 'We are doing half-heusler-type structure. <space group #216, prototype MgSiAl>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'ABCD', 
                        'slipping_direction': '1/2[100]',
                        'faulting_possible': True,
                        },
                '110': {'stacking': 'AB', 
                        'slipping_direction': '1/2[110]',
                        'faulting_possible': True,
                        },
                '111': {'stacking': 'ABC', 
                        'slipping_direction': '1/2[111]',
                        'faulting_possible': True,
                        },
            }
        },
        'E21': {
            'info': 'We are doing perovskite-type structure. <space group #221, prototype BaTiO3>. '
                    'I don\'t know the usual gliding plane. ',
            'possible_gliding_planes': {
                '100': {'stacking': 'AB', 
                        'slipping_direction': '1/2[010]',
                        'faulting_possible': True,
                        },
            }
        },
    }

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('skip_uc', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                help='Whether to skip the unitcell calculation')
        spec.input('n_layers', valid_type=orm.Int, required=True, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('slipping_system', valid_type=orm.List, required=True,
                help="""
                """)
        spec.input('do_surface_energy', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                help='Whether to calculate the surface energy')
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
            cls.run_USF,
            cls.inspect_USF,
            if_(cls.should_run_surface_energy)(
                cls.run_surface_energy,
                cls.inspect_surface_energy,
            ),
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message='The `PwCalculation` for the scf run failed.',
        )

        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwCalculation` for the USF run failed.',
        )

        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwCalculation` for the surface energy calculation failed.',
        )

    def validate_slipping_system(self):
        structure_type, gliding_plane, slipping_direction = self.inputs.slipping_system

        if structure_type in self._IMPLEMENTED_SLIPPING_SYSTEMS:
            self.report(f'{self._IMPLEMENTED_SLIPPING_SYSTEMS[structure_type]["info"]}')
            possible_gliding_planes = self._IMPLEMENTED_SLIPPING_SYSTEMS[structure_type]['possible_gliding_planes']
            if gliding_plane in possible_gliding_planes:
                self.report(f'Now we are working on {gliding_plane} gliding plane.')
                if possible_gliding_planes[gliding_plane]['faulting_possible']:
                    self.report(
                        f'Faulting is possible for {gliding_plane} gliding plane.'
                        f'The stacking order is {possible_gliding_planes[gliding_plane]["stacking"]}. '
                        )
                    if slipping_direction:
                        self.report(
                            f'Slipping direction is manually specified: {slipping_direction}. '
                            )
                else:
                    raise ValueError(f'Faulting is not possible for {gliding_plane} gliding plane.')
            else:
                raise ValueError(f'Gliding plane {gliding_plane} not implemented yet. ')
        else:   
            raise NotImplementedError(f'Slipping system {self.inputs.slipping_system} not implemented yet.')

            
    def setup(self):
        """
        Get the unstable faulted structure and kpoints for the supercell.
        """
        structure_sc, kpoints_sc, multiplicity, surface_area, structure_cl, kpoints_cl, multiplicity_cl = get_unstable_faulted_structure_and_kpoints(
            self.inputs.structure, self.inputs.kpoints, self.inputs.n_layers, self.inputs.slipping_system
            )
        
        if not is_primitive_cell(self.inputs.structure):
            self.logger.warning(
                'Composition between the structure and primitive cell are different.' 
                'You might use a non-primitive cell.'
                'The results might be incorrect.')

        self.ctx.structure_sc = structure_sc
        self.ctx.kpoints_sc = kpoints_sc
        self.ctx.multiplicity = multiplicity
        self.ctx.surface_area = surface_area

        self.ctx.structure_cl = structure_cl
        self.ctx.kpoints_cl = kpoints_cl
        self.ctx.multiplicity_cl = multiplicity_cl

    def should_run_uc(self):
        if self.inputs.skip_uc:
            self.logger.warning(
                "You are skipping the unitcell calculation. "
                "Hopefully you know what you are doing!"
            )   
        else:
            self.report(
                "We firstly calculate the total energy of the unit cell. "
            )

        return not self.inputs.skip_uc
    def run_scf(self):
        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace='pw'))

        inputs.metadata.call_link_label = f'scf_uc'

        inputs.structure = self.inputs.structure
        inputs.kpoints = self.inputs.kpoints

        running = self.submit(PwCalculation, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for {self.inputs.structure.get_formula()} unit cell geometry.')

        self.to_context(workchain_scf_uc = running)

    def inspect_scf(self):
        """Verify that the `PwCalculation` for the scf run successfully finished."""
        calcjob = self.ctx.workchain_scf_uc

        if not calcjob.is_finished_ok:
            self.report(
                f"scf {calcjob.process_label} failed with exit status {calcjob.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f"Totel energy of unit cell: {calcjob.outputs.output_parameters.get('energy') / self._RY2eV} Ry")
        
        self.report(f'SCF calculation<{self.ctx.workchain_scf_uc.pk}> finished')


    def run_USF(self):

        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace='pw'))
        inputs.metadata.call_link_label = f'scf_faulted_supercell'


        inputs.structure = self.ctx.structure_sc
        inputs.kpoints = self.ctx.kpoints_sc


        running = self.submit(PwCalculation, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for USF geometry.')

        self.to_context(workchain_scf_faulted_geometry = running)

    def inspect_USF(self):
        calcjob = self.ctx.workchain_scf_faulted_geometry

        if not calcjob.is_finished_ok:
            self.report(
                f"USF {calcjob.process_label} failed with exit status {calcjob.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF
        
        self.report(
            f'SCF calculation<{self.ctx.workchain_scf_faulted_geometry.pk}> for USF geometry finished'
            )
        
    def should_run_surface_energy(self):
        if self.inputs.do_surface_energy:
            self.report(
                "We secondly calculate the surface energy of the faulted geometry. "
            )
        else:
            self.report(
                "We skip the surface energy calculation. "
                )
        return self.inputs.do_surface_energy

    def run_surface_energy(self):
        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace='pw'))
        inputs.metadata.call_link_label = f'surface_energy'

        inputs.structure = self.ctx.structure_cl
        inputs.kpoints = self.ctx.kpoints_cl

        running = self.submit(PwCalculation, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')

        self.to_context(workchain_surface_energy = running)

    def inspect_surface_energy(self):
        
        calcjob = self.ctx.workchain_surface_energy

        if not calcjob.is_finished_ok:
            self.report(
                f"surface energy {calcjob.process_label} failed with exit status {calcjob.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

    def results(self):
        total_energy_faulted_geometry = self.ctx.workchain_scf_faulted_geometry.outputs.output_parameters.get('energy')
        if self.inputs.do_surface_energy:
            total_energy_surface_energy = self.ctx.workchain_surface_energy.outputs.output_parameters.get('energy')
        else:
            total_energy_surface_energy = None

        if not self.inputs.skip_uc:
            total_energy_unit_cell = self.ctx.workchain_scf_uc.outputs.output_parameters.get('energy')

            energy_difference = total_energy_faulted_geometry - total_energy_unit_cell * self.ctx.multiplicity

            ## $\gamma = \frac{E_{USF} - E_{UC}}{2 \times A}$
            USF = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2

            self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')
            
            if self.inputs.do_surface_energy:
                total_energy_slab = self.ctx.workchain_surface_energy.outputs.output_parameters.get('energy')
                SE = (total_energy_unit_cell * self.ctx.multiplicity_cl - total_energy_slab ) * 2 / self.ctx.surface_area * self._eVA22Jm2
                self.report(f'Surface energy: {SE.value} J/m^2')
                
                Rice_ratio = USF / SE
                self.report(f'Rice ratio: {Rice_ratio.value}')
        else:
            self.report(f'Total energy of the faulted geometry: {total_energy_faulted_geometry} eV')
            if self.inputs.do_surface_energy:
                self.report(f'Total energy of the cleaved geometry: {total_energy_surface_energy} eV')
        self.report(f'Results')
