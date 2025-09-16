from tkinter import W
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from ..tools import get_unstable_faulted_structure_and_kpoints, is_primitive_cell

class USFWorkChain(ProtocolMixin, WorkChain):
    """GSFE WorkChain"""

    _NAMESPACE = 'usf'
    _PW_RELAX_NAMESPACE = "pw_relax"
    _PW_SCF_NAMESPACE = "pw_scf"
    _PW_USF_NAMESPACE = "pw_usf"
    _PW_SURFACE_ENERGY_NAMESPACE = "pw_surface_energy"

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

        spec.input('n_layers', valid_type=orm.Int, required=True, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('P', valid_type=orm.List, required=True,
                help='The transformation matrix for the supercell. Note that please always put the z axis at the last.')
        spec.input('slipping_direction', valid_type=orm.List, required=True,
                help='The slipping direction for the dislocation. It should be based on the transformed cell.')
        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints', valid_type=orm.KpointsData, required=True,)
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.5),
                help='The distance between kpoints for the kpoints generation')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SCF_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_USF_NAMESPACE,
            exclude=('structure', 'clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SURFACE_ENERGY_NAMESPACE,
            exclude=('structure', 'clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
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
            cls.run_usf,
            cls.inspect_usf,
            if_(cls.should_run_surface_energy)(
                cls.run_surface_energy,
                cls.inspect_surface_energy,
            ),
            cls.results,
        )

        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._PW_SCF_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )

        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._PW_USF_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )

        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._PW_SURFACE_ENERGY_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_RELAX",
            message='The `PwBaseWorkChain` for the relax run failed.',
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message='The `PwBaseWorkChain` for the scf run failed.',
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )

        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwBaseWorkChain` for the surface energy calculation failed.',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` of the default protocol."""
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)


    @classmethod
    def get_builder_from_protocol(
            cls,
            code,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (code, structure, protocol)

        builder = cls.get_builder()

        # Set up the pw relax sub-workchain
        pw_relax_builder = PwRelaxWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._PW_RELAX_NAMESPACE, {}),
            **kwargs
        )

        pw_relax_builder.pop('structure', None)
        pw_relax_builder.pop('clean_workdir', None)
        pw_relax_builder.pop('base_final_scf', None)

        builder[cls._PW_RELAX_NAMESPACE]._data = pw_relax_builder._data

        # Set up the pw scf sub-workchain
        for namespace in [cls._PW_SCF_NAMESPACE, cls._PW_USF_NAMESPACE, cls._PW_SURFACE_ENERGY_NAMESPACE]:
            pw_base_builder = PwBaseWorkChain.get_builder_from_protocol(
                *args,
                overrides=inputs.get(namespace, {}),
            )
            pw_base_builder.pop('structure', None)
            pw_base_builder.pop('clean_workdir', None)

            builder[namespace]._data = pw_base_builder._data

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder


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
        self.ctx.current_structure = self.inputs.structure

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

    def should_run_relax(self):
        return self._PW_RELAX_NAMESPACE in self.inputs

    def run_relax(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._PW_RELAX_NAMESPACE

        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.inputs.kpoints

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} unit cell geometry.')

        self.to_context(workchain_relax_uc = running)

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax_uc

        if not workchain.is_finished_ok:
            self.report(
                f"relax {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'relax calculation<{workchain.pk}> finished')

        self.ctx.current_structure = workchain.outputs.structure
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE,
            ),
        )
        self.ctx.total_energy_unit_cell = workchain.outputs.output_parameters.get('energy')

    def should_run_scf(self):
        if self._PW_SCF_NAMESPACE in self.inputs:
            if self._PW_RELAX_NAMESPACE in self.inputs:
                self.report(
                    "You are running relaxation calculation in unit cell. "
                    "single scf will be skipped althought it is provided!"
                )
                return False
            else:
                self.report(
                "You are running single scf calculation in unit cell. "
            )
                return True
        else:
            self.report(
                "We skip the calculation of scf calculation in unit cell. "
            )
            return False

    def run_scf(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SCF_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._PW_SCF_NAMESPACE

        inputs.structure = self.inputs.structure
        inputs.kpoints = self.inputs.kpoints

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} unit cell geometry.')

        self.to_context(workchain_scf_uc = running)

    def inspect_scf(self):
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf_uc

        if not workchain.is_finished_ok:
            self.report(
                f"scf {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f"Totel energy of unit cell: {workchain.outputs.output_parameters.get('energy') / self._RY2eV} Ry")

        self.report(f'SCF calculation<{self.ctx.workchain_scf_uc.pk}> finished')
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_SCF_NAMESPACE,
            ),
        )
        self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')

    def run_usf(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_USF_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_USF_NAMESPACE


        inputs.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints_sc


        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for USF geometry.')

        self.to_context(workchain_scf_faulted_geometry = running)

    def inspect_usf(self):
        workchain = self.ctx.workchain_scf_faulted_geometry

        if not workchain.is_finished_ok:
            self.report(
                f"USF {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(
            f'SCF calculation<{workchain.pk}> for USF geometry finished'
            )
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_USF_NAMESPACE,
            ),
        )
        self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        USF = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        self.ctx.USF = USF
        self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')


    def should_run_surface_energy(self):
        if self._PW_SURFACE_ENERGY_NAMESPACE in self.inputs:
            self.report(
                "We are running surface energy calculation. "
            )
            return True
        else:
            self.report(
                "We skip the surface energy calculation. "
                )
            return False

    def run_surface_energy(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SURFACE_ENERGY_NAMESPACE
                )
            )
        inputs.metadata.call_link_label = self._PW_SURFACE_ENERGY_NAMESPACE

        inputs.structure = self.ctx.structure_cl
        inputs.kpoints = self.ctx.kpoints_cl

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')

        self.to_context(workchain_surface_energy = running)

    def inspect_surface_energy(self):

        workchain = self.ctx.workchain_surface_energy

        if not workchain.is_finished_ok:
            self.report(
                f"surface energy {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_SURFACE_ENERGY_NAMESPACE,
            ),
        )
        total_energy_slab = workchain.outputs.output_parameters.get('energy')
        self.report(f'Total energy of the cleaved geometry: {total_energy_slab} eV')
        SE = ( self.ctx.total_energy_unit_cell * self.ctx.multiplicity_cl - total_energy_slab ) * 2 / self.ctx.surface_area * self._eVA22Jm2
        self.report(f'Surface energy: {SE.value} J/m^2')
        self.ctx.SE = SE
        Rice_ratio = self.ctx.USF / SE
        self.report(f'Rice ratio: {Rice_ratio.value}')

    def results(self):
        pass
