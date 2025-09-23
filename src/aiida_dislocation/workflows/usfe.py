from aiida import orm
from aiida.common import AttributeDict

from aiida.engine import ProcessBuilder, if_, WorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from ..tools.structures import get_unstable_faulted_structure_and_kpoints, is_primitive_cell
from ..data.system import _IMPLEMENTED_SLIPPING_SYSTEMS

from qe_tools import CONSTANTS

class USFEWorkChain(ProtocolMixin, WorkChain):
    """USFE WorkChain"""

    _NAMESPACE = 'usfe'
    _UC_NAMESPACE = 'pw_base_uc'
    _USF_NAMESPACE = 'pw_base_usf'
    _SURFACE_NAMESPACE = 'pw_base_surface'

    _RY2eV    = CONSTANTS.ry_to_ev
    _RYA22Jm2 = CONSTANTS.ry_si * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20

    @classmethod
    def define(cls, spec):
        super().define(spec)


        spec.input('n_layers', valid_type=orm.Int, required=True, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('slipping_system', valid_type=orm.List, required=True,
                help="""
                """)

        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints_distance_uc', valid_type=orm.Float, required=False)
        spec.input('kpoints_uc', valid_type=orm.KpointsData, required=False)

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._UC_NAMESPACE,
            exclude=('structure', 'kpoints', 'kpoints_distance','clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `scf` calculation for the unit cell.'
            }
            )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._USF_NAMESPACE,
            exclude=('structure', 'kpoints', 'kpoints_distance','clean_workdir'),
            namespace_options={
                'required': True,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `scf` calculation for the USF geometry.'
            }
            )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._SURFACE_NAMESPACE,
            exclude=('structure', 'kpoints', 'kpoints_distance','clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `scf` calculation for the surface energy.'
            }
            )
        spec.outline(
            cls.validate_slipping_system,
            cls.setup,
            if_(cls.should_run_uc)(
                cls.run_uc,
                cls.inspect_uc,
            ),
            cls.run_usf,
            cls.inspect_usf,
            if_(cls.should_run_surface)(
                cls.run_surface,
                cls.inspect_surface,
            ),
            cls.results,
        )

        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._UC_NAMESPACE,
            )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._USF_NAMESPACE,
            )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._SURFACE_NAMESPACE,
            )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_UC",
            message='The `PwBaseWorkChain` for the scf run failed.',
        )

        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )

        spec.exit_code(
            403,
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
        protocol=None,
        overrides=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs according to the previous Wannier90OptimizeWorkChain and PhBaseWorkChain and protocol for other namespaces.
        :param code: A code for the different calculations. Should be in the following format:
            code pw.x,
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, protocol)
        builder = cls.get_builder()
        builder.structure = structure
        builder.n_layers = orm.Int(inputs['n_layers'])
        builder.slipping_system = orm.List(inputs['slipping_system'])
        builder.kpoints_distance_uc = orm.Float(inputs['kpoints_distance_uc'])
        builder.kpoints_uc = orm.KpointsData(inputs['kpoints_uc'])

        builder_pw_base_uc = cls.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._UC_NAMESPACE, {}),
            **kwargs
        )
        builder_pw_base_usf = cls.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._USF_NAMESPACE, {}),
            **kwargs
        )
        builder_pw_surface = cls.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._SURFACE_NAMESPACE, {}),
            **kwargs
        )
        builder.pw_base_uc = builder_pw_base_uc
        builder.pw_base_usf = builder_pw_base_usf
        builder.pw_base_surface = builder_pw_surface

        return builder

    def validate_slipping_system(self):
        structure_type, gliding_plane, slipping_direction = self.inputs.slipping_system

        if structure_type in _IMPLEMENTED_SLIPPING_SYSTEMS:
            self.report(f'{_IMPLEMENTED_SLIPPING_SYSTEMS[structure_type]["info"]}')
            possible_gliding_planes = _IMPLEMENTED_SLIPPING_SYSTEMS[structure_type]['possible_gliding_planes']
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
        if 'pw_base_uc' in self.inputs:
            self.logger.warning(
                "You are skipping the unitcell calculation!"
            )
            return True
        else:
            self.report(
                "We firstly calculate the total energy of the unit cell. "
            )
            return False

    def run_uc(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._UC_NAMESPACE,
                )
            )

        inputs.metadata.call_link_label = f'{self._UC_NAMESPACE}'

        inputs.structure = self.inputs.structure
        inputs.kpoints = self.inputs.kpoints

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for unit cell.')

        self.to_context(workchain_pw_base_uc = running)

    def inspect_uc(self):
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_pw_base_uc

        if not workchain.is_finished_ok:
            self.report(
                f"{self._UC_NAMESPACE} {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_UC

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_pw_base_uc,
                PwBaseWorkChain,
                namespace=self._UC_NAMESPACE,
            ),
        )

    def run_usf(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._USF_NAMESPACE,
                )
            )
        inputs.metadata.call_link_label = f'{self._USF_NAMESPACE}'

        inputs.structure = self.ctx.structure_sc
        inputs.kpoints = self.ctx.kpoints_sc

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for unstable faulted geometry.')

        self.to_context(workchain_pw_base_usf = running)

    def inspect_usf(self):
        workchain = self.ctx.workchain_pw_base_usf

        if not workchain.is_finished_ok:
            self.report(
                f"{self._USF_NAMESPACE} {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_pw_base_usf,
                PwBaseWorkChain,
                namespace=self._USF_NAMESPACE,
            ),
        )

    def should_run_surface(self):
        if 'pw_base_surface' in self.inputs:
            self.report(
                "We calculate the surface energy of the faulted geometry. "
            )
            return True
        else:
            self.report(
                "You are skipping the surface energy calculation!"
                )
            return False

    def run_surface(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._SURFACE_NAMESPACE,
                )
            )
        inputs.metadata.call_link_label = f'{self._SURFACE_NAMESPACE}'

        inputs.structure = self.ctx.structure_cl
        inputs.kpoints = self.ctx.kpoints_cl

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for surface energy calculation.')

        self.to_context(workchain_pw_base_surface = running)

    def inspect_surface(self):

        workchain = self.ctx.workchain_pw_base_surface

        if not workchain.is_finished_ok:
            self.report(
                f"{self._SURFACE_NAMESPACE} {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

    def results(self):
        total_energy_faulted_geometry = self.ctx.workchain_pw_base_usf.outputs.output_parameters.get('energy')
        if 'pw_base_surface' in self.inputs:
            total_energy_surface_energy = self.ctx.workchain_pw_base_surface.outputs.output_parameters.get('energy')
        else:
            total_energy_surface_energy = None

        if 'pw_base_uc' in self.inputs:
            total_energy_unit_cell = self.ctx.workchain_pw_base_uc.outputs.output_parameters.get('energy')

            energy_difference = total_energy_faulted_geometry - total_energy_unit_cell * self.ctx.multiplicity

            ## $\gamma = \frac{E_{USF} - E_{UC}}{2 \times A}$
            USF = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2

            self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')

            if 'pw_base_surface' in self.inputs:
                total_energy_slab = self.ctx.workchain_pw_base_surface.outputs.output_parameters.get('energy')
                SE = (total_energy_unit_cell * self.ctx.multiplicity_cl - total_energy_slab ) * 2 / self.ctx.surface_area * self._eVA22Jm2
                self.report(f'Surface energy: {SE.value} J/m^2')

                Rice_ratio = USF / SE
                self.report(f'Rice ratio: {Rice_ratio.value}')
        else:
            self.report(f'Total energy of the faulted geometry: {total_energy_faulted_geometry} eV')
            if 'pw_base_surface' in self.inputs:
                self.report(f'Total energy of the cleaved geometry: {total_energy_surface_energy} eV')


    @staticmethod
    def _clean_workdir(node):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""

        cleaned_calcs = []

        for called_descendant in node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        return cleaned_calcs

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = self._clean_workdir(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

