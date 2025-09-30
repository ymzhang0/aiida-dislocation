from tkinter import W
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

class SFEBaseWorkChain(ProtocolMixin, WorkChain):
    """SFEBase WorkChain"""

    _NAMESPACE = 'sfebase'
    _PW_RELAX_NAMESPACE = "pw_relax"
    _PW_SCF_NAMESPACE = "pw_scf"
    _PW_SFE_NAMESPACE = "pw_sfe"
    _PW_SURFACE_ENERGY_NAMESPACE = "pw_surface_energy"

    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('n_unit_cells', valid_type=orm.Int, required=False, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('P', valid_type=orm.List, required=False,
                help='The transformation matrix for the supercell. Note that please always put the z axis at the last.')
        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints', valid_type=orm.KpointsData, required=False,)
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.5),
                help='The distance between kpoints for the kpoints generation')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                # 'kpoints',
                # 'kpoints_distance'
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
                # 'kpoints',
                # 'kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SFE_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                # 'kpoints',
                # 'kpoints_distance'
            ),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SURFACE_ENERGY_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                # 'kpoints',
                # 'kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain`.'
            }
        )

        spec.outline(
            cls.setup,
            # cls.generate_kpoints_for_unit_cell,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.generate_faulted_structure,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_sfe,
            cls.inspect_sfe,
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
            namespace=cls._PW_SFE_NAMESPACE,
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

        # Set up the sub-workchains
        for namespace, workchain_type in [
            (cls._PW_RELAX_NAMESPACE, PwRelaxWorkChain),
            (cls._PW_SCF_NAMESPACE, PwBaseWorkChain),
            (cls._PW_SFE_NAMESPACE, PwBaseWorkChain),
            (cls._PW_SURFACE_ENERGY_NAMESPACE, PwBaseWorkChain),
        ]:
            sub_builder = workchain_type.get_builder_from_protocol(
                *args,
                overrides=inputs.get(namespace, {}),
            )
            sub_builder.pop('structure', None)
            sub_builder.pop('clean_workdir', None)

            builder[namespace]._data = sub_builder._data

        builder[cls._PW_RELAX_NAMESPACE].pop('base_final_scf', None)
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        self.ctx.current_structure = self.inputs.structure
        
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

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} unit cell geometry.')

        self.to_context(workchain_relax = running)

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(
                f"relax {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'relax calculation<{workchain.pk}> finished')

        self.ctx.current_structure = workchain.outputs.output_structure
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE,
            ),
        )
        self.ctx.total_energy_unit_cell = workchain.outputs.output_parameters.get('energy')

    def should_run_scf(self):
        # if self._PW_SCF_NAMESPACE in self.inputs:
        #     if self._PW_RELAX_NAMESPACE in self.inputs:
        #         self.report(
        #             "You are running relaxation calculation in unit cell. "
        #             "single scf will be skipped althought it is provided!"
        #         )
        #         return False
        #     else:
        #         self.report(
        #         "You are running single scf calculation in unit cell. "
        #     )
        #         return True
        # else:
        #     self.report(
        #         "We skip the calculation of scf calculation in unit cell. "
        #     )
        #     return False
        return self._PW_SCF_NAMESPACE in self.inputs
    
    def run_scf(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SCF_NAMESPACE
                )
            )

        inputs.metadata.call_link_label = self._PW_SCF_NAMESPACE

        inputs.pw.structure = self.ctx.structure_scf
        # inputs.kpoints = self.ctx.kpoint_scf

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> for {self.inputs.structure.get_formula()} conventional geometry.')

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
        self.ctx.total_energy_conventional_geometry = workchain.outputs.output_parameters.get('energy')

    def inspect_sfe(self):
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
                namespace=self._PW_SFE_NAMESPACE,
            ),
        )
        # self.ctx.total_energy_faulted_geometry = workchain.outputs.output_parameters.get('energy')
        # energy_difference = self.ctx.total_energy_faulted_geometry - self.ctx.total_energy_unit_cell * self.ctx.multiplicity
        # USF = (energy_difference / self.ctx.surface_area) * self._eVA22Jm2 / 2
        # self.ctx.USF = USF
        # self.report(f'Unstable faulted surface energy: {USF.value} J/m^2')


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

        inputs.pw.structure = self.ctx.structure_cl
        # inputs.kpoints = self.ctx.kpoints_cl

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
