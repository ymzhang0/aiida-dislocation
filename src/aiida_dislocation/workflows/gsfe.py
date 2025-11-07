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

from ..tools import get_unstable_faulted_structure


class GSFEWorkChain(ProtocolMixin, WorkChain):
    """GSFE WorkChain"""

    _NAMESPACE = 'gsfe'

    _SCF_NAMESPACE = "scf"
    _USF_NAMESPACE = "usf"
    _SURFACE_ENERGY_NAMESPACE = "surface_energy"
    _CURVE_NAMESPACE = "curve"

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
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace=cls._PW_CURVE_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        
        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_RELAX",
            message='The `PwBaseWorkChain` for the GSF run failed.',
        )
        
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_USF",
            message='The `PwBaseWorkChain` for the USF run failed.',
        )
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_CURVE",
            message='The `PwBaseWorkChain` for the curve run failed.',
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY",
            message='The `PwBaseWorkChain` for the surface energy run failed.',
        )
        
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

            if namespace != cls._PW_RELAX_NAMESPACE:
                sub_builder.pop('kpoints', None)
                sub_builder.pop('kpoints_distance', None)

            builder[namespace]._data = sub_builder._data

        builder[cls._PW_RELAX_NAMESPACE].pop('base_final_scf', None)
        builder[cls._PW_RELAX_NAMESPACE]['base'].pop('kpoints', None)
        builder[cls._PW_RELAX_NAMESPACE]['base'].pop('kpoints_distance', None)
        builder.structure = structure
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.gliding_plane = orm.Str(inputs.get('gliding_plane', ''))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder
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
