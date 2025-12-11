from tkinter import W
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain, PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_dislocation.tools import (
    get_faulted_structures,
)

class RelaxSpacingWorkChain(ProtocolMixin, WorkChain):
    """Relax Spacing WorkChain"""

    _NAMESPACE = 'relax'

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.3),
                help='The distance between kpoints for the kpoints generation')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.input('pseudo_family', valid_type=orm.Str, required=False, default=lambda: orm.Str('SSSP/1.0/PBE/efficiency'),
                    help='The pseudo family to use for the calculation.')
        spec.input('additional_spacings', valid_type=orm.List, required=False, default=lambda: orm.List(),
                    help='The additional spacings to add to the structure.')
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._NAMESPACE,
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

        spec.outline(
            cls.setup,
            while_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_RELAX",
            message='The `PwRelaxWorkChain` for the relax run failed.',
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

        overrides = inputs.get(cls._NAMESPACE, {})
        overrides['pseudo_family'] = inputs.get('pseudo_family', None)
        sub_builder = PwRelaxWorkChain.get_builder_from_protocol(
                *args,
                overrides=overrides,
            )
        sub_builder.pop('clean_workdir', None)
        sub_builder.pop('base_init_relax', None)
        builder[cls._NAMESPACE]._data = sub_builder._data

        builder.structure = structure
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.additional_spacings = inputs['additional_spacings']
        return builder

    def setup(self):
        self.ctx.current_structure = self.inputs.structure
        self.ctx.iteration = 1
        self.ctx.structures = get_faulted_structures(
            self.inputs.structure, 'removal', 1, 1, self.inputs.additional_spacings)

    def should_run_relax(self):
        
        if self.ctx.structures == []:
            return False
        
        return True
    
    def run_relax(self):

        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._NAMESPACE
                )
            )

        inputs.metadata.call_link_label = f'relax_{self.ctx.iteration}'

        inputs.structure = self.ctx.structures.pop()
        self.ctx.iteration += 1
        inputs.base_relax.kpoints_distance = self.inputs.kpoints_distance

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for {self.ctx.current_structure.get_formula()} faulted unit cell geometry.')

        return {f"workchain_relax_{self.ctx.iteration}": append_(running)}

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax[-1]

        if not workchain.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{workchain.pk}> for {self.ctx.current_structure.get_formula()} faulted unit cell geometry failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'PwRelaxWorkChain<{workchain.pk}> for {self.inputs.structure.get_formula()} unit cell geometry finished OK')


    def results(self):
        pass
