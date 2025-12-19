from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from ..tools import get_unstable_faulted_structure_and_kpoints


class GSFEWorkChain(ProtocolMixin, WorkChain):
    """GSFE WorkChain"""

    _NAMESPACE = 'gsfe'

    _PW_RELAX_NAMESPACE = "relax"
    _PW_SCF_NAMESPACE = "scf"
    _PW_USF_NAMESPACE = "usf"
    _PW_SURFACE_ENERGY_NAMESPACE = "surface_energy"
    _PW_CURVE_NAMESPACE = "curve"



    @classmethod
    def define(cls, spec):
        super().define(spec)


        spec.input('n_repeats', valid_type=orm.Int, required=False, default=lambda: orm.Int(4),
                help='The number of layers in the supercell')
        spec.input('gliding_plane', valid_type=orm.Str, required=False, default=lambda: orm.Str(),
                help='The normal vector for the supercell. Note that please always put the z axis at the last.')
        spec.input('structure', valid_type=orm.StructureData, required=True,)
        spec.input('kpoints_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.3),
                help='The distance between kpoints for the kpoints generation')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.input('n_layers', valid_type=orm.Int, required=False,
                    help='Number of layers for GSFE calculation.')
        spec.input('slipping_system', valid_type=orm.List, required=False,
                    help='Slipping system for GSFE calculation.')
        spec.input('only_USF', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
                    help='If True, only calculate USF.')

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            exclude=('structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SCF_NAMESPACE,
            exclude=('pw.structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for SCF calculation.'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_USF_NAMESPACE,
            exclude=('pw.structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for USF calculation.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_SURFACE_ENERGY_NAMESPACE,
            exclude=('pw.structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for surface energy calculation.'
            }
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace=cls._PW_CURVE_NAMESPACE,
            exclude=('pw.structure', 'clean_workdir', 'kpoints', 'kpoints_distance'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for curve calculation.'
            }
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
            while_(cls.should_run_curve)(
                while_(cls.should_run_sfe)(
                    cls.setup_supercell_kpoints,
                    cls.run_sfe,
                    cls.inspect_sfe,
                ),
            ),
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
            (cls._PW_USF_NAMESPACE, PwBaseWorkChain),
            (cls._PW_CURVE_NAMESPACE, PwBaseWorkChain),
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

        if cls._PW_RELAX_NAMESPACE in builder:
            builder[cls._PW_RELAX_NAMESPACE].pop('base_final_scf', None)
            if 'base_relax' in builder[cls._PW_RELAX_NAMESPACE]:
                builder[cls._PW_RELAX_NAMESPACE]['base_relax'].pop('kpoints', None)
                builder[cls._PW_RELAX_NAMESPACE]['base_relax'].pop('kpoints_distance', None)
        builder.structure = structure
        builder.kpoints_distance = orm.Float(inputs['kpoints_distance'])
        builder.gliding_plane = orm.Str(inputs.get('gliding_plane', ''))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder
    def setup(self):
        self.ctx.current_structure = self.inputs.structure
        self.ctx.sf_count = 0
        self.ctx.sf_energy = []
        self.ctx.sf_stress = []
        self.ctx.sf_strain = []
        self.ctx.sf_displacement = []

    def should_run_relax(self):
        return self._PW_RELAX_NAMESPACE in self.inputs

    def run_relax(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = f'relax_uc'
        inputs.structure = self.inputs.structure
        inputs.base_relax.kpoints_distance = self.inputs.kpoints_distance
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> for unitcell {self.inputs.structure.get_formula()}')
        return ToContext(workchain_relax_uc=running)

    def inspect_relax(self):
        workchain = self.ctx.workchain_relax_uc
        if not workchain.is_finished_ok:
            self.report(f'Relax calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.report(f'Relax calculation<{self.ctx.workchain_relax_uc.pk}> finished')

        self.ctx.current_structure = workchain.outputs.output_structure
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE
            )
        )

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

        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints_distance = self.inputs.kpoints_distance

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for unitcell {self.inputs.structure.get_formula()}')

        return ToContext(workchain_scf_uc=running)

    def inspect_scf(self):
        workchain = self.ctx.workchain_scf_uc
        if not workchain.is_finished_ok:
            self.report(f'SCF calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.report(f'SCF calculation<{self.ctx.workchain_scf_uc.pk}> finished')
        
        # Extract kpoints from workchain
        if 'create_kpoints_from_distance' in [link.label for link in workchain.base.links.get_outgoing()]:
            self.ctx.kpoints = workchain.base.links.get_outgoing(
                link_label_filter='create_kpoints_from_distance'
            ).first().node.outputs.result
        else:
            # Fallback: create kpoints from distance
            from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
            kpoints_inputs = {
                'structure': orm.StructureData(ase=self.ctx.current_structure.get_ase()),
                'distance': self.inputs.kpoints_distance,
                'metadata': {'call_link_label': 'create_kpoints_from_distance'}
            }
            self.ctx.kpoints = create_kpoints_from_distance(**kpoints_inputs)
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_SCF_NAMESPACE
            )
        )

    def should_run_usf(self):
        if not hasattr(self.inputs, 'only_USF') or not self.inputs.only_USF.value:
            return False
        return self._PW_USF_NAMESPACE in self.inputs

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
        
        n_layers = self.inputs.n_layers.value if hasattr(self.inputs, 'n_layers') and self.inputs.n_layers else None
        slipping_system = self.inputs.slipping_system.get_list() if hasattr(self.inputs, 'slipping_system') and self.inputs.slipping_system else None
        
        if n_layers is None or slipping_system is None:
            raise ValueError('n_layers and slipping_system are required for USF calculation.')
        
        structure_sc, kpoints_sc = get_unstable_faulted_structure_and_kpoints(
            structure_uc, kpoints_uc, n_layers, slipping_system
        )
        inputs.pw.structure = structure_sc
        inputs.kpoints = kpoints_sc

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for faulted supercell.')

        return ToContext(workchain_scf_faulted_supercell=running)

    def inspect_usf(self):
        workchain = self.ctx.workchain_scf_faulted_supercell
        if not workchain.is_finished_ok:
            self.report(f'USF calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_USF

        self.report(f'USF calculation<{self.ctx.workchain_scf_faulted_supercell.pk}> finished')
        self.ctx.current_structure = workchain.outputs.output_structure
        
        # Extract kpoints if available
        if 'kpoints' in workchain.outputs:
            self.ctx.kpoints = workchain.outputs.kpoints
        elif 'create_kpoints_from_distance' in [link.label for link in workchain.base.links.get_outgoing()]:
            self.ctx.kpoints = workchain.base.links.get_outgoing(
                link_label_filter='create_kpoints_from_distance'
            ).first().node.outputs.result
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_USF_NAMESPACE
            )
        )

    def should_run_curve(self):
        return self._PW_CURVE_NAMESPACE in self.inputs

    def run_curve(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_CURVE_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._PW_CURVE_NAMESPACE
        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for curve calculation.')
        return ToContext(workchain_curve=running)

    def inspect_curve(self):
        workchain = self.ctx.workchain_curve
        if not workchain.is_finished_ok:
            self.report(f'Curve calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CURVE

        self.report(f'Curve calculation<{self.ctx.workchain_curve.pk}> finished')
        self.ctx.current_structure = workchain.outputs.output_structure
        
        # Extract kpoints if available
        if 'kpoints' in workchain.outputs:
            self.ctx.kpoints = workchain.outputs.kpoints
        elif 'create_kpoints_from_distance' in [link.label for link in workchain.base.links.get_outgoing()]:
            self.ctx.kpoints = workchain.base.links.get_outgoing(
                link_label_filter='create_kpoints_from_distance'
            ).first().node.outputs.result
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_CURVE_NAMESPACE
            )
        )

    def should_run_surface_energy(self):
        return self._PW_SURFACE_ENERGY_NAMESPACE in self.inputs

    def run_surface_energy(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                PwBaseWorkChain,
                namespace=self._PW_SURFACE_ENERGY_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._PW_SURFACE_ENERGY_NAMESPACE
        inputs.pw.structure = self.ctx.current_structure
        inputs.kpoints = self.ctx.kpoints
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwCalculation<{running.pk}> for surface energy calculation.')
        return ToContext(workchain_surface_energy=running)

    def inspect_surface_energy(self):
        workchain = self.ctx.workchain_surface_energy
        if not workchain.is_finished_ok:
            self.report(f'Surface energy calculation<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SURFACE_ENERGY

        self.report(f'Surface energy calculation<{self.ctx.workchain_surface_energy.pk}> finished')
        self.ctx.current_structure = workchain.outputs.output_structure
        
        # Extract kpoints if available
        if 'kpoints' in workchain.outputs:
            self.ctx.kpoints = workchain.outputs.kpoints
        
        # Expose outputs
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBaseWorkChain,
                namespace=self._PW_SURFACE_ENERGY_NAMESPACE
            )
        )

    def should_run_sfe(self):
        """Check if there are more curve points to process."""
        # This should be implemented based on GSFE curve generation logic
        # For now, return False to skip the inner loop
        return False
    
    def setup_supercell_kpoints(self):
        """Setup kpoints for the current curve point."""
        # This should be implemented based on GSFE curve generation logic
        pass
    
    def run_sfe(self):
        """Run the SFE calculation for current curve point."""
        # This should be implemented based on GSFE curve generation logic
        pass
    
    def inspect_sfe(self):
        """Inspect the SFE calculation results for current curve point."""
        # This should be implemented based on GSFE curve generation logic
        pass

    def results(self):
        """Output collected results."""
        self.report(f'Results')
