"""`CalcJob` implementation for the pw.x code of Quantum ESPRESSO."""
import os
import warnings

from aiida import orm
from aiida.common import AttributeDict, datastructures, exceptions
from aiida.common.lang import classproperty
from aiida.plugins import factories

from aiida.engine import CalcJob
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations import _uppercase_dict

from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

class Thermo_pwBaseCalculation(PwCalculation):
    """
    Base class for Thermo_pw calculations.
    Thermo_pw share the same input file as PwCalculation. 
    We directly reuse the definition of PwCalculation with an extra thermo_control input.
    """
    _DEFAULT_THERMO_CONTROL = 'thermo_control'
    _DEFAULT_INPUT_FILE = 'aiida.in'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _DEFAULT_RETRIEVE_LIST = [
        _DEFAULT_OUTPUT_FILE,
        'elastic_constants/*',
        'gnuplot_files/*',
        'thermo_files/*',
        ]
    
    _COMPULSORY_NAMELISTS = [
        'INPUT_THERMO',
        ]
    _ENABLED_KEYWORDS = [ 'what', 'find_ibrav', 'frozen_ions' ]

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'metadata.options.parser_name', 
            valid_type=str, 
            default='dislocation.thermo_pw',
            help='The parser to use for the calculation.'
            )

        
        spec.input(
            'code', 
            valid_type=orm.Code, 
            help='The thermo_pw.x code to run the calculation.'
            )
        
        spec.input(
            'thermo_control', 
            valid_type=orm.Dict, 
            help='The parameters for thermo_control file.'
            )
        
        spec.output(
            'output_parameters', 
            valid_type=orm.Dict,
            help='The `output_parameters` output node of the successful calculation.'
            )
        
        spec.default_output_node = 'output_parameters'


    def prepare_for_submission(self, folder):

        # Reuse the prepare_for_submission method of PwCalculation

        calcinfo = super().prepare_for_submission(folder)

        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}   

        # local_copy_list = calcinfo.local_copy_list

        thermo_control = self.inputs.thermo_control.get_dict()

        for flag in thermo_control.keys():
            if flag not in self._ENABLED_KEYWORDS:
                raise exceptions.InputValidationError(
                    f"'{flag}' flag is not enabled for now."
                )
                
        with folder.open(self._DEFAULT_THERMO_CONTROL, 'w') as handle:
            handle.write('&INPUT_THERMO\n')
            for key, value in thermo_control.items():
                handle.write(convert_input_to_namelist_entry( key, value))
            handle.write('/\n')

        # local_copy_list.append(
        #     self._DEFAULT_THERMO_CONTROL
        #     )
        
        cmdline_params = self._add_parallelization_flags_to_cmdline_params(cmdline_params=settings.pop('CMDLINE', []))

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid    = self.inputs.code.uuid
        codeinfo.cmdline_params = (list(cmdline_params) + ['-in', self.metadata.options.input_filename])
        codeinfo.stdout_name  = self.metadata.options.output_filename

        calcinfo.codes_info     = [codeinfo]
        calcinfo.retrieve_list.extend(self._DEFAULT_RETRIEVE_LIST)
        # calcinfo.local_copy_list = local_copy_list

        return calcinfo