from aiida_quantumespresso.parsers.base import BaseParser
from aiida_quantumespresso.utils.mapping import get_logging_container


from aiida import orm
import numpy
import re

class ThermoPwParser(BaseParser):
    """
    Parser for the ThermoPwCalculation calculation job class.
    """
    
    success_string='JOB DONE.'
    
    def parse(self, **kwargs):
        """Parse the retrieved files of a completed ``ThermoPwCalculation`` into output nodes."""
        logs = get_logging_container()

        stdout, parsed_data, logs = self.parse_stdout_from_retrieved(logs)

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)

        parsed_thermo_pw, logs = self.parse_stdout(stdout, logs)
        parsed_data.update(parsed_thermo_pw)

        self.out('output_parameters', orm.Dict(parsed_data))

        if 'ERROR_OUTPUT_STDOUT_INCOMPLETE' in logs.error:
            return self.exit(self.exit_codes.get('ERROR_OUTPUT_STDOUT_INCOMPLETE'), logs)

        return self.exit(logs=logs)

    @staticmethod
    def parse_stdout(stdout, logs):
        """Parse the ``stdout``."""

        parsed_data = {}
        
        head_regex = re.compile(
            r"^\s*-{2,}\s*\n"               # 行首空格 + 两个以上 '-' + 换行
            r"\s*Applying the following strain",  # 下一行的关键字
            re.MULTILINE
        )
        head_index = sorted([m.start() for m in head_regex.finditer(stdout)])
        foot_regex = re.compile(r"""
            ^\s*[-]{20,}\s*\n                
            \s*Elastic\ constant         
            """, re.MULTILINE | re.VERBOSE)

        foot_index = foot_regex.search(stdout).start()

        work_regex =  re.compile(r'Doing work\s+(\d+)\s*/\s*(\d+)')

        work_data_regex = (
        ('fermi_energy', float, re.compile(r'\s+the Fermi energy is \s+([\d\.-]+)\seV')),
        ('total_energy', float, re.compile(r'!\s*total\s+energy\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('smearing_contrib', float, re.compile(r'smearing\s+contrib.*?=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('internal_energy', float, re.compile(r'internal\s+energy\s+E=F\+TS\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('one_electron', float, re.compile(r'one-electron\s+contribution\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('hartree', float, re.compile(r'hartree\s+contribution\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('xc', float, re.compile(r'xc\s+contribution\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('ewald', float, re.compile(r'ewald\s+contribution\s*=\s*([-\d\.E+]+)\s*Ry', re.IGNORECASE)),
        ('pressure', float, re.compile(r'pressure.*?P=\s*([-\d\.E+]+)')),
        )
        
        strain_regex = re.compile(
            r"Applying the following strain\s*\n"
            r"\s*\(\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*\)\s*\n"
            r"\s*\(\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*\)\s*\n"
            r"\s*\(\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*,\s*([-\d\.E+]+)\s*\)",
            re.MULTILINE
        )
        stress_regex = re.compile(
            r"^\s*([-\d\.E+]+)\s+([-\d\.E+]+)\s+([-\d\.E+]+)\s+", 
            re.MULTILINE
        )

        for work_start, work_end in zip(head_index, head_index[1:] + [foot_index]):
            work_block = stdout[work_start:work_end]
            work = re.search(work_regex, work_block).group(1)
            parsed_data[str(work)] = {}
            
            match = strain_regex.search(work_block)
            strain = numpy.array(list(map(float, match.groups()))).reshape(3, 3).tolist()
            parsed_data[work]['strain'] = strain

            for data_key, type, re_pattern in work_data_regex:
                match = re.search(re_pattern, work_block)
                if match:
                    parsed_data[work][data_key] = type(match.group(1))
                    
            match = stress_regex.search(work_block)
            stress = [list(map(float, match.groups())) for match in stress_regex.finditer(work_block)]
        
            parsed_data[work]['stress'] = stress

        foot = stdout[foot_index:]
        
        fitting_regex = re.compile(r'''
            Elastic\ constant\s+(\d+)\s+(\d+)    
            .*?                                 
            strain\s+stress.*?\n                
            (                                   
            (?:\s*[-\d\.E+]+\s+[-\d\.E+]+\s*\n)+   
            )
            .*?                                 
            Polynomial\ coefficients\s*\n
            \s*a1=\s*([-\d\.E+]+)\s*\n           
            \s*a2=\s*([-\d\.E+]+)\s*\n           
            \s*a3=\s*([-\d\.E+]+)                
        ''', re.DOTALL | re.VERBOSE)
        
        parsed_data['fitting'] = {}
        
        for m in fitting_regex.finditer(foot):
            i, j = m.group(1), m.group(2)
            data_block = m.group(3).strip().splitlines()
            strains, stresses = zip(*(map(float, line.split()[:2]) for line in data_block))        # 三个系数
            a1, a2, a3 = map(float, m.group(4,5,6))
            
            parsed_data['fitting'].update({
                i: {
                    j: {
                        'strains': strains,
                        'stresses': stresses,
                        'a1': a1,
                        'a2': a2,
                        'a3': a3
                    }
                }
            })


        tensor_data_regex = (
            ('Cij', re.compile(
                r"""Elastic\s+constants\s+C_ij.*?\n      
                    \s*i\s+j=.*?\n                       
                    (                                   
                    (?:^\s*[1-6]\s+                  
                        (?:[-\d\.E+]+(?:\s+|$)){6}    
                        \s*\n            
                    ){6}               
                    )           
                """,
                re.MULTILINE | re.DOTALL | re.VERBOSE)
            ),
            ('Sij', re.compile(
                r"""Elastic\s+compliances\s+S_ij.*?\n     
                    \s*i\s+j=.*?\n                     
                    (                         
                    (?:^\s*[1-6]\s+                   
                        (?:[-\d\.E+]+(?:\s+|$)){6}      
                        \s*\n                          
                    ){6}                               
                    )                                   
                """,
                re.MULTILINE | re.DOTALL | re.VERBOSE)
            )
        )
        for data_key, regex in tensor_data_regex:
            m = regex.search(foot)
            block = m.group(1)

            rows = block.strip().splitlines()
            tensor = [list(map(float, row.split()[1:])) for row in rows]
            parsed_data[data_key] = tensor

        modulus_regex = {
            'Voigt': re.compile(
                r"Voigt approximation:\s*"
                r"Bulk modulus\s*B\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Young modulus\s*E\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Shear modulus\s*G\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Poisson Ratio\s*n\s*=\s*([-\d\.E+]+)\s*"
                r"Pugh Ratio\s*r\s*=\s*([-\d\.E+]+)",
                re.DOTALL
            ),
            'Reuss': re.compile(
                r"Reuss approximation:\s*"
                r"Bulk modulus\s*B\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Young modulus\s*E\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Shear modulus\s*G\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Poisson Ratio\s*n\s*=\s*([-\d\.E+]+)\s*"
                r"Pugh Ratio\s*r\s*=\s*([-\d\.E+]+)",
                re.DOTALL
            ),
            'VRH': re.compile(
                r"Voigt-Reuss-Hill average of the two approximations:\s*"
                r"Bulk modulus\s*B\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Young modulus\s*E\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Shear modulus\s*G\s*=\s*([-\d\.E+]+)\s*kbar\s*"
                r"Poisson Ratio\s*n\s*=\s*([-\d\.E+]+)\s*"
                r"Pugh Ratio\s*r\s*=\s*([-\d\.E+]+)",
                re.DOTALL
            )
        }

        parsed_data['modulus'] = {}
        for data_key, re_pattern in modulus_regex.items():
            m = re_pattern.search(foot)
            B, E, G, n, r = m.groups()
            parsed_data['modulus'][data_key] = {
                'B': float(B),
                'E': float(E),
                'G': float(G),
                'n': float(n),
                'r': float(r),
            }

        sound_velocity_regex = re.compile(
            r"Voigt-Reuss-Hill average; sound velocities:\s*"
            r"Compressional V_P\s*=\s*([-\d\.E+]+)\s*m/s\s*"
            r"Bulk          V_B\s*=\s*([-\d\.E+]+)\s*m/s\s*"
            r"Shear         V_G\s*=\s*([-\d\.E+]+)\s*m/s\s*",
            re.DOTALL
            )
        
        m = sound_velocity_regex.search(foot)
        V_P, V_B, V_G = m.groups()
        parsed_data['sound_velocity'] = {
            'V_P': float(V_P),
            'V_B': float(V_B),
            'V_G': float(V_G),
        }
        
        debye_regex = re.compile(
            r"Average Debye sound velocity =\s*([-\d\.E+]+)\s*m/s\s*"
            r"Debye temperature\s*=\s*([-\d\.E+]+)\s*K\s*",
            re.DOTALL
        )
        
        m = debye_regex.search(foot)
        V_D, T_D = m.groups()
        parsed_data['debye'] = {
            'V_D': float(V_D),
            'T_D': float(T_D),
        }
        # for data_key, data_marker, block_parser in data_block_marker_parser:
        #     if data_marker in line:
        #         parsed_data[data_key] = block_parser(stdout[line_number:])

        return parsed_data, logs