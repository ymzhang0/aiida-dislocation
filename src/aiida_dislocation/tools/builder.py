from ..workflows import USFWorkChain, GSFEWorkChain

def get_builder(workchain_type: str, **kwargs):
    if workchain_type == 'usf':
        return USFWorkChain.get_builder()
    elif workchain_type == 'gsfe':
        return GSFEWorkChain.get_builder()
    else:
        raise ValueError(f"Invalid workchain type: {workchain_type}")