from ..workflows import USFEWorkChain, GSFEWorkChain

def get_builder(workchain_type: str, **kwargs):
    """Get a builder for the specified workchain type.
    
    :param workchain_type: Type of workchain ('usfe' or 'gsfe')
    :param kwargs: Additional arguments to pass to get_builder()
    :return: WorkChain builder
    """
    if workchain_type == 'usfe':
        return USFEWorkChain.get_builder(**kwargs)
    elif workchain_type == 'gsfe':
        return GSFEWorkChain.get_builder(**kwargs)
    else:
        raise ValueError(
            f"Invalid workchain type: {workchain_type}. "
            f"Supported types: 'usfe', 'gsfe'"
        )