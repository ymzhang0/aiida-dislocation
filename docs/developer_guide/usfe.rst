=======================
USFWorkChain
=======================

This work chain is used to calculate the unstable stacking fault energy.

It is a wrapper of the following three PwBaseWorkChain work chains:

- pw_base_uc: calculate the total energy of the unit cell (*optional*)
- pw_base_usf: calculate the total energy of the faulted geometry
- pw_base_surface: calculate the total energy of the cleaved geometry (*optional*)

The workflow is organized as follows:

.. code-block:: python

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

When *validate_slipping_system*, 