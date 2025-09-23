=======================
GSFEWorkChain
=======================

This work chain is used to calculate the generalized stacking fault energy.

It is a wrapper of the following three PwBaseWorkChain work chains:

- pw_base_uc: calculate the total energy of the unit cell (*optional*)
- pw_base_sf: calculate the total energy of a series of faulted geometries
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
        while_(cls.should_continue_sf)(
            cls.run_sf,
            cls.inspect_sf,
            ),
        if_(cls.should_run_surface)(
            cls.run_surface,
            cls.inspect_surface,
        ),
        cls.results,
    )

Firstly, the work chain validates the slipping system. It will generate a series of faulted geometries for the given slipping system.

Then, the work chain will calculate the total energy of the unit cell. This is optional.

Then, the work chain will calculate the total energy of the faulted geometries. It will repeat this process until there is no remaining faulted geometry.

Then, the work chain will calculate the total energy of the cleaved geometry. This is optional.

Finally, the work chain will return the results.
