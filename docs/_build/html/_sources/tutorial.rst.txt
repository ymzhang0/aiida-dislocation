************
Quick Start
************

This tutorial will guide you through running complete `USFWorkChain` to unstable stacking fault energy.

Step 1: Setup your AiiDA environment
=======================================

First, make sure you are in a running `verdi shell` or have loaded the AiiDA profile in your Python script.

.. code-block:: python

   from aiida import orm, engine

   # Load all the necessary codes
   codes = {
       'pw': orm.load_code('pw@my_cluster'),
   }


Important: please download the package from https://github.com/ymzhang0/aiida-dislocation.

Step 2: Prepare the input structure
====================================

Load the crystal structure you want to calculate.

For dislocation calculation, lead is a good example.

.. code-block:: python

   # Load a structure from its PK or UUID
   structure = read_structure(f"../examples/structures/Pb.xsf")

Step 3: Create the builder
==========================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_dislocation.workflows.usf import USFWorkChain

   builder = USFWorkChain.get_builder_from_protocol(
       codes=codes,
       structure=structure,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       overrides={
           'pw_base_uc': {
               'pw': {
                   'metadata': {'options': {'max_wallclock_seconds': 1800}}
               }
           },
           'pw_base_usf': {
               'pw': {
                   'metadata': {'options': {'max_wallclock_seconds': 1800}}
               }
           },
           'pw_base_surface': {
               'pw': {
                   'metadata': {'options': {'max_wallclock_seconds': 1800}}
               }
           }
       },
   )


Step 4: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   node, results = engine.run_get_node(builder)

Step 5: Inspect the results
===========================

Once the `EpwSuperconWorkChain` has finished successfully, you can inspect its outputs.

.. code-block:: python

   print(f"WorkChain finished with status: {node.process_state}")
   print(f"Available outputs: {results.keys()}")
