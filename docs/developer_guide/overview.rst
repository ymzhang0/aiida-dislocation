=================
Project Structure
=================

This page provides an overview of the ``aiida-dislocation`` repository structure, explaining the role of each key directory and module. The layout is based on standard AiiDA plugin conventions to facilitate maintainability and collaboration.

High-Level Overview
-------------------

The project follows a layered architecture to maximize code reuse and separate concerns.

.. code-block:: text

   aiida-dislocation/
   ├── docs/...
   ├── aiida_dislocation/
   │   ├── data/
   │   │   ├── __init__.py
   │   │   ├── system.py
   │   │   └── ...
   │   ├── examples/...
   │   ├── tools/
   │   │   ├── __init__.py
   │   │   ├── structures.py
   │   │   └── ...
   │   ├── workflows/
   │   │   ├── __init__.py
   │   │   ├── usf.py
   │   │   ├── gsfe.py
   │   │   └── ...
   ├── tests/...
   └── pyproject.toml

Module Breakdown
================

``aiida_dislocation/workflows/``
************************************

This is the core of the plugin, containing all the AiiDA ``WorkChain`` definitions.

``usfe.py``: `USFEWorkChain`
==================================

This work chain is used to calculate the unstable stacking fault energy.

``gsfe.py``: `GSFEWorkChain`
===================================

This work chain is used to calculate the generailzed stacking fault energy.

``aiida_dislocation/data/``
*********************************

This directory contains the data for the dislocation system.

``aiida_dislocation/tools/``
***********************************