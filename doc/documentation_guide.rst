============================
Elephant Documentation Guide
============================

This guide describes the guidelines for writing Elephant documentation, and the technical background of how the `sphinx` documentation package is configured.


Structure of the Elephant documentation
---------------------------------------

Documentation in Elephant is written exclusively using the `sphinx` package, and resides in the `doc` folder, in addition to the docstrings contained of the `elephant` modules. In the following, we outline the main components of the Elephant documenation 


Top-level documentation
~~~~~~~~~~~~~~~~~~~~~~~

General information about the Elephant package and a gentle introduction are contained in various `.rst` files in the top-level directory of the Elephant package. Here, `index.rst` is the central starting point, and the hierarchical document structure is specified using the `toctree` directives. In particular, these files contain a general introduction and tutorial on Elephant, the release notes of Elephant versions, and this development guide.


Module and function reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All modules in Elephant are semi-automatically documented. To this end, for each module `x` a file `doc/reference/x.rst` exists with the following contents:

.. code:: rst

    ============================
    `x` - Short descriptive name
    ============================

    .. automodule:: elephant.x

This instructs sphinx to add the module documentation in the module docstring into the file.


Writing documentation for Elephant
----------------------------------
