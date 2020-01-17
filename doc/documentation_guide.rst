============================
Elephant Documentation Guide
============================

This guide describes the guidelines for writing Elephant documentation, and the technical background of how the Sphinx_ documentation package is configured.


Structure of the Elephant documentation
---------------------------------------

Documentation in Elephant is written exclusively using the `sphinx` package, and resides in the `doc` folder, in addition to the docstrings contained of the `elephant` modules. In the following, we outline the main components of the Elephant documenation.


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

This instructs Sphinx to add the module documentation in the module docstring into the file.

The module docstring of `elephant/x.py` is also standardized in its structure:

.. code:: rst 

    .. include:: x-overview.rst
    
    .. current_module elephant.x
    
    Overview of Functions
    ---------------------
    
    <<Heading 1 to group functions (optional)>>
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    .. autosummary::
        :toctree: x/
    
        function1
        function2

    <<Heading 2 to group functions (optional)>>
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    .. autosummary::
        :toctree: x/
    
        function3
        function4

    :copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.

Each module documentation starts with a short, understandable introduction to the functionality of the module, the "Overview". This text is written in a separate file residing in `doc/reference/x-overview.rst`, and is included on the first line. This keeps the docstring in the code short.

Next, we specify the current module as `x`, in order to avoid confusion if a module uses submodules.

In the following, all functions of the module are listed in an order that is intuitive for users. Where it makes sense, these functions can be thematically grouped using third-level headings. For small modules, no such headings are needed.

Finally, a copyright statement and the license statements are inserted.


Writing documentation for Elephant
----------------------------------

:: _`Sphinx`: https://www.sphinx-doc.org
