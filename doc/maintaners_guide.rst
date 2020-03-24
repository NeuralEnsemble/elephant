=================
Maintainers guide
=================

This guide is for Elephant maintainers only.


Python 3
--------

Elephant should work with Python 2.7 and Python 3.

So far, we have managed to write code that works with both Python 2 and 3.
Mainly this involves and putting

.. code-block:: python

    from __future__ import division, print_function, unicode_literals

at the beginning of each source file. The most important thing to remember is
to run tests with at least one version of Python 2 and at least one version of
Python 3.

If in doubt, `Porting to Python 3 <http://python3porting.com/>`_ by Lennart
Regebro is an excellent resource.

All code should conform as much as possible to
`PEP 8 <http://www.python.org/dev/peps/pep-0008/>`_.

Each source file should have a copyright and a license note.


Structure of the doc/ folder
----------------------------

Documentation in Elephant is written exclusively using the ``sphinx`` package
and resides in the ``doc`` folder, in addition to the docstrings contained of
the ``elephant`` modules. In the following, we outline the main components of
the Elephant documentation.


Top-level documentation
~~~~~~~~~~~~~~~~~~~~~~~

General information about the Elephant package and a gentle introduction are
contained in various ``.rst`` files in the top-level directory of the Elephant
package. Here, :file:`index.rst` is the central starting point, and the hierarchical
document structure is specified using the ``toctree`` directives. In particular,
these files contain a general introduction and tutorial on Elephant, the
release notes of Elephant versions, and this development guide.


Module and function reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All modules in Elephant are semi-automatically documented. To this end, for
each module ``x`` a file ``doc/reference/x.rst`` exists with the following
contents:

.. code:: rst

    ============================
    `x` - Short descriptive name
    ============================

    .. automodule:: elephant.x

This instructs Sphinx to add the module documentation in the module docstring
into the file.

The module docstring of ``elephant/x.py`` is also standardized in its structure:

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


Each module documentation starts with a short, understandable introduction to
the functionality of the module, the "Overview". This text is written in a
separate file residing in `doc/reference/x-overview.rst`, and is included on
the first line. This keeps the docstring in the code short.

Next, we specify the current module as `x`, in order to avoid confusion if
a module uses submodules.

In the following, all functions of the module are listed in an order that is
intuitive for users. Where it makes sense, these functions can be thematically
grouped using third-level headings. For small modules, no such headings are
needed.



Making a release
----------------

1. Increment the Elephant package version in :file:`elephant/VERSION`.

2. Add a section in :file:`doc/release_notes.rst`, describing in short the
changes made from the previous release.

3. Check that the copyright statement (in :file:`LICENSE.txt`,
:file:`README.md`, and :file:`doc/conf.py`) is correct.

4. If there is a new module do not forget to add the modulename to the
:file:`doc/modules.rst` and make a file with a short description in
:file:`doc/reference/<modulename>.rst`.

To build a source package (see `Packaging Python Projects
<https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_)::

    $ pip install --user --upgrade twine
    $ python setup.py sdist

To upload the package to `PyPI <http://pypi.python.org>`_
(if you have the necessary permissions)::

    $ python -m twine upload dist/elephant-x.x.x.tar.gz

Finally, tag the release in the Git repository and push it::

    $ git tag <version>
    $ git push --tags upstream

Here, version should be of the form ``vX.Y.Z``.
