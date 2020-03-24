.. _documentation_guide:

===================
Documentation Guide
===================


Writing the documentation
-------------------------

Each module (python source file) should start with a short description of the
listed functionality. Class and function docstrings should conform to the
`NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

.. note:: We highly recommend exploring our :ref:`style_guide`.


Building the documentation
--------------------------

The documentation in :file:`doc/` folder is written in `reStructuredText
<http://docutils.sourceforge.net/rst.html>`_, using the
`Sphinx <http://sphinx-doc.org/>`_ documentation system. To build the
documentation::

1. Install requirements-docs.txt and requirements-tutorials.txt in the same way
   as it's explained in :ref:`developers_guide` step 3::

    $ pip install -r requirements/requirements-docs.txt
    $ pip install -r requirements/requirements-tutorials.txt

2. Build the documentation::

    $ cd doc
    $ export PYTHONPATH=.:../..  # to find elephant package
    $ make html

3. Open :file:`_build/html/index.html` in your browser.

4. (Optional) To check that all URLs in the documentation are correct, run::

    $ make linkcheck

