.. _install:

************
Installation
************

The easiest way to install Elephant is by creating a conda environment, followed by ``pip install elephant``.
Below is the explanation of how to proceed with these two steps.


.. _prerequisites:

Prerequisites
=============

Elephant requires Python_ 2.7, 3.5, 3.6, 3.7, or 3.8.

.. tabs::


    .. tab:: (recommended) Conda (Linux/MacOS/Windows)

        1. Create your conda environment (e.g., `elephant_env`):

           .. code-block:: sh

              conda create --name elephant_env python=3.7 numpy scipy tqdm

        2. Activate your environment:

           .. code-block:: sh

              conda activate elephant_env


    .. tab:: Debian/Ubuntu

        Open a terminal and run:

        .. code-block:: sh

           sudo apt-get install python-pip python-numpy python-scipy python-pip python-six python-tqdm



Installation
============

.. tabs::


    .. tab:: Stable release version

        The easiest way to install Elephant is via pip_:

           .. code-block:: sh

              pip install elephant

        To upgrade to a newer release use the ``--upgrade`` flag:

           .. code-block:: sh

              pip install --upgrade elephant

        If you do not have permission to install software systemwide, you can
        install into your user directory using the ``--user`` flag:

           .. code-block:: sh

              pip install --user elephant

        To install Elephant with all extra packages, do:

           .. code-block:: sh

              pip install elephant[extras]


    .. tab:: Development version

        If you have `Git <https://git-scm.com/>`_ installed on your system,
        it is also possible to install the development version of Elephant.

        1. Before installing the development version, you may need to uninstall
           the previously installed version of Elephant:

           .. code-block:: sh

              pip uninstall elephant

        2. Clone the repository and install the local version:

           .. code-block:: sh

              git clone git://github.com/NeuralEnsemble/elephant.git
              cd elephant
              pip install -e .



Dependencies
------------

The following packages are required to use Elephant (refer to requirements_ for the exact package versions):

    * numpy_ - fast array computations
    * scipy_ - scientific library for Python
    * quantities_ - support for physical quantities with units (mV, ms, etc.)
    * neo_ - electrophysiology data manipulations
    * tqdm_ - progress bar
    * six_ - Python 2 and 3 compatibility utilities

These packages are automatically installed when you run ``pip install elephant``.

The following packages are optional in order to run certain parts of Elephant:

    * `pandas <https://pypi.org/project/pandas/>`_ - for the :doc:`pandas_bridge <reference/pandas_bridge>` module
    * `scikit-learn <https://pypi.org/project/scikit-learn/>`_ - for the :doc:`ASSET <reference/asset>` analysis
    * `nose <https://pypi.org/project/nose/>`_ - for running tests
    * `numpydoc <https://pypi.org/project/numpydoc/>`_ and `sphinx <https://pypi.org/project/Sphinx/>`_ - for building the documentation

These and above packages are automatically installed when you run ``pip install elephant[extras]``.

.. _`Python`: http://python.org/
.. _`numpy`: http://www.numpy.org/
.. _`scipy`: https://www.scipy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`neo`: http://pypi.python.org/pypi/neo
.. _`pip`: http://pypi.python.org/pypi/pip
.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _`Conda environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _`tqdm`: https://pypi.org/project/tqdm/
.. _`six`: https://pypi.org/project/six/
.. _requirements: https://github.com/NeuralEnsemble/elephant/blob/master/requirements/requirements.txt
.. _PyPI: https://pypi.org/
