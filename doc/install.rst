.. _install:

************
Installation
************

Elephant requires Python 2.7, 3.5, 3.6, 3.7, or 3.8. If you do not already have a Python environment configured on your computer, please see the instructions for installing the :ref:`dependencies`. Otherwise, scroll to :ref:`installation`.


.. _dependencies:

Dependencies
============

The following packages are required to use Elephant (refer to requirements_ for the exact package versions):
    * Python_ 2.7, 3.5
    * numpy_ - fast array computations
    * scipy_ - scientific library for Python
    * quantities_ - support for physical quantities with units (mV, ms, etc.)
    * neo_ - electrophysiology data manipulations
    * tqdm_ - progress bar
    * six_ - Python 2 and 3 compatibility utilities

The following packages are optional in order to run certain parts of Elephant:
    * `pandas <https://pypi.org/project/pandas/>`_ - for the :doc:`pandas_bridge <reference/pandas_bridge>` module
    * `scikit-learn <https://pypi.org/project/scikit-learn/>`_ - for the :doc:`ASSET <reference/asset>` analysis
    * `nose <https://pypi.org/project/nose/>`_ - for running tests
    * `numpydoc <https://pypi.org/project/numpydoc/>`_ and `sphinx <https://pypi.org/project/Sphinx/>`_ - for building the documentation

All dependencies can be found on the Python package index (PyPI_).


Conda (Windows/Linux/macOS)
--------------------------

We recommend using the Anaconda_ Python distribution and installing all dependencies in a `Conda environment`_, e.g.::

    $ conda create -n neuroscience python numpy scipy pip six tqdm
    $ conda activate neuroscience
    $ pip install quantities neo



Debian/Ubuntu
-------------
For Debian/Ubuntu, you can install numpy and scipy as system packages using apt-get::

    $ sudo apt-get install python-pip python-numpy python-scipy python-pip python-six python-tqdm

Further packages are found on the Python package index (PyPI_) and can only be installed with pip_::

    $ pip install quantities neo


.. _installation:

Installation
============

Install the released version
----------------------------

The easiest way to install Elephant is via pip_::

    $ pip install elephant    

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade elephant

If you do not have permission to install software systemwide, you can install into your user directory using the ``--user`` flag::

    $ pip install --user elephant

To install Elephant with all extra packages, do::

    $ pip install elephant[extras]


Install the development version
-------------------------------

If you have `Git <https://git-scm.com/>`_ installed on your system, it is also possible to install the development version of Elephant.

Before installing the development version, you may need to uninstall the standard version of Elephant using pip::

    $ pip uninstall elephant

Then do::

    $ git clone git://github.com/NeuralEnsemble/elephant.git
    $ cd elephant
    $ pip install -e .

or::

    $ pip install git+https://github.com/NeuralEnsemble/elephant.git


.. _`Python`: http://python.org/
.. _`numpy`: http://www.numpy.org/
.. _`scipy`: http://scipy.org/scipylib/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`neo`: http://pypi.python.org/pypi/neo
.. _`pip`: http://pypi.python.org/pypi/pip
.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _`Conda environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _`tqdm`: https://pypi.org/project/tqdm/
.. _`six`: https://pypi.org/project/six/
.. _requirements: https://github.com/NeuralEnsemble/elephant/blob/master/requirements.txt
.. _PyPI: https://pypi.org/
