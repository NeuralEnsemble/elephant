.. _install:

============
Installation
============

The easiest way to install Elephant is by creating a conda environment, followed by ``pip install elephant``.
Below is the explanation of how to proceed with these two steps.


.. _prerequisites:

*************
Prerequisites
*************

Elephant requires `Python <http://python.org/>`_ 3.6, 3.7, 3.8, or 3.9.

.. tabs::


    .. tab:: (recommended) Conda (Linux/MacOS/Windows)

        1. Create your conda environment (e.g., `elephant`):

           .. code-block:: sh

              conda create --name elephant python=3.7 numpy scipy tqdm

        2. Activate your environment:

           .. code-block:: sh

              conda activate elephant


    .. tab:: Debian/Ubuntu

        Open a terminal and run:

        .. code-block:: sh

           sudo apt-get install python-pip python-numpy python-scipy python-pip python-six python-tqdm


************
Installation
************

.. tabs::


    .. tab:: Stable release version

        The easiest way to install Elephant is via `pip <http://pypi.python.org/pypi/pip>`_:

           .. code-block:: sh

              pip install elephant

        If you want to use advanced features of Elephant, install the package
        with extras:

           .. code-block:: sh

              pip install elephant[extras]


        To upgrade to a newer release use the ``--upgrade`` flag:

           .. code-block:: sh

              pip install --upgrade elephant

        If you do not have permission to install software systemwide, you can
        install into your user directory using the ``--user`` flag:

           .. code-block:: sh

              pip install --user elephant


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

        .. tabs::

            .. tab:: Minimal setup

                .. code-block:: sh

                    pip install -e .


            .. tab:: conda (with extras)

                .. code-block:: sh

                    conda remove -n elephant --all  # remove the previous environment
                    conda env create -f requirements/environment.yml
                    conda activate elephant
                    pip install -e .

***********
MPI support
***********

Some Elephant modules (ASSET, SPADE, etc.) are parallelized to run with MPI.
In order to make use of MPI parallelization, you need to install ``mpi4py``
package:

.. tabs::

    .. tab:: conda (easiest)

        .. code-block:: sh

            conda install -c conda-forge mpi4py

    .. tab:: pip (Linux)

        .. code-block:: sh

            sudo apt install -y libopenmpi-dev openmpi-bin
            pip install mpi4py

To run a python script that supports MPI parallelization, run in a terminal:

.. code-block:: sh

    mpiexec -n numprocs python -m mpi4py pyfile [arg] ...

For more information, refer to `mpi4py
<https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html>`_ documentation.


***********************
CUDA and OpenCL support
***********************

:ref:`asset` module supports CUDA and OpenCL. These are experimental features.
You can have one, both, or none installed in your system.

.. tabs::

    .. tab:: CUDA

        To leverage CUDA acceleration on an NVIDIA GPU card, `CUDA toolkit
        <https://developer.nvidia.com/cuda-downloads>`_ must installed on
        your system. Then run the following command in a terminal:

        .. code-block:: sh

            pip install pycuda

        In case you experience issues installing PyCUDA, `this guide
        <https://medium.com/leadkaro/setting-up-pycuda-on-ubuntu-18-04-for-
        gpu-programming-with-python-830e03fc4b81>`_ offers a step-by-step
        installation manual.

        If PyCUDA is detected and installed, CUDA backend is used by default in
        Elephant ASSET module. To turn off CUDA support, set ``ELEPHANT_USE_CUDA``
        environment flag to ``0``.


    .. tab:: OpenCL

        If you have a laptop with a built-in Intel Graphics Card, you can still
        leverage significant performance optimization with OpenCL backend.
        The simplest way to install PyOpenCL is to run a conda command:

        .. code-block:: sh

            conda install -c conda-forge pyopencl intel-compute-runtime

        However, if you have root (sudo) privileges, it's recommended to install
        up-to-date `Intel Graphics Compute Runtime
        <https://github.com/intel/compute-runtime/releases>`_ system-wide and then
        install PyOpenCL as follows:

        .. code-block:: sh

            conda install -c conda-forge pyopencl ocl-icd-system

        Set ``ELEPHANT_USE_OPENCL`` environment flag to ``0`` to turn off
        PyOpenCL support.

        .. note::

            Make sure you've disabled GPU Hangcheck as described in the
            `Intel GPU developers documentation <https://software.intel.com/
            content/www/us/en/develop/documentation/get-started-with-intel-
            oneapi-base-linux/top/before-you-begin.html>`_. Do it with caution -
            using your graphics card to perform computations may make the system
            unresponsive until the compute program terminates.


************
Dependencies
************

Elephant relies on two special packages, installed by default:

    * `quantities <http://pypi.python.org/pypi/quantities>`_ - support for physical quantities with units (mV, ms, etc.)
    * `neo <http://pypi.python.org/pypi/neo>`_ - electrophysiology data manipulations
