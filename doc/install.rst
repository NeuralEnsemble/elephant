.. _install:

****************************
Prerequisites / Installation
****************************

Elephant is a pure Python package so that it should be easy to install on any system.


Dependencies
============

The following packages are required to use Elephant:
    * Python_ >= 2.7
    * numpy_ >= 1.8.2
    * scipy_ >= 0.14.0
    * quantities_ >= 0.10.1
    * neo_ >= 0.5.0

The following packages are optional in order to run certain parts of Elephant:
    * For using the pandas_bridge module: 
        * pandas >= 0.14.1
    * For using the ASSET analysis
    * scikit-learn >= 0.15.1
    * For building the documentation:
        * numpydoc >= 0.5
        * sphinx >= 1.2.2
    * For running tests:
        * nose >= 1.3.3

All dependencies can be found on the Python package index (PyPI).


Debian/Ubuntu
-------------
For Debian/Ubuntu, we recommend to install numpy and scipy as system packages using apt-get::
    
    $ apt-get install python-numpy python-scipy python-pip python-six

Further packages are found on the Python package index (pypi) and should be installed with pip_::
    
    $ pip install quantities
    $ pip install neo

We highly recommend to install these packages using a virtual environment provided by virtualenv_ or locally in the home directory using the ``--user`` option of pip (e.g., ``pip install --user quantities``), neither of which require administrator privileges.

Windows/Mac OS X
----------------

On non-Linux operating systems we recommend using the Anaconda_ Python distribution, and installing all dependencies in a `Conda environment`_, e.g.::

    $ conda create -n neuroscience python numpy scipy pip six
    $ source activate neuroscience
    $ pip install quantities
    $ pip install neo


Installation
============

Automatic installation from PyPI
--------------------------------

The easiest way to install Elephant is via pip_::

    $ pip install elephant    


Manual installation from pypi
-----------------------------

To download and install manually, download the latest package from http://pypi.python.org/pypi/elephant

Then::

    $ tar xzf elephant-0.4.3.tar.gz
    $ cd elephant-0.4.3
    $ python setup.py install
    
or::

    $ python3 setup.py install
    
depending on which version of Python you are using.


Installation of the latest build from source
--------------------------------------------

To install the latest version of Elephant from the Git repository::

    $ git clone git://github.com/NeuralEnsemble/elephant.git
    $ cd elephant
    $ python setup.py install



.. _`Python`: http://python.org/
.. _`numpy`: http://www.numpy.org/
.. _`scipy`: http://scipy.org/scipylib/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`neo`: http://pypi.python.org/pypi/neo
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/
.. _`this snapshot`: https://github.com/NeuralEnsemble/python-neo/archive/snapshot-20150821.zip
.. _Anaconda: http://continuum.io/downloads
.. _`Conda environment`: http://conda.pydata.org/docs/faq.html#creating-new-environments
