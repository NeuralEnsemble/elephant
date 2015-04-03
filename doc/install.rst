.. _install:

****************************
Prerequisites / Installation
****************************

elephant is a pure Python package so that it should be easy to install on any system.


Dependencies
============

The following packages are required to use elephant:
    * Python_ >= 2.6
    * numpy_ >= 1.6.2
    * scipy_>=0.11.0
    * quantities_ >= 0.9.0
    * neo_ == 0.4.0

The following packages are optional in order to run certain parts of elephant:
    * For using the pandas_bridge module: 
        * pandas>=>=0.14.0
    * For building the documentation:
        * numpydoc==0.5
        * sphinx==1.2.2
    * For running tests:
        * nose==1.3.3

All dependencies can be found on the Python package index (pypi).


Debian/Ubuntu
-------------
For Debian/Ubuntu, we recommend to install numpy and scipy as system packages using apt-get::
    
    $ apt-get install python-numpy python-scipy python-pip

Further packages are found on the Python package index (pypi) and should be installed and pip_::
    
    $ pip install quantities
    $ pip install neo

We highly recommend to install these packages locally in the home directory using the ``--user`` option of pip (e.g., ``pip install --user quantities``), or using a virtual environment provided by virtualenv_ , both of which do not require administrator privileges.



Installation
============


Automatic installation from pypi
--------------------------------

The easiest option to install elephant is via pip_::

    $ pip install elephant    

*Linux:* We recommend to install elephant locally in the home directory using the ``--user`` option of pip, or using a virtual environment provided by virtualenv_ , both of which do not require administrator privileges.

Alternatively, if you have setuptools_, please use::
    
    $ easy_install elephant


Manual installation from pypi
-----------------------------

To download and install manually, download the latest package from:

    http://pypi.python.org/pypi/elephant

Then::

    $ tar xzf elephant-0.1.tar.gz
    $ cd elephant-0.1
    $ python setup.py install
    
or::

    $ python3 setup.py install
    
depending on which version of Python you are using.


Installation of the latest build from source
--------------------------------------------

To install the latest version of Neo from the Git repository::

    $ git clone git://github.com/NeuralEnsemble/elephant.git
    $ cd elephant
    $ python setup.py install



.. _`Python`: http://python.org/
.. _`numpy`: http://numpy.scipy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`neo`: http://pypi.python.org/pypi/neo
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
