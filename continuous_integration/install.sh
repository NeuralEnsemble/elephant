#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

if [[ "$DISTRIB" == "conda_min" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage \
        six=$SIX_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv
    conda install libgfortran=1

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes --no-update-dependencies mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

elif [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage six=$SIX_VERSION \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION pandas=$PANDAS_VERSION scikit-learn
    source activate testenv
    conda install libgfortran=1

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes --no-update-dependencies mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

    if [[ "$COVERAGE" == "true" ]]; then
        pip install coveralls
    fi

    python -c "import pandas; import os; assert os.getenv('PANDAS_VERSION') == pandas.__version__"

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    virtualenv --system-site-packages testenv
    source testenv/bin/activate
    pip install nose
    pip install coverage
    pip install numpy==$NUMPY_VERSION
    pip install scipy==$SCIPY_VERSION
    pip install six==$SIX_VERSION
    pip install quantities
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coveralls
fi

# pip install neo==0.3.3
wget https://github.com/NeuralEnsemble/python-neo/archive/snapshot-20150821.tar.gz
tar -xzvf snapshot-20150821.tar.gz
pushd python-neo-snapshot-20150821
python setup.py install
popd

pip install .


python -c "import numpy; import os; assert os.getenv('NUMPY_VERSION') == numpy.__version__"
python -c "import scipy; import os; assert os.getenv('SCIPY_VERSION') == scipy.__version__"
