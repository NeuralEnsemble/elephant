#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

sudo apt-get update -qq
if [[ "$INSTALL_ATLAS" == "true" ]]; then
    sudo apt-get install -qq libatlas3gf-base libatlas-dev
fi

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
    conda config --set always_yes yes --set changeps1 no
    conda update --yes conda
    conda info -a

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
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
    conda config --set always_yes yes --set changeps1 no
    conda update --yes conda
    conda info -a

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION pandas=$PANDAS_VERSION
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

    if [[ "$COVERAGE" == "true" ]]; then
        pip install coveralls
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    # sudo apt-get install -qq python-nose python-pip \
    #    python-pandas python-coverage
    sudo apt-get build-dep python-scipy 
    deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
     virtualenv --system-site-packages testenv
     source testenv/bin/activate
     pip install nose
     pip install coverage
     pip install numpy==1.6.2
     travis_wait pip install scipy==0.14.0 --verbose
     pip install pandas
     pip install quantities

fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coveralls
fi

# pip install neo==0.3.3
wget https://github.com/NeuralEnsemble/python-neo/archive/apibreak.tar.gz
tar -xzvf apibreak.tar.gz
pushd python-neo-apibreak
python setup.py install
popd

pip install .
