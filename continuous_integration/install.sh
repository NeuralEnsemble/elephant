#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Use the miniconda installer for faster download / install of conda
# itself
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
export MINICONDA_PATH=$HOME/miniconda
bash miniconda.sh -b -p ${MINICONDA_PATH}
export PATH=${MINICONDA_PATH}/bin:$PATH
conda config --set always_yes yes
conda update --yes conda

conda install python=${TRAVIS_PYTHON_VERSION} coveralls
pip install -r requirements.txt

if [[ "${INSTALL_MKL}" == "true" ]]; then
    conda install --yes --no-update-dependencies mkl
else
    # Make sure that MKL is not used
    conda remove --yes --features mkl || echo "MKL is not installed"
fi

pip install -r requirements.txt

if [[ "${DISTRIB}" == "extra" ]]; then
    pip install -r requirements-extras.txt
fi

# todo do we need this?
pip install .
