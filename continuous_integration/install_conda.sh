#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Use the miniconda installer for faster download / install of conda
# itself
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
export MINICONDA_PATH=$HOME/miniconda
bash miniconda.sh -b -p ${MINICONDA_PATH}
export PATH=${MINICONDA_PATH}/bin:$PATH
conda config --set always_yes yes
conda update conda

conda install python==${TRAVIS_PYTHON_VERSION} pip
conda install mkl  # should be installed first not to override next
conda config --append channels conda-forge
sed -i '/^neo/d' requirements.txt  # remove neo from requirements.txt
conda install --file requirements.txt
# python-neo conda package is not well supported (outdated, for example)
# not constraining python-neo to a specific version thus
conda install -c conda-forge python-neo
pip list
