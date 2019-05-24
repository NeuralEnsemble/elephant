#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version
pip list

if [[ "${DISTRIB}" == "extra" ]]; then
    # extra packages covers mpi
    mpiexec -n 1 nosetests --with-coverage --cover-package=elephant
else
    nosetests --with-coverage --cover-package=elephant
fi
