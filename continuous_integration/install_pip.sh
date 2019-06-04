#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

pip install -r requirements.txt

#if [[ "${DISTRIB}" == "extra" ]]; then
#    pip install -r requirements-extras.txt
#fi
