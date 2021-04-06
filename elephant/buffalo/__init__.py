# -*- coding: utf-8 -*-
"""
Buffalo is a package that implements electrophysiology analysis objects to
produce standardized outputs and provenance
capture during the analysis workflow in Elephant.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

try:
    from . import objects
    from .decorator import (Provenance, activate, deactivate,
                            save_provenance, save_graph, print_history)
    HAVE_PROV = True
except ImportError:
    # requirements-prov are missing
    HAVE_PROV = False


__version__ = '0.0.1'


USE_ANALYSIS_OBJECTS = False
USE_NIX = False
