# -*- coding: utf-8 -*-
"""
Elephant is a package for the analysis of neurophysiology data, based on Neo.

:copyright: Copyright 2014-2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from . import (statistics,
               spike_train_generation,
               spike_train_correlation,
               spectral,
               spike_train_surrogates,
               signal_processing,
               current_source_density,
               sta,
               conversion,
               neo_tools)

try:
    from . import pandas_bridge
    from . import asset
    from . import csd #current_source_density
except ImportError:
    pass

__version__ = "0.2.1"
