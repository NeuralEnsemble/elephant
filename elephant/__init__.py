# -*- coding: utf-8 -*-
"""
Elephant is a package for the analysis of neurophysiology data, based on Neo.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from . import (statistics,
               spike_train_generation,
               spike_train_correlation,
               unitary_event_analysis,
               cubic,
               spectral,
               kernels,
               spike_train_dissimilarity,
               spike_train_surrogates,
               signal_processing,
               current_source_density,
               sta,
               conversion,
               neo_tools,
               spade)

try:
    from . import pandas_bridge
    from . import asset
except ImportError:
    pass

__version__ = "0.4.3"
