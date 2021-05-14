# -*- coding: utf-8 -*-
"""
Elephant is a package for the analysis of neurophysiology data, based on Neo.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
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
               change_point_detection,
               phase_analysis,
               sta,
               conversion,
               neo_tools,
               cell_assembly_detection,
               spade,
               waveform_features,
               gpfa)

# not included modules on purpose:
#   parallel: avoid warns when elephant is imported

try:
    from . import asset
    from . import spade
except ImportError:
    # requirements-extras are missing
    # please install Elephant with `pip install elephant[extras]`
    pass


def _get_version():
    import os
    elephant_dir = os.path.dirname(__file__)
    with open(os.path.join(elephant_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


__version__ = _get_version()
