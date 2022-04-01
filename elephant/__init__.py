# -*- coding: utf-8 -*-
"""
Elephant is a package for the analysis of neurophysiology data, based on Neo.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from . import (
    cell_assembly_detection,
    change_point_detection,
    conversion,
    cubic,
    current_source_density,
    kernels,
    neo_tools,
    phase_analysis,
    signal_processing,
    spade,
    spectral,
    spike_train_correlation,
    spike_train_dissimilarity,
    spike_train_generation,
    spike_train_surrogates,
    spike_train_synchrony,
    sta,
    statistics,
    unitary_event_analysis,
    waveform_features,
)

# not included modules on purpose:
#   parallel: avoid warns when elephant is imported

try:  # the following require requirements-extras
    from . import asset
    from . import gpfa
except ImportError:  # pragma: no cover
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
