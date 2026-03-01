"""
Functions for analysing and estimating firing patterns and connectivity among
neurons in order to better understand the underlying neural networks and
information flow between neurons.


Network connectivity estimation
*******************************

.. autosummary::
    :toctree: _toctree/functional_connectivity/

    total_spiking_probability_edges

References
----------

.. bibliography::
   :keyprefix: functional_connectivity-


:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from elephant.functional_connectivity_src.total_spiking_probability_edges import (
    total_spiking_probability_edges,
)

__all__ = ["total_spiking_probability_edges"]
