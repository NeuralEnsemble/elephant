# -*- coding: utf-8 -*-
"""
This module provides function to estimate causal influences of signals on each
other.

.. include:: causality-overview.rst

.. current_module elephant.causality

Overview of Functions
---------------------
Various formulations of Granger causality have been developed. In this module you will find functions for time-series data to test pairwise Granger causality (`pairwise_granger`), trivariate Granger causality (`trivariate_granger`) and a method to estimate Granger causality for point-process data developed by Krumin and Shoham (`point_process_granger`) [2].

Time-series Granger causality
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: statistics/

    pairwise_granger
    trivariate_granger
    point_process_granger


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

# TODO: Pairwise Granger implementation
def pairwise_granger(lfp_signals, order_method='BIC',
                     statistical_method='F-ratio'):
    pass

