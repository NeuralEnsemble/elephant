# -*- coding: utf-8 -*-
"""
This module provides function to estimate causal influences of signals on each
other.

.. include:: causality-overview.rst

.. current_module elephant.causality

Overview of Functions
---------------------

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


"""
Granger causality is a method to determine causal influence of one signal on another based on autoregressive modelling. It
was developed by Nobel prize laureate Clive Granger and has been adopted in various numerical fields ever since [1]. In its
simplest form, the method tests whether the past values of one signal help to reduce the prediction error of another
signal, compared to the past values of the latter signal alone. If it does reduce the prediction error, the first
signal is said to Granger cause the other signal.

The user must be mindful of the method's limitations, which are assumptions of covariance stationary data, linearity imposed by
 the underlying autoregressive modelling as well as the fact that the variables not included in the model will not be accounted for.

Various formulations of Granger causality have been developed. In this module you will find functions for time-series data to test pairwise Granger causality (`pairwise_granger`),
trivariate Granger causality (`trivariate_granger`) and a method to estimate Granger causality for point-process data developed by Krumin and Shoham (`point_process_granger`) [2].

[1] Granger CWJ (1969) Investigating causal relations by econometric models and cross-spectral methods. Econometrica 37:424-438.
[2] Krumin M, Shoham S (2010) Multivariate autoregressive modeling and Granger causality analysis of multiple spike trains. Computational intelligence and neuroscience 2010:1-9.
[3] Ding M, Chen Y, Bressler SL (2006) Granger causality: Basic theory and application to neuroscience. In Schelter S, Winterhalder N, Timmer J. Handbook of Time Series Analysis. Wiley, Wienheim.

[Example] [1] Yu MB, Cunningham JP, Santhanam G, Ryu SI, Shenoy K V, Sahani M (2009)
Gaussian-process factor analysis for low-dimensional single-trial analysis of
neural population activity. J Neurophysiol 102:614-635.


:copyright: Copyright 2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""


# TODO: Pairwise Granger implementation
def pairwise_granger(lfp_signals, order_method='BIC',
                     statistical_method='F-ratio'):
    pass

