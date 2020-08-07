# -*- coding: utf-8 -*-
'''
CuBIC is a statistical method for the detection of higher order of
correlations in parallel spike trains based on the analysis of the
cumulants of the population count.
Given a list sts of SpikeTrains, the analysis comprises the following
steps:

1) compute the population histogram (PSTH) with the desired bin size
       >>> bin_size = 5 * pq.ms
       >>> pop_count = elephant.statistics.time_histogram(sts, bin_size)

2) apply CuBIC to the population count
       >>> alpha = 0.05  # significance level of the tests used
       >>> xi, p_val, k = cubic(data, max_iterations=100, alpha=0.05,
       ... errorval=4.):

:copyright: Copyright 2016 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
'''
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import scipy.stats
import scipy.special
import math
import warnings

from elephant.utils import deprecated_alias


# Based on matlab code by Benjamin Staude
# Adaptation to python by Pietro Quaglio and Emiliano Torre


@deprecated_alias(data='histogram', ximax='max_iterations')
def cubic(histogram, max_iterations=100, alpha=0.05):
    r"""
    Performs the CuBIC analysis [1]_ on a population histogram, calculated
    from a population of spiking neurons.

    The null hypothesis :math:`H_0: k_3(data)<=k^*_{3,\xi}` is iteratively
    tested with increasing correlation order :math:`\xi` until it is possible
    to accept, with a significance level `alpha`, that :math:`\hat{\xi}` is
    the minimum order of correlation necessary to explain the third cumulant
    :math:`k_3(data)`.

    :math:`k^*_{3,\xi}` is the maximized third cumulant, supposing a Compound
    Poisson Process (CPP) model for correlated spike trains (see [1]_)
    with maximum order of correlation equal to :math:`\xi`.

    Parameters
    ----------
    histogram : neo.AnalogSignal
        The population histogram (count of spikes per time bin) of the entire
        population of neurons.
    max_iterations : int, optional
         The maximum number of iterations of the hypothesis test. Corresponds
         to the :math:`\hat{\xi_{\text{max}}}` in [1]_. If it is not possible
         to compute the :math:`\hat{\xi}` before `max_iterations` iteration,
         the CuBIC procedure is aborted.
         Default: 100.
    alpha : float, optional
         The significance level of the hypothesis tests performed.
         Default: 0.05.

    Returns
    -------
    xi_hat : int
        The minimum correlation order estimated by CuBIC, necessary to
        explain the value of the third cumulant calculated from the population.
    p : list
        The ordered list of all the p-values of the hypothesis tests that have
        been performed. If the maximum number of iteration `max_iterations` is
        reached, the last p-value is set to -4.
    kappa : list
        The list of the first three cumulants of the data.
    test_aborted : bool
        Whether the test was aborted because reached the maximum number of
        iteration, `max_iterations`.

    References
    ----------
    .. [1] Staude, Rotter, Gruen, (2009) J. Comp. Neurosci

    """
    # alpha in in the interval [0,1]
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'the significance level alpha (= %s) has to be in [0,1]' % alpha)

    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("'max_iterations' ({}) has to be a positive integer"
                         .format(max_iterations))

    # dict of all possible rate functions
    try:
        histogram = histogram.magnitude
    except AttributeError:
        pass
    L = len(histogram)

    # compute first three cumulants
    kappa = _kstat(histogram)
    xi_hat = 1
    xi = 1
    pval = 0.
    p = []
    test_aborted = False

    # compute xi_hat iteratively
    while pval < alpha:
        xi_hat = xi
        if xi > max_iterations:
            warnings.warn('Test aborted, xihat= %i > ximax= %i' % (
                xi, max_iterations))
            test_aborted = True
            break

        # compute p-value
        pval = _H03xi(kappa, xi, L)
        p.append(pval)
        xi = xi + 1

    return xi_hat, p, kappa, test_aborted


def _H03xi(kappa, xi, L):
    '''
    Computes the p_value for testing  the :math:`H_0: k_3(data)<=k^*_{3,\\xi}`
    hypothesis of CuBIC in the stationary rate version

    Parameters
    -----
    kappa : list
        The first three cumulants of the populaton of spike trains
    xi : int
        The the maximum order of correlation :math:`\\xi` supposed in the
        hypothesis for which is computed the p value of :math:`H_0`
    L : float
        The length of the orginal population histogram on which is performed
        the CuBIC analysis

    Returns
    -----
    p : float
        The p-value of the hypothesis tests
    '''

    # Check the order condition of the cumulants necessary to perform CuBIC
    if kappa[1] < kappa[0]:
        raise ValueError(
            'H_0 can not be tested:'
            'kappa(2) = %f < %f = kappa(1)!!!' % (kappa[1], kappa[0]))
    else:
        # computation of the maximized cumulants
        kstar = [_kappamstar(kappa[:2], i, xi) for i in range(2, 7)]
        k3star = kstar[1]

        # variance of third cumulant (from Stuart & Ord)
        sigmak3star = math.sqrt(
            kstar[4] / L + 9 * (kstar[2] * kstar[0] + kstar[1] ** 2) /
            (L - 1) + 6 * L * kstar[0] ** 3 / ((L - 1) * (L - 2)))
        # computation of the p-value (the third cumulant is supposed to
        # be gaussian distributed)
        p = 1 - scipy.stats.norm(k3star, sigmak3star).cdf(kappa[2])
        return p


def _kappamstar(kappa, m, xi):
    '''
    Computes maximized cumulant of order m

    Parameters
    -----
    kappa : list
        The first two cumulants of the data
    xi : int
        The :math:`\\xi` for which is computed the p value of :math:`H_0`
    m : float
        The order of the cumulant

    Returns
    -----
    k_out : list
        The maximized cumulant of order m
    '''

    if xi == 1:
        kappa_out = kappa[1]
    else:
        kappa_out = \
            (kappa[1] * (xi ** (m - 1) - 1) -
                kappa[0] * (xi ** (m - 1) - xi)) / (xi - 1)
    return kappa_out


def _kstat(data):
    '''
    Compute first three cumulants of a population count of a population of
    spiking
    See http://mathworld.wolfram.com/k-Statistic.html

    Parameters
    -----
    data : numpy.ndarray
        The population histogram of the population on which are computed
        the cumulants

    Returns
    -----
    moments : list
        The first three unbiased cumulants of the population count
    '''
    if len(data) == 0:
        raise ValueError('The input data must be a non-empty array')
    moments = [scipy.stats.kstat(data, n=n) for n in [1, 2, 3]]
    return moments
