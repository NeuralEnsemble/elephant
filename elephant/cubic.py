# -*- coding: utf-8 -*-
'''
CuBIC is a statistical method for the detection of higher order of
correlations in parallel spike trains based on the analysis of the
cumulants of the population count.
Given a list sts of SpikeTrains, the analysis comprises the following
steps:

1) compute the population histogram (PSTH) with the desired bin size
       >>> binsize = 5 * pq.ms
       >>> pop_count = elephant.statistics.time_histogram(sts, binsize)

2) apply CuBIC to the population count
       >>> alpha = 0.05  # significance level of the tests used
       >>> xi, p_val, k = cubic(data, ximax=100, alpha=0.05, errorval=4.):

:copyright: Copyright 2016 by the Elephant team, see AUTHORS.txt.
:license: BSD, see LICENSE.txt for details.
'''
# -*- coding: utf-8 -*-
from __future__ import division
import scipy.stats
import scipy.special
import math
import warnings


# Based on matlab code by Benjamin Staude
# Adaptation to python by Pietro Quaglio and Emiliano Torre


def cubic(data, ximax=100, alpha=0.05):
    '''
    Performs the CuBIC analysis [1] on a population histogram, calculated from
    a population of spiking neurons.

    The null hypothesis :math:`H_0: k_3(data)<=k^*_{3,\\xi}` is iteratively
    tested with increasing correlation order :math:`\\xi` (correspondent to
    variable xi) until it is possible to accept, with a significance level alpha,
    that :math:`\\hat{\\xi}` (corresponding to variable xi_hat) is the minimum
    order of correlation necessary to explain the third cumulant
    :math:`k_3(data)`.

    :math:`k^*_{3,\\xi}` is the maximized third cumulant, supposing a Compund
    Poisson Process (CPP) model for correlated spike trains (see [1])
    with maximum order of correlation equal to :math:`\\xi`.

    Parameters
    ----------
    data : neo.AnalogSignal
        The population histogram (count of spikes per time bin) of the entire
        population of neurons.
    ximax : int
         The maximum number of iteration of the hypothesis test:
         if it is not possible to compute the :math:`\\hat{\\xi}` before ximax
         iteration the CuBIC procedure is aborted.
         Default: 100
    alpha : float
         The significance level of the hypothesis tests perfomed.
         Default: 0.05

    Returns
    -------
    xi_hat : int
        The minimum correlation order estimated by CuBIC, necessary to
        explain the value of the third cumulant calculated from the population.
    p : list
        The ordred list of all the p-values of the hypothesis tests that have
        been performed. If the maximum number of iteration ximax is reached the
        last p-value is set to -4
    kappa : list
        The list of the first three cumulants of the data.
    test_aborted : bool
        Wheter the test was aborted because reached the maximum number of
        iteration ximax

    References
    ----------
    [1]Staude, Rotter, Gruen, (2009) J. Comp. Neurosci
    '''
    # alpha in in the interval [0,1]
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'the significance level alpha (= %s) has to be in [0,1]' % alpha)

    if not isinstance(ximax, int) or ximax < 0:
        raise ValueError(
            'The maximum number of iterations ximax(= %i) has to be a positive'
            % alpha + ' integer')

    # dict of all possible rate functions
    try:
        data = data.magnitude
    except AttributeError:
        pass
    L = len(data)

    # compute first three cumulants
    kappa = _kstat(data)
    xi_hat = 1
    xi = 1
    pval = 0.
    p = []
    test_aborted = False

    # compute xi_hat iteratively
    while pval < alpha:
        xi_hat = xi
        if xi > ximax:
            warnings.warn('Test aborted, xihat= %i > ximax= %i' % (xi, ximax))
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
        # p = errorval
        kstar = [0]
        raise ValueError(
            'H_0 can not be tested:'
            'kappa(2)= %f<%f=kappa(1)!!!' % (kappa[1], kappa[0]))
    else:
        # computation of the maximized cumulants
        kstar = [_kappamstar(kappa[:2], i, xi) for i in range(2, 7)]
        k3star = kstar[1]

        # variance of third cumulant (from Stuart & Ord)
        sigmak3star = math.sqrt(
            kstar[4] / L + 9 * (kstar[2] * kstar[0] + kstar[1] ** 2) /
            (L - 1) + 6 * L * kstar[0] ** 3 / ((L - 1) * (L - 2)))
        # computation of the p-value (the third cumulant is supposed to
        # be gaussian istribuited)
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
    data : numpy.aray
        The population histogram of the population on which are computed
        the cumulants

    Returns
    -----
    kappa : list
        The first three cumulants of the population count
    '''
    L = len(data)
    if L == 0:
        raise ValueError('The input data must be a non-empty array')
    S = [(data ** r).sum() for r in range(1, 4)]
    kappa = []
    kappa.append(S[0] / float(L))
    kappa.append((L * S[1] - S[0] ** 2) / (L * (L - 1)))
    kappa.append(
        (2 * S[0] ** 3 - 3 * L * S[0] * S[1] + L ** 2 * S[2]) / (
            L * (L - 1) * (L - 2)))
    return kappa
