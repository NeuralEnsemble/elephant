"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
:cite:`gpfa-Yu2008_1881` for neural trajectory visualization of parallel spike
trains. GPFA applies factor analysis (FA) to time-binned spike count data to
reduce the dimensionality and at the same time smoothes the resulting
low-dimensional trajectories by fitting a Gaussian process (GP) model to them.

The input consists of a set of trials (Y), each containing a list of spike
trains (N neurons). The output is the projection (X) of the data in a space
of pre-chosen dimensionality x_dim < N.

Under the assumption of a linear relation (transform matrix C) between the
latent variable X following a Gaussian process and the spike train data Y with
a bias d and  a noise term of zero mean and (co)variance R (i.e.,
:math:`Y = C X + d + Gauss(0,R)`), the projection corresponds to the
conditional probability E[X|Y].
The parameters (C, d, R) as well as the time scales and variances of the
Gaussian process are estimated from the data using an expectation-maximization
(EM) algorithm.

Internally, the analysis consists of the following steps:

0) bin the spike train data to get a sequence of N dimensional vectors of spike
counts in respective time bins, and choose the reduced dimensionality x_dim

1) expectation-maximization for fitting of the parameters C, d, R and the
time-scales and variances of the Gaussian process, using all the trials
provided as input (c.f., `gpfa_core.em()`)

2) projection of single trials in the low dimensional space (c.f.,
`gpfa_core.exact_inference_with_ll()`)

3) orthonormalization of the matrix C and the corresponding subspace, for
visualization purposes: (c.f., `gpfa_core.orthonormalize()`)


.. autosummary::
    :toctree: _toctree/gpfa

    GPFA


Visualization
-------------
Visualization of GPFA transforms is covered in Viziphant:
https://viziphant.readthedocs.io/en/latest/modules.html


Tutorial
--------

:doc:`View tutorial <../tutorials/gpfa>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/gpfa.ipynb


Original code
-------------
The code was ported from the MATLAB code based on Byron Yu's implementation.
The original MATLAB code is available at Byron Yu's website:
https://users.ece.cmu.edu/~byronyu/software.shtml

:copyright: Copyright 2014-2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import neo
import numpy as np
import quantities as pq
import sklearn
import warnings

from elephant.gpfa import gpfa_core, gpfa_util
from elephant.utils import deprecated_alias


__all__ = [
    "GPFA"
]


class GPFA(sklearn.base.BaseEstimator):
    """
    Apply Gaussian process factor analysis (GPFA) to spike train data

    There are two principle scenarios of using the GPFA analysis, both of which
    can be performed in an instance of the GPFA() class.

    In the first scenario, only one single dataset is used to fit the model and
    to extract the neural trajectories. The parameters that describe the
    transformation are first extracted from the data using the `fit()` method
    of the GPFA class. Then the same data is projected into the orthonormal
    basis using the method `transform()`. The `fit_transform()` method can be
    used to perform these two steps at once.

    In the second scenario, a single dataset is split into training and test
    datasets. Here, the parameters are estimated from the training data. Then
    the test data is projected into the low-dimensional space previously
    obtained from the training data. This analysis is performed by executing
    first the `fit()` method on the training data, followed by the
    `transform()` method on the test dataset.

    The GPFA class is compatible to the cross-validation functions of
    `sklearn.model_selection`, such that users can perform cross-validation to
    search for a set of parameters yielding best performance using these
    functions.

    Parameters
    ----------
    x_dim : int, optional
        state dimensionality
        Default: 3
    bin_size : float, optional
        spike bin width in msec
        Default: 20.0
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    em_max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tau_init : float, optional
        GP timescale initialization in msec
        Default: 100
    eps_init : float, optional
        GP noise variance initialization
        Default: 1e-3
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Attributes
    ----------
    valid_data_names : tuple of str
        Names of the data contained in the resultant data structure, used to
        check the validity of users' request
    has_spikes_bool : np.ndarray of bool
        Indicates if a neuron has any spikes across trials of the training
        data.
    params_estimated : dict
        Estimated model parameters. Updated at each run of the fit() method.

        covType : str
            type of GP covariance, either 'rbf', 'tri', or 'logexp'.
            Currently, only 'rbf' is supported.
        gamma : (1, #latent_vars) np.ndarray
            related to GP timescales of latent variables before
            orthonormalization by :math:`bin_size / sqrt(gamma)`
        eps : (1, #latent_vars) np.ndarray
            GP noise variances
        d : (#units, 1) np.ndarray
            observation mean
        C : (#units, #latent_vars) np.ndarray
            loading matrix, representing the mapping between the neuronal data
            space and the latent variable space
        R : (#units, #latent_vars) np.ndarray
            observation noise covariance
    fit_info : dict
        Information of the fitting process. Updated at each run of the fit()
        method.

        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
        log_likelihoods : list
            log likelihoods after each EM iteration.
    transform_info : dict
        Information of the transforming process. Updated at each run of the
        transform() method.

        log_likelihood : float
            maximized likelihood of the transformed data
        num_bins : nd.array
            number of bins in each trial
        Corth : (#units, #latent_vars) np.ndarray
            mapping between the neuronal data space and the orthonormal
            latent variable space

    Methods
    -------
    fit
    transform
    fit_transform
    score

    Raises
    ------
    ValueError
        If `bin_size` or `tau_init` is not a `pq.Quantity`.

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    independent Poisson spike trains recorded in 50 trials with randomized
    rates up to 100 Hz.

    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa import GPFA
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> data = []
    >>> for trial in range(50):
    >>>     n_channels = 20
    >>>     firing_rates = np.random.randint(low=1, high=100,
    ...                                      size=n_channels) * pq.Hz
    >>>     spike_times = [homogeneous_poisson_process(rate=rate)
    ...                    for rate in firing_rates]
    >>>     data.append((trial, spike_times))
    ...
    >>> gpfa = GPFA(bin_size=20*pq.ms, x_dim=8)
    >>> gpfa.fit(data)
    >>> results = gpfa.transform(data, returned_data=['latent_variable_orth',
    ...                                               'latent_variable'])
    >>> latent_variable_orth = results['latent_variable_orth']
    >>> latent_variable = results['latent_variable']

    or simply

    >>> results = GPFA(bin_size=20*pq.ms, x_dim=8).fit_transform(data,
    ...                returned_data=['latent_variable_orth',
    ...                               'latent_variable'])
    """

    @deprecated_alias(binsize='bin_size')
    def __init__(self, bin_size=20 * pq.ms, x_dim=3, min_var_frac=0.01,
                 tau_init=100.0 * pq.ms, eps_init=1.0E-3, em_tol=1.0E-8,
                 em_max_iters=500, freq_ll=5, verbose=False):
        self.bin_size = bin_size
        self.x_dim = x_dim
        self.min_var_frac = min_var_frac
        self.tau_init = tau_init
        self.eps_init = eps_init
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names = (
            'latent_variable_orth',
            'latent_variable',
            'Vsm',
            'VsmGP',
            'y')
        self.verbose = verbose

        if not isinstance(self.bin_size, pq.Quantity):
            raise ValueError("'bin_size' must be of type pq.Quantity")
        if not isinstance(self.tau_init, pq.Quantity):
            raise ValueError("'tau_init' must be of type pq.Quantity")

        # will be updated later
        self.params_estimated = dict()
        self.fit_info = dict()
        self.transform_info = dict()

    @property
    def binsize(self):
        warnings.warn("'binsize' is deprecated; use 'bin_size'")
        return self.bin_size

    def fit(self, spiketrains):
        """
        Fit the model with the given training data.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be fit to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choices of `l`, `k`, and `n`.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `spiketrains` is an empty list.

            If `spiketrains[0][0]` is not a `neo.SpikeTrain`.

            If covariance matrix of input spike data is rank deficient.
        """
        self._check_training_data(spiketrains)
        seqs_train = self._format_training_data(spiketrains)
        # Check if training data covariance is full rank
        y_all = np.hstack(seqs_train['y'])
        y_dim = y_all.shape[0]

        if np.linalg.matrix_rank(np.cov(y_all)) < y_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if self.verbose:
            print('Number of training trials: {}'.format(len(seqs_train)))
            print('Latent space dimensionality: {}'.format(self.x_dim))
            print('Observation dimensionality: {}'.format(
                self.has_spikes_bool.sum()))

        # The following does the heavy lifting.
        self.params_estimated, self.fit_info = gpfa_core.fit(
            seqs_train=seqs_train,
            x_dim=self.x_dim,
            bin_width=self.bin_size.rescale('ms').magnitude,
            min_var_frac=self.min_var_frac,
            em_max_iters=self.em_max_iters,
            em_tol=self.em_tol,
            tau_init=self.tau_init.rescale('ms').magnitude,
            eps_init=self.eps_init,
            freq_ll=self.freq_ll,
            verbose=self.verbose)

        return self

    @staticmethod
    def _check_training_data(spiketrains):
        if len(spiketrains) == 0:
            raise ValueError("Input spiketrains cannot be empty")
        if not isinstance(spiketrains[0][0], neo.SpikeTrain):
            raise ValueError("structure of the spiketrains is not correct: "
                             "0-axis should be trials, 1-axis neo.SpikeTrain"
                             "and 2-axis spike times")

    def _format_training_data(self, spiketrains):
        seqs = gpfa_util.get_seqs(spiketrains, self.bin_size)
        # Remove inactive units based on training set
        self.has_spikes_bool = np.hstack(seqs['y']).any(axis=1)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        return seqs

    def transform(self, spiketrains, returned_data=['latent_variable_orth']):
        """
        Obtain trajectories of neural activity in a low-dimensional latent
        variable space by inferring the posterior mean of the obtained GPFA
        model and applying an orthonormalization on the latent variable space.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be transformed to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choices of `l`, `k`, and `n`.
        returned_data : list of str
            The dimensionality reduction transform generates the following
            resultant data:

               'latent_variable_orth': orthonormalized posterior mean of latent
               variable

               'latent_variable': posterior mean of latent variable before
               orthonormalization

               'Vsm': posterior covariance between latent variables

               'VsmGP': posterior covariance over time for each latent variable

               'y': neural data used to estimate the GPFA model parameters

            `returned_data` specifies the keys by which the data dict is
            returned.

            Default is ['latent_variable_orth'].

        Returns
        -------
        np.ndarray or dict
            When the length of `returned_data` is one, a single np.ndarray,
            containing the requested data (the first entry in `returned_data`
            keys list), is returned. Otherwise, a dict of multiple np.ndarrays
            with the keys identical to the data names in `returned_data` is
            returned.

            N-th entry of each np.ndarray is a np.ndarray of the following
            shape, specific to each data type, containing the corresponding
            data for the n-th trial:

                `latent_variable_orth`: (#latent_vars, #bins) np.ndarray

                `latent_variable`:  (#latent_vars, #bins) np.ndarray

                `y`:  (#units, #bins) np.ndarray

                `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

                `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

            Note that the num. of bins (#bins) can vary across trials,
            reflecting the trial durations in the given `spiketrains` data.

        Raises
        ------
        ValueError
            If the number of neurons in `spiketrains` is different from that
            in the training spiketrain data.

            If `returned_data` contains keys different from the ones in
            `self.valid_data_names`.
        """
        if len(spiketrains[0]) != len(self.has_spikes_bool):
            raise ValueError("'spiketrains' must contain the same number of "
                             "neurons as the training spiketrain data")
        invalid_keys = set(returned_data).difference(self.valid_data_names)
        if len(invalid_keys) > 0:
            raise ValueError("'returned_data' can only have the following "
                             "entries: {}".format(self.valid_data_names))
        seqs = gpfa_util.get_seqs(spiketrains, self.bin_size)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        seqs, ll = gpfa_core.exact_inference_with_ll(seqs,
                                                     self.params_estimated,
                                                     get_ll=True)
        self.transform_info['log_likelihood'] = ll
        self.transform_info['num_bins'] = seqs['T']
        Corth, seqs = gpfa_core.orthonormalize(self.params_estimated, seqs)
        self.transform_info['Corth'] = Corth
        if len(returned_data) == 1:
            return seqs[returned_data[0]]
        return {x: seqs[x] for x in returned_data}

    def fit_transform(self, spiketrains, returned_data=[
                      'latent_variable_orth']):
        """
        Fit the model with `spiketrains` data and apply the dimensionality
        reduction on `spiketrains`.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Refer to the :func:`GPFA.fit` docstring.

        returned_data : list of str
            Refer to the :func:`GPFA.transform` docstring.

        Returns
        -------
        np.ndarray or dict
            Refer to the :func:`GPFA.transform` docstring.

        Raises
        ------
        ValueError
             Refer to :func:`GPFA.fit` and :func:`GPFA.transform`.

        See Also
        --------
        GPFA.fit : fit the model with `spiketrains`
        GPFA.transform : transform `spiketrains` into trajectories

        """
        self.fit(spiketrains)
        return self.transform(spiketrains, returned_data=returned_data)

    def score(self, spiketrains):
        """
        Returns the log-likelihood of the given data under the fitted model

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be scored.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choice of `l`, `k`, and `n`.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the given spiketrains under the fitted model.
        """
        self.transform(spiketrains)
        return self.transform_info['log_likelihood']
