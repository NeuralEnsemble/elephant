"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
 [1] for neural trajectory visualization of parallel spike trains. GPFA applies
 factor analysis (FA) time-binned spike count data to reduce the dimensionality
 and at the same time smoothes the resulting low-dimensional trajectories by
 fitting a Gaussian process (GP) model to them.

The input consists of a set of trials (Y), each containing a list of spike
trains (N neurons). The output is the projection (X) of the data in space
of pre-chosen dimension x_dim < N.

Under the assumption of a linear relation between the latent variable X and the
actual data Y in addition to a noise term (i.e.,
:math:`Y = C * X + d + Gauss(0,R)`), the projection corresponds to the
conditional probability E[X|Y].

A Gaussian process (X) of dimension x_dim < N is adopted to extract smooth
neural trajectories. The parameters (C, d, R) are estimated from the data using
factor analysis technique.

Internally, the analysis consists of the following steps:

0) bin the data to get a sequence of N dimensional vectors of spike counts for
   each time bin, and choose the reduced dimension x_dim

1) expectation maximization for the parameters C, d, R and the time-scale of
   the Gaussian process, using all the trials provided as input (cf.,
   `gpfa_core.em()`)

2) projection of single trials in the low dimensional space (cf.,
   `gpfa_core.exact_inference_with_ll()`)

3) orthonormalization of the matrix C and the corresponding subspace:
   (cf., `gpfa_core.orthonormalize()`)


There are two principle scenarios of using the GPFA analysis, both of which can
be performed in an instance of the GPFA() class.

In the first scenario, only one single dataset is used to fit the model and to
extract the neural trajectories. The parameters that describe the
transformation are first extracted from the data using the fit() method of the
GPFA class. Then the same data is projected into the orthonormal basis using
the method transform(). The fit_transform() method can be used to perform these
two steps at once.

In the second scenario, a single dataset is split into training and test
datasets. Here, the parameters are estimated from the training data. Then the
test data is projected into the low-dimensional space previously obtained from
the training data. This analysis is performed by executing first the fit()
method on the training data, followed by the transform() method on the test
dataset.

The GPFA class is compatible to the cross-validation functions of
`sklearn.model_selection`, such that users can perform cross-validation to
search for a set of parameters yielding best performance using these functions.

References
----------
The code was ported from the MATLAB code based on Byron Yu's implementation.
The original MATLAB code is available at Byron Yu's website:
https://users.ece.cmu.edu/~byronyu/software.shtml

[1] Yu MB, Cunningham JP, Santhanam G, Ryu SI, Shenoy K V, Sahani M (2009)
Gaussian-process factor analysis for low-dimensional single-trial analysis of
neural population activity. J Neurophysiol 102:614-635.

:copyright: Copyright 2015-2019 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""


import numpy as np
import neo
import quantities as pq
import sklearn

from elephant.gpfa import gpfa_core, gpfa_util


class GPFA(sklearn.base.BaseEstimator):
    """
    Prepares data and calls functions for extracting neural trajectories in the
    orthonormal space.

    Parameters
    ----------
    bin_size : quantities.Quantity, optional
        Width of each time bin.
        Default is 20 ms.
    x_dim : int, optional
        State dimensionality.
        Default is 3.
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        (default: 0.01) (See Martin & McDonald, Psychometrika, Dec 1975.)
    tau_init : quantities.Quantity, optional
        GP timescale initialization in msec (default: 100 ms)
    eps_init : float, optional
        GP noise variance initialization (default: 1e-3)
    tol : float, optional
          stopping criterion for EM (default: 1e-8)
    em_max_iters : int, optional
        Number of EM iterations to run (default: 500).

    Raises
    ------
    ValueError
        If `data` is an empty list.
        If `bin_size` if not a `pq.Quantity`.
        If `tau_init` if not a `pq.Quantity`.
        If `data[0][1][0]` is not a `neo.SpikeTrain`.

    Attributes
    ----------
    params_est: dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
                Currently, only 'rbf' is supported.
            gamma: ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: ndarray of shape (1, #latent_vars)
                GP noise variances
            d: ndarray of shape (#units, 1)
                observation mean
            C: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            Corth: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the orthonormal
                latent variable space
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    fit_info : dict
        Information of the fitting process and the parameters used there:
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
        log_likelihood : float
            maximized likelihood obtained in the E-step of the EM algorithm.
        bin_size: int
            Width of the bins.
        has_spikes_bool : ndarray
            Indicates if a neuron has any spikes across trials.
        method: str
            Method name.

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    Poisson spike train generators recorded in 50 trials with randomized
    rates up to 100 Hz.
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa import GPFA
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> data = []
    >>> for trial in range(50):
    >>>     n_channels = 20
    >>>     firing_rates = np.random.randint(low=1, high=100,
    >>>         size=n_channels) * pq.Hz
    >>>     spike_times = [homogeneous_poisson_process(rate=rate)
    >>>         for rate in firing_rates]
    >>>     data.append((trial, spike_times))
    >>> gpfa = GPFA(bin_size=20*pq.ms, x_dim=8)
    >>> gpfa.fit(data)
    >>> seqs = gpfa.transform(data)
    or simply
    >>> seqs = GPFA(bin_size=20*pq.ms, x_dim=8).fit_transform(data)
    """

    def __init__(self, bin_size=20*pq.ms, x_dim=3, min_var_frac=0.01,
                 tau_init=100.0*pq.ms, eps_init=1.0E-3, em_tol=1.0E-8,
                 em_max_iters=500, freq_ll=5, valid_data_names=('xorth', 'xsm', 'Vsm', 'VsmGP', 'y')):
        self.bin_size = bin_size
        self.x_dim = x_dim
        self.min_var_frac = min_var_frac
        self.tau_init = tau_init
        self.eps_init = eps_init
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names = valid_data_names

        if not isinstance(self.bin_size, pq.Quantity):
            raise ValueError("'bin_size' must be of type pq.Quantity")
        if not isinstance(self.tau_init, pq.Quantity):
            raise ValueError("'tau_init' must be of type pq.Quantity")

    def fit(self, data, verbose=False):
        """
        Fit the model with the given training data.

        Parameters
        ----------
        data : list of list of Spiketrain objects
            Spike train data to be transformed to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that data[l][n] is the
            Spiketrain of neuron n in trial l. Note that the number and order
            of Spiketrains objects per trial must be fixed such that data[l][n]
            and data[k][n] refer to spike trains of the same neuron for any
            choice of l,k and n.
        verbose : bool, optional
            specifies whether to display status messages (default: False)

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `data` is an empty list.
            If `data[0][1][0]` is not a `neo.SpikeTrain`.
        """
        self._check_training_data(data)
        seqs_train = self._format_training_data(data)
        return self._fit(seqs_train, verbose)

    def _check_training_data(self, data):
        if len(data) == 0:
            raise ValueError("`data` cannot be empty")
        if not isinstance(data[0][0], neo.SpikeTrain):
            raise ValueError("structure of the data is not correct: 0-axis "
                             "should be trials, 1-axis neo spike trains "
                             "and 2-axis spike times")

    def _format_training_data(self, data):
        seqs = gpfa_util.get_seqs(data, self.bin_size)
        # Remove inactive units based on training set
        self.has_spikes_bool = (np.hstack(seqs['y']).mean(1) != 0)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        return seqs

    def _fit(self, seqs_train, verbose):
        # Check if training data covariance is full rank
        y_all = np.hstack(seqs_train['y'])
        y_dim = y_all.shape[0]

        if np.linalg.matrix_rank(np.cov(y_all)) < y_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if verbose:
            print('Number of training trials: {}'.format(len(seqs_train)))
            print('Latent space dimensionality: {}'.format(self.x_dim))
            print('Observation dimensionality: {}'.format(self.has_spikes_bool.sum()))

        # The following does the heavy lifting.
        self.params_est, self.fit_info = gpfa_core.fit(
            seqs_train=seqs_train,
            x_dim=self.x_dim,
            bin_width=self.bin_size.rescale('ms').magnitude,
            min_var_frac=self.min_var_frac,
            em_max_iters=self.em_max_iters,
            em_tol=self.em_tol,
            tau_init = self.tau_init.rescale('ms').magnitude,
            eps_init = self.eps_init,
            freq_ll = self.freq_ll,
            verbose=verbose)

        self.fit_info['has_spikes_bool'] = self.has_spikes_bool
        self.fit_info['min_var_frac'] = self.min_var_frac
        self.fit_info['bin_size'] = self.bin_size

        return self

    def transform(self, data, returned_data=['xorth']):
        """
        Obtain trajectories of neural activity in a low-dimensional latent
        variable space by inferring the posterior mean of the obtained GPFA
        model and applying an orthonormalization of the latent variable space

        Parameters
        ----------
        data : list of list of Spiketrain objects
            Spike train data to be transformed to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that data[l][n] is the
            Spiketrain of neuron n in trial l. Note that the number and order
            of Spiketrains objects per trial must be fixed such that data[l][n]
            and data[k][n] refer to spike trains of the same neuron for any
            choice of l,k and n.
        returned_data : a list of str
            The dimensionality reduction transform generates the following
            resultant data:
               'xorth': orthonormalized posterior mean of latent variable
               'xsm': posterior mean of latent variable before
               orthonormalization
               'Vsm': posterior covariance between latent variables
               'VsmGP': posterior covariance over time for each latent variable
               'y': neural data used to estimate the GPFA model parameters
            `returned_data` specifies which data are to be returned.
            Default is ['xorth'].

        Returns
        -------
        output : numpy.ndarray or dict
            When the length of `returned_data` is one, a single ndarray
            containing the requested data is returned. Otherwise, a dict of
            multiple ndarrays with the keys identical to the data names in
            `returned_data` is returned.
            N-th entry of each ndarray is a ndarray of the following shape,
            specific to each data type, containing the corresponding data for
            the n-th trial:
                `xorth`: ndarray of shape (#latent_vars, #bins)
                `xsm`: ndarray of shape (#latent_vars, #bins)
                `y`: ndarray of shape (#units, #bins)
                `Vsm`: ndarray of shape (#latent_vars, #latent_vars, #bins)
                `VsmGP`: ndarray of shape (#bins, #bins, #latent_vars)
            Note that #bins can vary across trials reflecting the trial
            durations in the provided data.

        Raises
        ------
        ValueError
            If the number of neurons in `data` is different from that in the
            training data.
        """
        if len(data[0]) != len(self.has_spikes_bool):
            raise ValueError("`data` must contain the same number of neurons as the training data")
        for data_name in returned_data:
            if data_name not in self.valid_data_names:
                raise ValueError("`returned_data` can only have the following entries: {}".format(self.valid_data_names))
        seqs = gpfa_util.get_seqs(data, self.bin_size)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        return self._transform(seqs, returned_data)

    def _transform(self, seqs, returned_data):
        seqs, ll = gpfa_core.exact_inference_with_ll(seqs, self.params_est, get_ll=True)
        self.fit_info['log_likelihood'] = ll
        self.T = seqs['T']
        self.params_est, seqs = gpfa_core.orthonormalize(self.params_est, seqs)
        if len(returned_data) == 1:
            return seqs[returned_data[0]]
        else:
            return {x: seqs[x] for x in returned_data}

    def fit_transform(self, data, returned_data=['xorth'], verbose=False):
        """
        Fit the model with the given data and apply dimensionality reduction to
        the same data with the estimated parameters.
        Refer to documentation of GPFA.fit() and GPFA.transform() for more
        details.
        """
        self._check_training_data(data)
        seqs_train = self._format_training_data(data)
        self._fit(seqs_train, verbose)
        return self._transform(seqs_train, returned_data)

    def score(self, data):
       self.transform(data)
       return self.fit_info['log_likelihood']
