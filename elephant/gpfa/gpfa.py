"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
 [1] for neural trajectory visualization of parallel spike trains.

The input consists of a set of trials (Y), each containing a list of spike
trains (N neurons). The output is the projection (X) of the data in space
of pre-chosen dimension x_dim < N.

Under the assumption of a linear relation between the latent
variable X and the actual data Y in addition to a noise term (i.e.,
Y = C * X + d + Gauss(0,R)), the projection corresponds to the conditional
probability E[X|Y].

A Gaussian process (X) of dimension x_dim < N is adopted to extract smooth
neural trajectories. The parameters (C, d, R) are estimated from the data using
factor analysis technique. GPFA is simply a set of Factor Analyzers (FA),
linked together in the low dimensional space by a Gaussian Process (GP).

Internally, the analysis consists of the following steps:

0) bin the data to get a sequence of N dimensional vectors for each time
   bin (cf., `gpfa_util.get_seq`), and choose the reduced dimension x_dim

1) expectation maximization for the parameters C, d, R and the time-scale of
   the gaussian process, using all the trials provided as input (cf.,
   `gpfa_core.em`)

2) projection of single trial in the low dimensional space (cf.,
   `gpfa_core.exact_inference_with_ll`)

3) orthonormalization of the matrix C and the corresponding subspace:
   (cf., `gpfa_util.orthogonalize`)


There are two principle scenarios of using the GPFA analysis. In the first
scenario, only one single dataset is available. The parameters that describe
the transformation are first extracted from the data, and the orthonormal basis
is constructed. Then the same data is projected into this basis. This analysis
is performed using the `gpfa()` function.

In the second scenario, both a training and a test data set is available. Here,
the parameters are estimated from the training data. In a second step the test
data is projected into the non-orthonormal space obtained from the training
data, and then orthonormalized. From this scenario, it is possible to perform a
cross-validation between training and test data sets. This analysis is
performed by exectuing first `extract_trajectory()` on the training data,
followed by `postprocess()` on the training and test datasets.


References:

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


def postprocess(params_est, seqs):
    """
    Orthonormalization and other cleanup.

    Parameters
    ----------

    params_est : dict
        First return value of extract_trajectory() on the training data set.
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
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    seqs: np.recarray
        Contains the embedding of the training data into the latent variable
        space.
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
            * T: int
                number of timesteps
            * y: ndarray of shape (#units, #bins)
                neural data
            * xsm: ndarray of shape (#latent_vars, #bins)
                posterior mean of latent variables at each time bin
            * Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                posterior covariance between latent variables at each
                timepoint
            * VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                posterior covariance over time for each latent variable

    Returns
    -------

    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs_train : np.recarray
        Training data structure that contains the new field `xorth`,
        the orthonormalized neural trajectories.
    seqs_test : np.recarray
        Test data structure that contains orthonormalized neural
        trajectories in `xorth`, obtained using `params_est`.
        When no test dataset is given, None is returned.


    Raises
    ------
    ValueError
        If `fit_info['kernSDList'] != kern_sd`.

    """
    C = params_est['C']
    X = np.hstack(seqs['xsm'])
    Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
    seqs = gpfa_util.segment_by_trial(seqs, Xorth, 'xorth')

    params_est['Corth'] = Corth

    return params_est, seqs


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

    fit_info: dict
        Information of the fitting process and the parameters used there:
            * iteration_time: A list containing the runtime for each iteration
                step in the EM algorithm.
            * log_likelihood: float, maximized likelihood obtained in the
                E-step of the EM algorithm.
            * bin_size: int, Width of the bins.
            * cvf: int, number for cross-validation folding
                Default is 0 (no cross-validation).
            * has_spikes_bool: Indicates if a neuron has any spikes across
                trials.
            * method: str, Method name.

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
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that data[l][n] is the
            Spiketrain of neuron n in trial l. Note that the number and order
            of Spiketrains objects per trial must be fixed such that data[l][n]
            and data[k][n] refer to the same spike generator for any choice of
            l,k and n.
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
        seqs = gpfa_util.get_seq(data, self.bin_size)
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
            seq_train=seqs_train,
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
        Apply dimensionality reduction to the given data with the estimated
        parameters

        Parameters
        ----------
        data : list of list of Spiketrain objects
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that data[l][n] is the
            Spiketrain of neuron n in trial l. Note that the number and order
            of Spiketrains objects per trial must be fixed such that data[l][n]
            and data[k][n] refer to the same spike generator for any choice of
            l,k and n.

        Returns
        -------
        seqs: numpy.recarray
            Contains the embedding of the data into the latent variable space.
            Data structure, whose n-th entry (corresponding to the n-th
            experimental trial) has fields
                * T: int
                    number of timesteps
                * y: ndarray of shape (#units, #bins)
                    neural data
                * xsm: ndarray of shape (#latent_vars, #bins)
                    posterior mean of latent variables at each time bin
                * Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                    posterior covariance between latent variables at each
                    timepoint
                * VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                    posterior covariance over time for each latent variable
                * xorth: ndarray of shape (#latent_vars, #bins)
                    trajectory in the orthonormalized space

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
        seqs = gpfa_util.get_seq(data, self.bin_size)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        return self._transform(seqs, returned_data)

    def _transform(self, seqs, returned_data):
        seqs, ll = gpfa_core.exact_inference_with_ll(seqs, self.params_est, get_ll=True)
        self.fit_info['log_likelihood'] = ll
        self.T = seqs['T']
        self.params_est, seqs = postprocess(self.params_est, seqs)
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
