"""
CAD :cite:`cad-Russo2017_e19428` is a method aimed to capture structures of
higher-order correlation in massively parallel spike trains. In particular, it
is able to extract patterns of spikes with arbitrary configuration of time lags
(time interval between spikes in a pattern), and at multiple time scales,
e.g. from synchronous patterns to firing rate co-modulations.

CAD consists of a statistical parametric testing done on the level of pairs
of neurons, followed by an agglomerative recursive algorithm, in order to
detect and test statistically precise repetitions of spikes in the data.
In particular, pairs of neurons are tested for significance under the null
hypothesis of independence, and then the significant pairs are agglomerated
into higher order patterns.

Given a list of discretized (binned) spike trains by a given temporal
scale (bin_size), assumed to be recorded in parallel, the CAD analysis can be
applied as demonstrated in this short toy example of 5 parallel spike trains
that exhibit fully synchronous events of order 5.

.. autosummary::
    :toctree: _toctree/cell_assembly_detection

    cell_assembly_detection


Visualization
-------------
Visualization of CAD method is covered in Viziphant
:func:`viziphant.patterns.plot_patterns`


See Also
--------
elephant.spade.spade : advanced synchronous patterns detection

Examples
--------
>>> import quantities as pq
>>> import numpy as np
>>> from elephant.cell_assembly_detection import cell_assembly_detection
>>> from elephant.spike_train_generation import compound_poisson_process
>>> from elephant.conversion import BinnedSpikeTrain

Generate correlated data and bin it with a bin_size of 10ms.

>>> np.random.seed(30)
>>> spiketrains = compound_poisson_process(rate=15*pq.Hz,
...     amplitude_distribution=[0, 0.95, 0, 0, 0, 0, 0.05], t_stop=5*pq.s)
>>> bst = BinnedSpikeTrain(spiketrains, bin_size=10 * pq.ms)
>>> bst.rescale('ms')

Call of the method.

>>> patterns = cell_assembly_detection(bst, max_lag=2)
>>> patterns[0]
{'neurons': [0, 2],
 'lags': array([0.]) * ms,
 'pvalue': [5.3848138041122556e-05],
 'times': array([  90.,  160.,  170.,  550.,  790.,  910.,  930., 1420., 1470.,
        1480., 1650., 2030., 2220., 2570., 3130., 3430., 3480., 3610.,
        3800., 3830., 3930., 4080., 4560., 4600., 4670.]) * ms,
 'signature': [[1, 83], [2, 25]]}

Refer to the Viziphant documentation regarding the visualization of this
example.

"""

from __future__ import division, print_function, unicode_literals

import copy
import math
import time
import warnings

import numpy as np
from scipy.stats import f

import elephant.conversion as conv
from elephant.utils import deprecated_alias

__all__ = [
    "cell_assembly_detection"
]


@deprecated_alias(data='binned_spiketrain', maxlag='max_lag',
                  min_occ='min_occurrences',
                  same_config_cut='same_configuration_pruning')
def cell_assembly_detection(binned_spiketrain, max_lag, reference_lag=2,
                            alpha=0.05, min_occurrences=1, size_chunks=100,
                            max_spikes=np.inf, significance_pruning=True,
                            subgroup_pruning=True,
                            same_configuration_pruning=False,
                            bool_times_format=None, verbose=False):
    """
    Perform the CAD analysis :cite:`cad-Russo2017_e19428` for the binned
    (discretized) spike trains given in the input. The method looks for
    candidate significant patterns with lags (number of bins between successive
    spikes in the pattern) ranging from `-max_lag` to `max_lag` (the second
    parameter of the function). Thus, between two successive spikes in the
    pattern there can be at most `max_lag`*`bin_size` units of time.

    The method agglomerates pairs of units (or a unit and a preexisting
    assembly), tests their significance by a statistical test
    and stops when the detected assemblies reach their maximal dimension
    (parameter `max_spikes`).

    At every agglomeration size step (e.g. from triplets to quadruplets), the
    method filters patterns having the same neurons involved, and keeps only
    the most significant one. This pruning is optional and the choice is
    identified by the parameter 'significance_pruning'.
    Assemblies already included in a bigger assembly are eliminated in a final
    pruning step. Also this pruning is optional, and the choice is identified
    by the parameter `subgroup_pruning`.

    Parameters
    ----------
    binned_spiketrain : elephant.conversion.BinnedSpikeTrain
        Binned spike trains containing data to be analyzed.
    max_lag : int
        Maximal lag to be tested. For a binning dimension of bin_size the
        method will test all pairs configurations with a time
        shift between '-max_lag' and 'max_lag'.
    reference_lag : int, optional
        Reference lag (in bins) for the non-stationarity correction in the
        statistical test.
        Default: 2
    alpha : float, optional
        Significance level for the statistical test.
        Default: 0.05
    min_occurrences : int, optional
        Minimal number of occurrences required for an assembly
        (all assemblies, even if significant, with fewer occurrences
        than min_occurrences are discarded).
        Default: 0
    size_chunks : int, optional
        Size (in bins) of chunks in which the spike trains are divided
        to compute the variance (to reduce non stationarity effects
        on variance estimation).
        Default: 100
    max_spikes : int, optional
        Maximal assembly order (the algorithm will return assemblies
        composed of maximum `max_spikes` elements).
        Default: `np.inf`
    significance_pruning : bool, optional
        If True, the method performs significance pruning among
        the detected assemblies.
        Default: True
    subgroup_pruning : bool, optional
        If True, the method performs subgroup pruning among
        the detected assemblies.
        Default: True
    same_configuration_pruning : bool, optional
        If True, performs pruning (not present in the original code and more
        efficient), not testing assemblies already formed
        if they appear in the very same configuration.
        Default: False
    bool_times_format : bool, optional
        .. deprecated:: 0.10.0
        Has no effect, the returning 'times' are always a quantity array
        specifying the pattern spike times.
        Default: None
    verbose : bool, optional
        Regulates the number of prints given by the method. If true all prints
        are given, otherwise the method does give any prints.
        Default: False

    Returns
    -------
    assembly : list of dict
        Contains the assemblies detected for the bin size chosen. Each
        assembly is a dictionary with attributes:

        'neurons' : list
            Vector of units taking part to the assembly (unit order correspond
            to the agglomeration order).
        'lag' : pq.Quantity
            Vector of time lags.
            `lag[z]` is the activation delay between `neurons[1]` and
            `neurons[z+1]`.
        'pvalue' : list
            Vector containing p-values.
            `pvalue[z]` is the p-value of the statistical test between
            performed adding `neurons[z+1]` to the `neurons[1:z]`.
        'times' : pq.Quantity
            Assembly activation times in the units of `binned_spiketrain`.
        'signature' : np.ndarray
            Array of two entries `(z,c)`. The first is the number of neurons
            participating in the assembly (size), and the second is number of
            assembly occurrences.

    Raises
    ------
    TypeError
        If `binned_spiketrain` is not an instance of
        `elephant.conversion.BinnedSpikeTrain`.
    ValueError
        If the parameters are out of bounds.

    Notes
    -----
    Alias: cad

    """
    initial_time = time.time()

    # check parameter input and raise errors if necessary
    _raise_errors(binned_spiketrain=binned_spiketrain,
                  max_lag=max_lag,
                  alpha=alpha,
                  min_occurrences=min_occurrences,
                  size_chunks=size_chunks,
                  max_spikes=max_spikes)

    if bool_times_format is not None:
        warnings.warn("'bool_times_format' is deprecated and has no effect; "
                      "the returning 'times' are always a quantity array "
                      "specifying the pattern spike times. Set this parameter "
                      "to None.", DeprecationWarning)

    bin_size = binned_spiketrain.bin_size
    t_start = binned_spiketrain.t_start

    # transform the binned spiketrain into array
    binned_spiketrain = binned_spiketrain.to_array()

    # zero order
    n_neurons = len(binned_spiketrain)

    # initialize empty assembly

    assembly_in = [{'neurons': None,
                    'lags': None,
                    'pvalue': None,
                    'times': None,
                    'signature': None} for _ in range(n_neurons)]

    # initializing the dictionaries
    if verbose:
        print('Initializing the dictionaries...')
    for w1 in range(n_neurons):
        assembly_in[w1]['neurons'] = [w1]
        assembly_in[w1]['lags'] = []
        assembly_in[w1]['pvalue'] = []
        assembly_in[w1]['times'] = binned_spiketrain[w1]
        assembly_in[w1]['signature'] = [[1, sum(binned_spiketrain[w1])]]

    # first order = test over pairs

    # denominator of the Bonferroni correction
    # divide alpha by the number of tests performed in the first
    # pairwise testing loop
    number_test_performed = n_neurons * (n_neurons - 1) * (2 * max_lag + 1)
    alpha = alpha * 2 / float(number_test_performed)
    if verbose:
        print('actual significance_level', alpha)

    # sign_pairs_matrix is the matrix with entry as 1 for the significant pairs
    sign_pairs_matrix = np.zeros((n_neurons, n_neurons), dtype=np.int)
    assembly = []
    if verbose:
        print('Testing on pairs...')

    # nns: count of the existing assemblies
    nns = 0

    # initialize the structure existing_patterns, storing the patterns
    # determined by neurons and lags:
    # if the pattern is already existing, don't do the test
    existing_patterns = []

    # for loop for the pairwise testing
    for w1 in range(n_neurons - 1):
        for w2 in range(w1 + 1, n_neurons):
            spiketrain2 = binned_spiketrain[w2]
            n2 = w2
            assembly_flag = 0

            # call of the function that does the pairwise testing
            call_tp = _test_pair(
                ensemble=assembly_in[w1],
                spiketrain2=spiketrain2,
                n2=n2,
                max_lag=max_lag,
                size_chunks=size_chunks,
                reference_lag=reference_lag,
                existing_patterns=existing_patterns,
                same_configuration_pruning=same_configuration_pruning)
            if same_configuration_pruning:
                assem_tp = call_tp[0]
            else:
                assem_tp = call_tp

            # if the assembly given in output is significant and the number
            # of occurrences is higher than the minimum requested number
            if assem_tp['pvalue'][-1] < alpha and \
                    assem_tp['signature'][-1][1] > min_occurrences:
                # save the assembly in the output
                assembly.append(assem_tp)
                sign_pairs_matrix[w1][w2] = 1
                assembly_flag = 1  # flag : it is indeed an assembly
                # put the item_candidate into the existing_patterns list
                if same_configuration_pruning:
                    item_candidate = call_tp[1]
                    if not existing_patterns:
                        existing_patterns = [item_candidate]
                    else:
                        existing_patterns.append(item_candidate)
            if assembly_flag:
                nns += 1  # count of the existing assemblies

    # making sign_pairs_matrix symmetric
    sign_pairs_matrix = sign_pairs_matrix + sign_pairs_matrix.T
    sign_pairs_matrix[sign_pairs_matrix == 2] = 1
    # print(sign_pairs_matrix)

    # second order and more: increase the assembly size by adding a new unit

    # the algorithm will return assemblies composed by
    # maximum max_spikes elements
    if verbose:
        print('\nTesting on higher order assemblies...\n')

    # keep the count of the current size of the assembly
    current_size_agglomeration = 2

    # number of groups previously found
    n_as = len(assembly)

    # w2_to_test_v : contains the elements to test with the elements that are
    # in the assembly in input
    w2_to_test_v = np.zeros(n_neurons)

    # testing for higher order assemblies

    w1 = 0

    while w1 < n_as:

        w1_elements = assembly[w1]['neurons']

        # Add only neurons that have significant first order
        # co-occurrences with members of the assembly
        # Find indices and values of nonzero elements

        for i in range(len(w1_elements)):
            w2_to_test_v += sign_pairs_matrix[w1_elements[i]]

        # w2_to_test_p : vector with the index of nonzero elements
        w2_to_test_p = np.nonzero(w2_to_test_v)[0]

        # list with the elements to test
        # that are not already in the assembly
        w2_to_test = [item for item in w2_to_test_p
                      if item not in w1_elements]
        pop_flag = 0

        # check that there are candidate neurons for agglomeration
        if w2_to_test:

            # bonferroni correction only for the tests actually performed
            alpha = alpha / float(len(w2_to_test) * n_as * (2 * max_lag + 1))

            # testing for the element in w2_to_test
            for ww2 in range(len(w2_to_test)):
                w2 = w2_to_test[ww2]
                spiketrain2 = binned_spiketrain[w2]
                assembly_flag = 0
                pop_flag = max(assembly_flag, 0)
                # testing for the assembly and the new neuron

                call_tp = _test_pair(
                    ensemble=assembly[w1],
                    spiketrain2=spiketrain2,
                    n2=w2,
                    max_lag=max_lag,
                    size_chunks=size_chunks,
                    reference_lag=reference_lag,
                    existing_patterns=existing_patterns,
                    same_configuration_pruning=same_configuration_pruning)
                if same_configuration_pruning:
                    assem_tp = call_tp[0]
                else:
                    assem_tp = call_tp

                # if it is significant and
                # the number of occurrences is sufficient and
                # the length of the assembly is less than the input limit
                if assem_tp['pvalue'][-1] < alpha and \
                        assem_tp['signature'][-1][1] > min_occurrences and \
                        assem_tp['signature'][-1][0] <= max_spikes:
                    # the assembly is saved in the output list of
                    # assemblies
                    assembly.append(assem_tp)
                    assembly_flag = 1

                    if len(assem_tp['neurons']) > current_size_agglomeration:
                        # up to the next agglomeration level
                        current_size_agglomeration += 1
                        # Pruning step 1
                        # between two assemblies with the same unit set
                        # arranged into different
                        # configurations, choose the most significant one
                        if significance_pruning is True and \
                                current_size_agglomeration > 3:
                            assembly, n_filtered_assemblies = \
                                _significance_pruning_step(
                                    pre_pruning_assembly=assembly)
                    if same_configuration_pruning:
                        item_candidate = call_tp[1]
                        existing_patterns.append(item_candidate)
                if assembly_flag:
                    # count one more assembly
                    nns += 1
                    n_as = len(assembly)
        # if at least once the assembly was agglomerated to a bigger one,
        # pop the smaller one
        if pop_flag:
            assembly.pop(w1)
        w1 = w1 + 1

    # Pruning step 1
    # between two assemblies with the same unit set arranged into different
    # configurations, choose the most significant one

    # Last call for pruning of last order agglomeration

    if significance_pruning:
        assembly = _significance_pruning_step(pre_pruning_assembly=assembly)[0]

    # Pruning step 2
    # Remove assemblies whom elements are already
    # ALL included in a bigger assembly
    if subgroup_pruning:
        assembly = _subgroup_pruning_step(pre_pruning_assembly=assembly)

    # Reformat of the activation times
    for pattern in assembly:
        times = np.where(pattern['times'] > 0)[0] * bin_size + t_start
        pattern['times'] = times
        pattern['lags'] = pattern['lags'] * bin_size
        pattern['signature'] = np.array(pattern['signature'], dtype=np.int32)

    # Give as output only the maximal groups
    if verbose:
        print('\nGiving outputs of the method...\n')
        print('final_assembly')
        for item in assembly:
            print(item['neurons'],
                  item['lags'],
                  item['signature'])

    # Time needed for the computation
    if verbose:
        print('\ntime', time.time() - initial_time)

    return assembly


def _chunking(binned_pair, size_chunks, max_lag, best_lag):
    """
    Chunking the object binned_pair into parts with the same bin length

    Parameters
    ----------
    binned_pair : np.array
        vector of the binned spike trains for the pair being analyzed
    size_chunks : int
        size of chunks desired
    max_lag : int
        max number of lags for the bin_size chosen
    best_lag : int
        lag with the higher number of coincidences

    Returns
    -------
    chunked : list
        list with the object binned_pair cut in size_chunks parts
    n_chunks : int
        number of chunks
    """

    length = len(binned_pair[0], )

    # number of chunks
    n_chunks = math.ceil((length - max_lag) / size_chunks)

    # new chunk size, this is to have all chunks of roughly the same size
    size_chunks = math.floor((length - max_lag) / n_chunks)

    n_chunks = np.int(n_chunks)
    size_chunks = np.int(size_chunks)

    chunked = [[[], []] for _ in range(n_chunks)]

    # cut the time series according to best_lag

    binned_pair_cut = np.array([np.zeros(length - max_lag, dtype=np.int),
                                np.zeros(length - max_lag, dtype=np.int)])

    # choose which entries to consider according to the best lag chosen
    if best_lag == 0:
        binned_pair_cut[0] = binned_pair[0][0:length - max_lag]
        binned_pair_cut[1] = binned_pair[1][0:length - max_lag]
    elif best_lag > 0:
        binned_pair_cut[0] = binned_pair[0][0:length - max_lag]
        binned_pair_cut[1] = binned_pair[1][
                             best_lag:length - max_lag + best_lag]
    else:
        binned_pair_cut[0] = binned_pair[0][
                             -best_lag:length - max_lag - best_lag]
        binned_pair_cut[1] = binned_pair[1][0:length - max_lag]

    # put the cut data into the chunked object
    for iii in range(n_chunks - 1):
        chunked[iii][0] = binned_pair_cut[0][
                          size_chunks * iii:size_chunks * (iii + 1)]
        chunked[iii][1] = binned_pair_cut[1][
                          size_chunks * iii:size_chunks * (iii + 1)]

    # last chunk can be of slightly different size
    chunked[n_chunks - 1][0] = binned_pair_cut[0][
                               size_chunks * (n_chunks - 1):length]
    chunked[n_chunks - 1][1] = binned_pair_cut[1][
                               size_chunks * (n_chunks - 1):length]

    return chunked, n_chunks


def _assert_same_pattern(item_candidate, existing_patterns, max_lag):
    """
    Tests if a particular pattern has already been tested and retrieved as
    significant.

    Parameters
    ----------
    item_candidate : list of list with two components
        in the first component there are the neurons involved in the assembly,
        in the second there are the correspondent lags
    existing_patterns : list
        list of the already significant patterns
    max_lag : int
        maximum lag to be tested

    Returns
    -------
        True if the pattern was already tested and retrieved as significant
        False if not
    """
    # unique representation of pattern in term of lags, maxlag and neurons
    # participating
    item_candidate = sorted(item_candidate[0] * 2 * max_lag +
                            item_candidate[1] + max_lag)
    if item_candidate in existing_patterns:
        return True
    else:
        return False


def _test_pair(ensemble, spiketrain2, n2, max_lag, size_chunks, reference_lag,
               existing_patterns, same_configuration_pruning):
    """
    Tests if two spike trains have repetitive patterns occurring more
    frequently than chance.

    Parameters
    ----------
    ensemble : dictionary
        structure with the previously formed assembly and its spike train
    spiketrain2 : list
        spike train of the new unit to be tested for significance
        (candidate to be a new assembly member)
    n2 : int
        new unit tested
    max_lag : int
        maximum lag to be tested
    size_chunks : int
        size (in bins) of chunks in which the spike trains is divided
        to compute the variance (to reduce non stationarity effects
        on variance estimation)
    reference_lag : int
        lag of reference; if zero or negative reference lag=-l
    existing_patterns : list
        list of the already significant patterns
    same_configuration_pruning : bool
        if True (not present in the original code and more
        efficient), does not test assemblies already formed
        if they appear in the very same configuration
        Default: False

    Returns
    -------
    assembly : dictionary
        assembly formed by the method (can be empty), with attributes:
        'elements' : vector of units taking part to the assembly
                     (unit order correspond to the agglomeration order)
        'lag' : vector of time lags (lag[z] is the activation delay between
                elements[1] and elements[z+1]
        'pvalue' : vector of pvalues. `pr[z]` is the p-value of the statistical
               test between performed adding elements[z+1] to the elements[1:z]
        'times' : assembly activation time. It reports how many times the
                 complete assembly activates in that bin.
                 time always refers to the activation of the first listed
                 assembly element (elements[1]), that doesn't necessarily
                 corresponds to the first unit firing.
        'signature' : array of two entries (z,c). The first is the number of
                        neurons participating in the assembly (size),
                        while the second is number of assembly occurrences.
    item_candidate : list of list with two components
        in the first component there are the neurons involved in the assembly,
        in the second there are the correspondent lags.

    """

    # list with the binned spike trains of the two neurons
    binned_pair = [ensemble['times'], spiketrain2]

    # For large bin_sizes, the binned spike counts may potentially fluctuate
    # around a high mean level and never fall below some minimum count
    # considerably larger than zero for the whole time series.
    # Entries up to this minimum count would contribute
    # to the coincidence count although they are completely
    # uninformative, so we subtract the minima.

    binned_pair = np.array([binned_pair[0] - min(binned_pair[0]),
                            binned_pair[1] - min(binned_pair[1])])

    ntp = len(binned_pair[0])  # trial length

    # Divide in parallel trials with 0/1 elements
    # max number of spikes in one bin for both neurons
    maxrate = np.int(max(max(binned_pair[0]), max(binned_pair[1])))

    # creation of the parallel processes, one for each rate up to maxrate
    # and computation of the coincidence count for both neurons
    par_processes = np.zeros((maxrate, 2, ntp), dtype=np.int)
    par_proc_expectation = np.zeros(maxrate, dtype=np.int)

    for i in range(maxrate):
        par_processes[i] = np.array(binned_pair > i, dtype=np.int)
        par_proc_expectation[i] = (np.sum(par_processes[i][0]) * np.sum(
            par_processes[i][1])) / float(ntp)

    # Decide which is the lag with most coincidences (l_ : best lag)
    # we are calculating the joint spike count of units A and B at lag l.
    # It is computed by counting the number
    # of times we have a spike in A and a corresponding spike in unit B
    # l times later for every lag,
    # we select the one corresponding to the highest count

    # structure with the coincidence counts for each lag
    fwd_coinc_count = np.array([0 for _ in range(max_lag + 1)])
    bwd_coinc_count = np.array([0 for _ in range(max_lag + 1)])

    for lag in range(max_lag + 1):
        time_fwd_cc = np.array([binned_pair[0][
                                0:len(binned_pair[0]) - max_lag],
                                binned_pair[1][
                                lag:len(binned_pair[1]) - max_lag + lag]])

        time_bwd_cc = np.array([binned_pair[0][
                                lag:len(binned_pair[0]) - max_lag + lag],
                                binned_pair[1][
                                0:len(binned_pair[1]) - max_lag]])

        # taking the minimum, place by place for the coincidences
        fwd_coinc_count[lag] = np.sum(np.minimum(time_fwd_cc[0],
                                                 time_fwd_cc[1]))
        bwd_coinc_count[lag] = np.sum(np.minimum(time_bwd_cc[0],
                                                 time_bwd_cc[1]))

    # choice of the best lag, taking into account the reference lag
    if reference_lag <= 0:
        # if the global maximum is in the forward process (A to B)
        if np.amax(fwd_coinc_count) > np.amax(bwd_coinc_count):
            # bwd_flag indicates whether we are in time_fwd_cc or time_bwd_cc
            fwd_flag = 1
            global_maximum_index = np.argmax(fwd_coinc_count)
        else:
            fwd_flag = 2
            global_maximum_index = np.argmax(bwd_coinc_count)
        best_lag = (fwd_flag == 1) * global_maximum_index - (
                fwd_flag == 2) * global_maximum_index
        max_coinc_count = max(np.amax(fwd_coinc_count),
                              np.amax(bwd_coinc_count))
    else:
        # reverse the ctAB_ object and not take into account the first entry
        bwd_coinc_count_rev = bwd_coinc_count[1:len(bwd_coinc_count)][::-1]
        hab_l = np.append(bwd_coinc_count_rev, fwd_coinc_count)
        lags = range(-max_lag, max_lag + 1)
        max_coinc_count = np.amax(hab_l)
        best_lag = lags[np.argmax(hab_l)]
        if best_lag < 0:
            lag_ref = best_lag + reference_lag
            coinc_count_ref = hab_l[lags.index(lag_ref)]
        else:
            lag_ref = best_lag - reference_lag
            coinc_count_ref = hab_l[lags.index(lag_ref)]

    # now check whether the pattern, with those neurons and that particular
    # configuration of lags,
    # is already in the list of the significant patterns
    # if it is, don't do the testing
    # if it is not, continue
    previous_neu = ensemble['neurons']
    pattern_candidate = copy.copy(previous_neu)
    pattern_candidate.append(n2)
    pattern_candidate = np.array(pattern_candidate)

    # add both the new lag and zero
    previous_lags = ensemble['lags']
    lags_candidate = copy.copy(previous_lags)
    lags_candidate.append(best_lag)

    lags_candidate[:0] = [0]
    pattern_candidate = list(pattern_candidate)
    lags_candidate = list(lags_candidate)
    item_candidate = [[pattern_candidate], [lags_candidate]]

    if same_configuration_pruning:
        if _assert_same_pattern(item_candidate=item_candidate,
                                existing_patterns=existing_patterns,
                                max_lag=max_lag):
            en_neurons = copy.copy(ensemble['neurons'])
            en_neurons.append(n2)
            en_lags = copy.copy(ensemble['lags'])
            en_lags.append(np.inf)
            en_pvalue = copy.copy(ensemble['pvalue'])
            en_pvalue.append(1)
            en_n_occ = copy.copy(ensemble['signature'])
            en_n_occ.append([0, 0])
            item_candidate = []
            assembly = {'neurons': en_neurons,
                        'lags': en_lags,
                        'pvalue': en_pvalue,
                        'times': [],
                        'signature': en_n_occ}
            return assembly, item_candidate
    else:
        # I go on with the testing

        pair_expectation = np.sum(par_proc_expectation)
        # case of no coincidences or limit for the F asimptotical
        # distribution (too few coincidences)
        if max_coinc_count == 0 or pair_expectation <= 5 or \
                pair_expectation >= (min(np.sum(binned_pair[0]),
                                         np.sum(binned_pair[1])) - 5):
            en_neurons = copy.copy(ensemble['neurons'])
            en_neurons.append(n2)
            en_lags = copy.copy(ensemble['lags'])
            en_lags.append(np.inf)
            en_pvalue = copy.copy(ensemble['pvalue'])
            en_pvalue.append(1)
            en_n_occ = copy.copy(ensemble['signature'])
            en_n_occ.append([0, 0])
            assembly = {'neurons': en_neurons,
                        'lags': en_lags,
                        'pvalue': en_pvalue,
                        'times': [],
                        'signature': en_n_occ}
            if same_configuration_pruning:
                item_candidate = []
                return assembly, item_candidate
            else:
                return assembly

        else:  # construct the activation series for binned_pair
            length = len(binned_pair[0])  # trial length
            activation_series = np.zeros(length)
            if reference_lag <= 0:
                if best_lag == 0:  # synchrony case
                    for i in range(maxrate):  # for all parallel processes
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        activation_series = \
                            np.add(activation_series,
                                   np.multiply(par_processes_a,
                                               par_processes_b))
                    coinc_count_matrix = np.array([[0, fwd_coinc_count[0]],
                                                   [bwd_coinc_count[2], 0]])
                    # matrix with #AB and #BA
                    # here we specifically choose
                    # 'l* = -2' for the synchrony case
                elif best_lag > 0:
                    for i in range(maxrate):
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        # multiplication between the two binned time series
                        # shifted by best_lag
                        activation_series[0:length - best_lag] = \
                            np.add(activation_series[0:length - best_lag],
                                   np.multiply(par_processes_a[
                                               0:length - best_lag],
                                               par_processes_b[
                                               best_lag:length]))
                    coinc_count_matrix = \
                        np.array([[0, fwd_coinc_count[global_maximum_index]],
                                  [bwd_coinc_count[global_maximum_index], 0]])
                else:
                    for i in range(maxrate):
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        activation_series[-best_lag:length] = \
                            np.add(activation_series[-best_lag:length],
                                   np.multiply(par_processes_a[
                                               -best_lag:length],
                                               par_processes_b[
                                               0:length + best_lag]))
                    coinc_count_matrix = \
                        np.array([[0, fwd_coinc_count[global_maximum_index]],
                                  [bwd_coinc_count[global_maximum_index], 0]])
            else:
                if best_lag == 0:
                    for i in range(maxrate):
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        activation_series = \
                            np.add(activation_series,
                                   np.multiply(par_processes_a,
                                               par_processes_b))
                elif best_lag > 0:
                    for i in range(maxrate):
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        activation_series[0:length - best_lag] = \
                            np.add(activation_series[0:length - best_lag],
                                   np.multiply(par_processes_a[
                                               0:length - best_lag],
                                               par_processes_b[
                                               best_lag:length]))
                else:
                    for i in range(maxrate):
                        par_processes_a = par_processes[i][0]
                        par_processes_b = par_processes[i][1]
                        activation_series[-best_lag:length] = \
                            np.add(activation_series[-best_lag:length],
                                   np.multiply(par_processes_a[
                                               -best_lag:length],
                                               par_processes_b[
                                               0:length + best_lag]))
                coinc_count_matrix = np.array([[0, max_coinc_count],
                                               [coinc_count_ref, 0]])

        # chunking

        chunked, nch = _chunking(binned_pair=binned_pair,
                                 size_chunks=size_chunks,
                                 max_lag=max_lag,
                                 best_lag=best_lag)

        marginal_counts = np.zeros((nch, maxrate, 2), dtype=np.int)

        # for every chunk, a vector with in each entry the sum of elements
        # in each parallel binary process, for each unit

        # maxrate_t : contains the maxrates for both neurons in each chunk
        maxrate_t = np.zeros(nch, dtype=np.int)

        # ch_nn : contains the length of the different chunks
        ch_nn = np.zeros(nch, dtype=np.int)
        count_sum = 0
        # for every chunk build the parallel processes
        # and the coincidence counts

        for iii in range(nch):
            binned_pair_chunked = np.array(chunked[iii])
            maxrate_t[iii] = max(max(binned_pair_chunked[0]),
                                 max(binned_pair_chunked[1]))
            ch_nn[iii] = len(chunked[iii][0])
            par_processes_chunked = [None for _ in range(
                np.int(maxrate_t[iii]))]

            for i in range(np.int(maxrate_t[iii])):
                par_processes_chunked[i] = np.zeros(
                    (2, len(binned_pair_chunked[0])), dtype=np.int)
                par_processes_chunked[i] = np.array(binned_pair_chunked > i,
                                                    dtype=np.int)

            for i in range(np.int(maxrate_t[iii])):
                par_processes_a = par_processes_chunked[i][0]
                par_processes_b = par_processes_chunked[i][1]
                marginal_counts[iii][i][0] = np.int(np.sum(par_processes_a))
                marginal_counts[iii][i][1] = np.int(np.sum(par_processes_b))
                count_sum = count_sum + min(marginal_counts[iii][i][0],
                                            marginal_counts[iii][i][1])

        # marginal_counts[iii][i] has in its entries
        # '[ #_a^{\alpha,c} , #_b^{\alpha,c}]'
        # where '\alpha' goes from 1 to maxrate, c goes from 1 to nch

        # calculation of variance for each chunk

        n = ntp - max_lag  # used in the calculation of the p-value
        var_x = [np.zeros((2, 2)) for _ in range(nch)]
        var_tot = 0
        cov_abab = [0 for _ in range(nch)]
        cov_abba = [0 for _ in range(nch)]
        var_t = [np.zeros((2, 2)) for _ in range(nch)]
        cov_x = [np.zeros((2, 2)) for _ in range(nch)]

        for iii in range(nch):  # for every chunk
            ch_size = ch_nn[iii]

            # evaluation of AB + variance and covariance

            cov_abab[iii] = [[0 for _ in range(maxrate_t[iii])]
                             for _ in range(maxrate_t[iii])]
            # for every rate up to the maxrate in that chunk
            for i in range(maxrate_t[iii]):
                par_marg_counts_i = \
                    np.outer(marginal_counts[iii][i], np.ones(2))

                cov_abab[iii][i][i] = \
                    np.multiply(
                        np.multiply(par_marg_counts_i, par_marg_counts_i.T)
                        / float(ch_size),
                        np.multiply(ch_size - par_marg_counts_i,
                                    ch_size - par_marg_counts_i.T)
                        / float(ch_size * (ch_size - 1)))

                # calculation of the variance
                var_t[iii] = var_t[iii] + cov_abab[iii][i][i]

                # cross covariances terms
                if maxrate_t[iii] > 1:
                    for j in range(i + 1, maxrate_t[iii]):
                        par_marg_counts_j = \
                            np.outer(marginal_counts[iii][j], np.ones(2))
                        cov_abab[iii][i][j] = \
                            2 * np.multiply(
                                np.multiply(par_marg_counts_j,
                                            par_marg_counts_j.T)
                                / float(ch_size),
                                np.multiply(ch_size - par_marg_counts_i,
                                            ch_size - par_marg_counts_i.T)
                                / float(ch_size * (ch_size - 1)))

                        # update of the variance
                        var_t[iii] = var_t[iii] + cov_abab[iii][i][j]

            # evaluation of coinc_count_matrix = #AB - #BA

            cov_abba[iii] = [[0 for _ in range(maxrate_t[iii])]
                             for _ in range(maxrate_t[iii])]

            for i in range(maxrate_t[iii]):
                par_marg_counts_i = \
                    np.outer(marginal_counts[iii][i], np.ones(2))
                cov_abba[iii][i][i] = \
                    np.multiply(
                        np.multiply(par_marg_counts_i, par_marg_counts_i.T)
                        / float(ch_size),
                        np.multiply(ch_size - par_marg_counts_i,
                                    ch_size - par_marg_counts_i.T)
                        / float(ch_size * (ch_size - 1) ** 2))
                cov_x[iii] = cov_x[iii] + cov_abba[iii][i][i]

                if maxrate_t[iii] > 1:
                    for j in range((i + 1), maxrate_t[iii]):
                        par_marg_counts_j = \
                            np.outer(marginal_counts[iii][j], np.ones(2))

                        cov_abba[iii][i][j] = \
                            2 * np.multiply(
                                np.multiply(par_marg_counts_j,
                                            par_marg_counts_j.T)
                                / float(ch_size),
                                np.multiply(ch_size - par_marg_counts_i,
                                            ch_size - par_marg_counts_i.T)
                                / float(ch_size * (ch_size - 1) ** 2))

                        cov_x[iii] = cov_x[iii] + cov_abba[iii][i][j]

            var_x[iii] = var_t[iii] + var_t[iii].T - cov_x[iii] - cov_x[iii].T
            var_tot = var_tot + var_x[iii]

        # Yates correction
        coinc_count_matrix = coinc_count_matrix - coinc_count_matrix.T
        if abs(coinc_count_matrix[0][1]) > 0:
            coinc_count_matrix = abs(coinc_count_matrix) - 0.5
        if var_tot[0][1] == 0:
            pr_f = 1
        # p-value obtained through approximation to a Fischer F distribution
        # (here we employ the survival function)
        else:
            fstat = coinc_count_matrix ** 2 / var_tot
            pr_f = f.sf(fstat[0][1], 1, n)

        # Creation of the dictionary with the results
        en_neurons = copy.copy(ensemble['neurons'])
        en_neurons.append(n2)
        en_lags = copy.copy(ensemble['lags'])
        en_lags.append(best_lag)
        en_pvalue = copy.copy(ensemble['pvalue'])
        en_pvalue.append(pr_f)
        en_n_occ = copy.copy(ensemble['signature'])
        en_n_occ.append([len(en_neurons), sum(activation_series)])
        assembly = {'neurons': en_neurons,
                    'lags': en_lags,
                    'pvalue': en_pvalue,
                    'times': activation_series,
                    'signature': en_n_occ}
        if same_configuration_pruning:
            return assembly, item_candidate
        else:
            return assembly


def _significance_pruning_step(pre_pruning_assembly):
    """
    Between two assemblies with the same unit set arranged into different
    configurations the most significant one is chosen.

    Parameters
    ----------
    pre_pruning_assembly : list
        contains the whole set of significant assemblies (unfiltered)

    Returns
    -------
    assembly : list
        contains the filtered assemblies
    n_filtered_assemblies : int
        number of filtered assemblies by the function
    """

    # number of assemblies before pruning
    nns = len(pre_pruning_assembly)

    # boolean array for selection of assemblies to keep
    selection = []

    # list storing the found assemblies
    assembly = []

    for i in range(nns):
        elem = sorted(pre_pruning_assembly[i]['neurons'])
        # in the list, so that membership can be checked
        if elem in selection:
            # find the element that was already in the list
            pre = selection.index(elem)

            if pre_pruning_assembly[i]['pvalue'][-1] <= \
                    assembly[pre]['pvalue'][-1]:
                # if the new element has a p-value that is smaller
                # than the one had previously
                selection[pre] = elem
                # substitute the prev element in the selection with the new
                assembly[pre] = pre_pruning_assembly[i]
                # substitute also in the list of the new assemblies
        if elem not in selection:
            selection.append(elem)
            assembly.append(pre_pruning_assembly[i])

    # number of assemblies filtered out is equal to the difference
    # between the pre and post pruning size
    n_filtered_assemblies = nns - len(assembly)

    return assembly, n_filtered_assemblies


def _subgroup_pruning_step(pre_pruning_assembly):
    """
    Removes assemblies which are already all included in a bigger assembly

    Parameters
    ----------
    pre_pruning_assembly : list
        contains the assemblies filtered by the significance value

    Returns
    --------
    final_assembly : list
        contains the assemblies filtered by inclusion

    """

    # reversing the semifinal_assembly makes the computation quicker
    # since the assembly are formed by agglomeration
    pre_pruning_assembly_r = list(reversed(pre_pruning_assembly))
    nns = len(pre_pruning_assembly_r)
    # boolean list with the selected assemblies
    selection = [True for _ in range(nns)]

    for i in range(nns):
        # check only in the range of the already selected assemblies
        if selection[i]:
            a = pre_pruning_assembly_r[i]['neurons']
            for j in range(i + 1, nns):
                if selection[j]:
                    b = pre_pruning_assembly_r[j]['neurons']
                    # check if a is included in b or vice versa
                    if set(a).issuperset(set(b)):
                        selection[j] = False
                    if set(b).issuperset(set(a)):
                        selection[i] = False
                    # only for the case in which significance_pruning=False
                    if set(a) == set(b):
                        selection[i] = True
                        selection[j] = True

    assembly_r = []

    # put into final_assembly only the selected ones
    for i in range(nns):
        if selection[i]:
            assembly_r.append(pre_pruning_assembly_r[i])

    assembly = list(reversed(assembly_r))

    return assembly


def _raise_errors(binned_spiketrain, max_lag, alpha, min_occurrences,
                  size_chunks, max_spikes):
    """
    Returns errors if the parameters given in input are not correct.

    Parameters
    ----------
    binned_spiketrain : BinnedSpikeTrain object
        binned spike trains containing data to be analysed
    max_lag : int
        maximal lag to be tested. For a binning dimension of bin_size the
        method will test all pairs configurations with a time
        shift between -max_lag and max_lag
    alpha : float
        alpha level.
    min_occurrences : int
        minimal number of occurrences required for an assembly
        (all assemblies, even if significant, with fewer occurrences
        than min_occurrences are discarded).
    size_chunks : int
        size (in bins) of chunks in which the spike trains is divided
        to compute the variance (to reduce non stationarity effects
        on variance estimation)
    max_spikes : int
        maximal assembly order (the algorithm will return assemblies of
        composed by maximum max_spikes elements).

    Raises
    ------
    TypeError
        if the data is not an elephant.conv.BinnedSpikeTrain object
    ValueError
        if the maximum lag considered is 1 or less
        if the significance level is not in [0,1]
        if the minimal number of occurrences for an assembly is less than 1
        if the length of the chunks for the variance computation is 1 or less
        if the maximal assembly order is not between 2
        and the number of neurons
        if the time series is too short (less than 100 bins)

    """

    if not isinstance(binned_spiketrain, conv.BinnedSpikeTrain):
        raise TypeError(
            'data must be in BinnedSpikeTrain format')

    if max_lag < 2:
        raise ValueError('max_lag value cant be less than 2')

    if alpha < 0 or alpha > 1:
        raise ValueError('significance level has to be in interval [0,1]')

    if min_occurrences < 1:
        raise ValueError('minimal number of occurrences for an assembly '
                         'must be at least 1')

    if size_chunks < 2:
        raise ValueError('length of the chunks cannot be 1 or less')

    if max_spikes < 2:
        raise ValueError('maximal assembly order must be less than 2')

    if binned_spiketrain.shape[1] - max_lag < 100:
        raise ValueError('The time series is too short, consider '
                         'taking a longer portion of spike train '
                         'or diminish the bin size to be tested')


# alias for the function
cad = cell_assembly_detection
