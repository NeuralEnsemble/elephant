"""
SPADE :cite:`spade-Torre2013_132,spade-Quaglio2017_41,spade-Stella2019_104022`
is the combination of a mining technique and multiple statistical tests to
detect and assess the statistical significance of repeated occurrences of spike
sequences (spatio-temporal patterns, STP).


.. autosummary::
    :toctree: _toctree/spade

    spade
    concepts_mining
    pvalue_spectrum
    test_signature_significance
    approximate_stability
    pattern_set_reduction
    concept_output_to_patterns


Visualization
-------------
Visualization of SPADE analysis is covered in Viziphant:
https://viziphant.readthedocs.io/en/latest/modules.html


Notes
-----
This modules relies on the implementation of the fp-growth algorithm contained
in the file fim.so which can be found here (http://www.borgelt.net/pyfim.html)
and should be available in the spade_src folder (elephant/spade_src/).
If the fim.so module is not present in the correct location or cannot be
imported (only available for linux OS) SPADE will make use of a python
implementation of the fast fca algorithm contained in
`elephant/spade_src/fast_fca.py`, which is about 10 times slower.

See Also
--------
elephant.cell_assembly_detection.cell_assembly_detection : another synchronous
patterns detection


Examples
--------
Given a list of Neo Spiketrain objects, assumed to be recorded in parallel, the
SPADE analysis can be applied as demonstrated in this short toy example of 10
artificial spike trains of exhibiting fully synchronous events of order 10.

>>> import quantities as pq
>>> import numpy as np
>>> from elephant.spike_train_generation import compound_poisson_process
>>> from elephant.spade import spade

Generate correlated spiketrains.

>>> np.random.seed(30)
>>> spiketrains = compound_poisson_process(rate=15*pq.Hz,
...     amplitude_distribution=[0, 0.95, 0, 0, 0, 0, 0.05], t_stop=5*pq.s)

Mining patterns with SPADE using a `bin_size` of 1 ms and a window length of 1
bin (i.e., detecting only synchronous patterns).

>>> patterns = spade(spiketrains, bin_size=10 * pq.ms, winlen=1,
...                  dither=5 * pq.ms, min_spikes=6, n_surr=10,
...                  psr_param=[0, 0, 3])['patterns']
>>> patterns[0]
{'itemset': (4, 3, 0, 2, 5, 1),
 'windows_ids': (9,
  16,
  55,
  91,
  ...,
  393,
  456,
  467),
 'neurons': [4, 3, 0, 2, 5, 1],
 'lags': array([0., 0., 0., 0., 0.]) * ms,
 'times': array([  90.,  160.,  550.,  910.,  930., 1420., 1480., 1650., 2570.,
        3130., 3430., 3480., 3610., 3800., 3830., 3930., 4560., 4670.]) * ms,
 'signature': (6, 18),
 'pvalue': 0.0}


Refer to Viziphant documentation to check how to visualzie such patterns.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function, unicode_literals

import operator
import time
import warnings
from collections import defaultdict
from functools import reduce
from itertools import chain, combinations

import neo
import numpy as np
import quantities as pq
from scipy import sparse

import elephant.conversion as conv
import elephant.spike_train_surrogates as surr
from elephant.spade_src import fast_fca
from elephant.utils import deprecated_alias

__all__ = [
    "spade",
    "concepts_mining",
    "pvalue_spectrum",
    "test_signature_significance",
    "approximate_stability",
    "pattern_set_reduction",
    "concept_output_to_patterns"
]

warnings.simplefilter('once', UserWarning)

try:
    from mpi4py import MPI  # for parallelized routines

    HAVE_MPI = True
except ImportError:  # pragma: no cover
    HAVE_MPI = False

try:
    from elephant.spade_src import fim

    HAVE_FIM = True
except ImportError:  # pragma: no cover
    HAVE_FIM = False


@deprecated_alias(binsize='bin_size')
def spade(spiketrains, bin_size, winlen, min_spikes=2, min_occ=2,
          max_spikes=None, max_occ=None, min_neu=1, approx_stab_pars=None,
          n_surr=0, dither=15 * pq.ms, spectrum='#',
          alpha=None, stat_corr='fdr_bh', surr_method='dither_spikes',
          psr_param=None, output_format='patterns', **surr_kwargs):
    r"""
    Perform the SPADE :cite:`spade-Torre2013_132`,
    :cite:`spade-Quaglio2017_41`, :cite:`spade-Stella2019_104022` analysis for
    the parallel input `spiketrains`. They are discretized with a temporal
    resolution equal to `bin_size` in a sliding window of `winlen*bin_size`.

    First, spike patterns are mined from the `spiketrains` using a technique
    called frequent itemset mining (FIM) or formal concept analysis (FCA). In
    this framework, a particular spatio-temporal spike pattern is called a
    "concept". It is then possible to compute the stability and the p-value
    of all pattern candidates. In a final step, concepts are filtered
    according to a stability threshold and a significance level `alpha`.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List containing the parallel spike trains to analyze
    bin_size : pq.Quantity
        The time precision used to discretize the spiketrains (binning).
    winlen : int
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by winlen*bin_size
    min_spikes : int, optional
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ : int, optional
        Minimum number of occurrences of a sequence to be considered as a
        pattern.
        Default: 2
    max_spikes : int, optional
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ : int, optional
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu : int, optional
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    approx_stab_pars : dict or None, optional
        Parameter values for approximate stability computation.

        'n_subsets': int
            Number of subsets of a concept used to approximate its stability.
            If `n_subsets` is 0, it is calculated according to to the formula
            given in Babin, Kuznetsov (2012), proposition 6:

            .. math::
                   n_{\text{subset}} = \frac{1}{2 \cdot \epsilon^2}
                    \ln{\left( \frac{2}{\delta} \right)} +1

        'delta' : float, optional
            delta: probability with at least :math:`1-\delta`
        'epsilon' : float, optional
            epsilon: absolute error
        'stability_thresh' : None or list of float
            List containing the stability thresholds used to filter the
            concepts.
            If `stability_thresh` is None, then the concepts are not filtered.
            Otherwise, only concepts with

            * intensional stability is greater than `stability_thresh[0]` or
            * extensional stability is greater than `stability_thresh[1]`

            are further analyzed.
    n_surr : int, optional
        Number of surrogates to generate to compute the p-value spectrum.
        This number should be large (`n_surr >= 1000` is recommended for 100
        spike trains in `spiketrains`). If `n_surr == 0`, then the p-value
        spectrum is not computed.
        Default: 0
    dither : pq.Quantity, optional
        Amount of spike time dithering for creating the surrogates for
        filtering the pattern spectrum. A spike at time `t` is placed randomly
        within `[t-dither, t+dither]` (see also
        :func:`elephant.spike_train_surrogates.dither_spikes`).
        Default: 15*pq.ms
    spectrum : {'#', '3d#'}, optional
        Define the signature of the patterns.

        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrences)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)

        Default: '#'
    alpha : float, optional
        The significance level of the hypothesis tests performed.
        If `alpha is None`, all the concepts are returned.
        If `0.<alpha<1.`, the concepts are filtered according to their
        signature in the p-value spectrum.
        Default: None
    stat_corr : str
        Method used for multiple testing.
        See: :func:`test_signature_significance`
        Default: 'fdr_bh'
    surr_method : str
        Method to generate surrogates. You can use every method defined in
        :func:`elephant.spike_train_surrogates.surrogates`.
        Default: 'dither_spikes'
    psr_param : None or list of int or tuple of int
        This list contains parameters used in the pattern spectrum filtering:

        `psr_param[0]`: correction parameter for subset filtering
            (see `h_subset_filtering` in :func:`pattern_set_reduction`).
        `psr_param[1]`: correction parameter for superset filtering
            (see `k_superset_filtering` in :func:`pattern_set_reduction`).
        `psr_param[2]`: correction parameter for covered-spikes criterion
            (see `l_covered_spikes` in :func:`pattern_set_reduction`).

    output_format : {'concepts', 'patterns'}
        Distinguish the format of the output (see Returns).
        Default: 'patterns'
    surr_kwargs
        Keyword arguments that are passed to the surrogate methods.

    Returns
    -------
    output : dict
        'patterns':
            If `output_format` is 'patterns', see the return of
            :func:`concept_output_to_patterns`

            If `output_format` is 'concepts', then `output['patterns']` is a
            tuple of patterns which in turn are tuples of
                1. spikes in the pattern
                2. occurrences of the pattern

            For details see :func:`concepts_mining`.

            if stability is calculated, there are also:
                3. intensional stability
                4. extensional stability

            For details see :func:`approximate_stability`.

        'pvalue_spectrum' (only if `n_surr > 0`):
            A list of signatures in tuples format:

            * size
            * number of occurrences
            * duration (only if `spectrum=='3d#'`)
            * p-value

        'non_sgnf_sgnt': list
            Non significant signatures of 'pvalue_spectrum'.

    Notes
    -----
    If detected, this function will use MPI to parallelize the analysis.

    Examples
    --------
    The following example applies SPADE to `spiketrains` (list of
    `neo.SpikeTrain`).

    >>> from elephant.spade import spade
    >>> import quantities as pq
    >>> bin_size = 3 * pq.ms # time resolution to discretize the spiketrains
    >>> winlen = 10 # maximal pattern length in bins (i.e., sliding window)
    >>> result_spade = spade(spiketrains, bin_size, winlen)

    """
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
    else:
        rank = 0

    compute_stability = _check_input(
        spiketrains=spiketrains, bin_size=bin_size, winlen=winlen,
        min_spikes=min_spikes, min_occ=min_occ,
        max_spikes=max_spikes, max_occ=max_occ, min_neu=min_neu,
        approx_stab_pars=approx_stab_pars,
        n_surr=n_surr, dither=dither, spectrum=spectrum,
        alpha=alpha, stat_corr=stat_corr, surr_method=surr_method,
        psr_param=psr_param, output_format=output_format)

    time_mining = time.time()
    if rank == 0 or compute_stability:
        # Mine the spiketrains for extraction of concepts
        concepts, rel_matrix = concepts_mining(
            spiketrains, bin_size, winlen, min_spikes=min_spikes,
            min_occ=min_occ, max_spikes=max_spikes, max_occ=max_occ,
            min_neu=min_neu, report='a')
        time_mining = time.time() - time_mining
        print("Time for data mining: {}".format(time_mining))

    # Decide if compute the approximated stability
    if compute_stability:
        if 'stability_thresh' in approx_stab_pars.keys():
            stability_thresh = approx_stab_pars.pop('stability_thresh')
        else:
            stability_thresh = None
        # Computing the approximated stability of all the concepts
        time_stability = time.time()
        concepts = approximate_stability(
            concepts, rel_matrix, **approx_stab_pars)
        time_stability = time.time() - time_stability
        print("Time for stability computation: {}".format(time_stability))
        # Filtering the concepts using stability thresholds
        if stability_thresh is not None:
            concepts = [concept for concept in concepts
                        if _stability_filter(concept, stability_thresh)]

    output = {}
    pv_spec = None  # initialize pv_spec to None
    # Decide whether compute pvalue spectrum
    if n_surr > 0:
        # Compute pvalue spectrum
        time_pvalue_spectrum = time.time()
        pv_spec = pvalue_spectrum(
            spiketrains, bin_size, winlen, dither=dither, n_surr=n_surr,
            min_spikes=min_spikes, min_occ=min_occ, max_spikes=max_spikes,
            max_occ=max_occ, min_neu=min_neu, spectrum=spectrum,
            surr_method=surr_method, **surr_kwargs)
        time_pvalue_spectrum = time.time() - time_pvalue_spectrum
        print("Time for pvalue spectrum computation: {}".format(
            time_pvalue_spectrum))
        # Storing pvalue spectrum
        output['pvalue_spectrum'] = pv_spec

    # rank!=0 returning None
    if rank != 0:
        warnings.warn('Returning None because executed on a process != 0')
        return None

    # Initialize non-significant signatures as empty list:
    ns_signatures = []
    # Decide whether filter concepts with psf
    if n_surr > 0:
        if len(pv_spec) > 0 and alpha is not None:
            # Computing non-significant entries of the spectrum applying
            # the statistical correction
            ns_signatures = test_signature_significance(
                pv_spec, concepts, alpha, winlen, corr=stat_corr,
                report='non_significant', spectrum=spectrum)
        # Storing non-significant entries of the pvalue spectrum
        output['non_sgnf_sgnt'] = ns_signatures
    # Filter concepts with pvalue spectrum (psf)
    if len(ns_signatures) > 0:
        concepts = [concept for concept in concepts
                    if _pattern_spectrum_filter(concept, ns_signatures,
                                                spectrum, winlen)]
        # Decide whether to filter concepts using psr
    if psr_param is not None:
        # Filter using conditional tests (psr)
        concepts = pattern_set_reduction(
            concepts, ns_signatures, winlen=winlen, spectrum=spectrum,
            h_subset_filtering=psr_param[0], k_superset_filtering=psr_param[1],
            l_covered_spikes=psr_param[2], min_spikes=min_spikes,
            min_occ=min_occ)
    # Storing patterns for output format concepts
    if output_format == 'concepts':
        output['patterns'] = concepts
    else:  # output_format == 'patterns':
        # Transforming concepts to dictionary containing pattern's infos
        output['patterns'] = concept_output_to_patterns(
            concepts, winlen, bin_size, pv_spec, spectrum,
            spiketrains[0].t_start)

    return output


def _check_input(
        spiketrains, bin_size, winlen, min_spikes=2, min_occ=2,
        max_spikes=None, max_occ=None, min_neu=1, approx_stab_pars=None,
        n_surr=0, dither=15 * pq.ms, spectrum='#',
        alpha=None, stat_corr='fdr_bh', surr_method='dither_spikes',
        psr_param=None, output_format='patterns'):
    """
    Checks all input given to SPADE
    Parameters
    ----------
    see :`func`:`spade`

    Returns
    -------
    compute_stability: bool
        if the stability calculation is used
    """

    # Check spiketrains
    if not all([isinstance(elem, neo.SpikeTrain) for elem in spiketrains]):
        raise TypeError(
            'spiketrains must be a list of SpikeTrains')
    # Check that all spiketrains have same t_start and same t_stop
    if not all([spiketrain.t_start == spiketrains[0].t_start
                for spiketrain in spiketrains]) or \
            not all([spiketrain.t_stop == spiketrains[0].t_stop
                     for spiketrain in spiketrains]):
        raise ValueError(
            'All spiketrains must have the same t_start and t_stop')

    # Check bin_size
    if not isinstance(bin_size, pq.Quantity):
        raise TypeError('bin_size must be a pq.Quantity')

    # Check winlen
    if not isinstance(winlen, int):
        raise TypeError('winlen must be an integer')

    # Check min_spikes
    if not isinstance(min_spikes, int):
        raise TypeError('min_spikes must be an integer')

    # Check min_occ
    if not isinstance(min_occ, int):
        raise TypeError('min_occ must be an integer')

    # Check max_spikes
    if not (isinstance(max_spikes, int) or max_spikes is None):
        raise TypeError('max_spikes must be an integer or None')

    # Check max_occ
    if not (isinstance(max_occ, int) or max_occ is None):
        raise TypeError('max_occ must be an integer or None')

    # Check min_neu
    if not isinstance(min_neu, int):
        raise TypeError('min_neu must be an integer')

    # Check approx_stab_pars
    compute_stability = False
    if isinstance(approx_stab_pars, dict):
        if 'n_subsets' in approx_stab_pars.keys() or\
            ('epsilon' in approx_stab_pars.keys() and
             'delta' in approx_stab_pars.keys()):
            compute_stability = True
        else:
            raise ValueError(
                'for approximate stability computation you need to '
                'pass n_subsets or epsilon and delta.')

    # Check n_surr
    if not isinstance(n_surr, int):
        raise TypeError('n_surr must be an integer')

    # Check dither
    if not isinstance(dither, pq.Quantity):
        raise TypeError('dither must be a pq.Quantity')

    # Check spectrum
    if spectrum not in ('#', '3d#'):
        raise ValueError("spectrum must be '#' or '3d#'")

    # Check alpha
    if isinstance(alpha, (int, float)):
        # Check redundant use of alpha
        if 0. < alpha < 1. and n_surr == 0:
            warnings.warn('0.<alpha<1. but p-value spectrum has not been '
                          'computed (n_surr==0)')
    elif alpha is not None:
        raise TypeError('alpha must be an integer, a float or None')

    # Check stat_corr:
    if stat_corr not in \
            ('bonferroni', 'sidak', 'holm-sidak', 'holm',
             'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
             'fdr_tsbh', 'fdr_tsbky', '', 'no'):
        raise ValueError("Parameter stat_corr not recognized")

    # Check surr_method
    if surr_method not in surr.SURR_METHODS:
        raise ValueError(
            'specified surr_method (=%s) not valid' % surr_method)

    # Check psr_param
    if psr_param is not None:
        if not isinstance(psr_param, (list, tuple)):
            raise TypeError('psr_param must be None or a list or tuple of '
                            'integer')
        if not all(isinstance(param, int) for param in psr_param):
            raise TypeError('elements of psr_param must be integers')

    # Check output_format
    if output_format not in ('concepts', 'patterns'):
        raise ValueError("The output_format value has to be"
                         "'patterns' or 'concepts'")

    return compute_stability


@deprecated_alias(binsize='bin_size')
def concepts_mining(spiketrains, bin_size, winlen, min_spikes=2, min_occ=2,
                    max_spikes=None, max_occ=None, min_neu=1, report='a'):
    """
    Find pattern candidates extracting all the concepts of the context, formed
    by the objects defined as all windows of length `winlen*bin_size` slided
    along the discretized `spiketrains` and the attributes as the spikes
    occurring in each of the windows. Hence, the output are all the repeated
    sequences of spikes with maximal length `winlen`, which are not trivially
    explained by the same number of occurrences of a superset of spikes.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or conv.BinnedSpikeTrain
        Either list of the spiketrains to analyze or
        BinningSpikeTrain object containing the binned spiketrains to analyze
    bin_size : pq.Quantity
        The time precision used to discretize the `spiketrains` (clipping).
    winlen : int
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by `winlen*bin_size`
    min_spikes : int, optional
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ : int, optional
        Minimum number of occurrences of a sequence to be considered as a
        pattern.
        Default: 2
    max_spikes : int, optional
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ : int, optional
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu : int, optional
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    report : {'a', '#', '3d#'}, optional
        Indicates the output of the function.

        'a':
            All the mined patterns

        '#':
            Pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)

        '3d#':
            Pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)

        Default: 'a'

    Returns
    -------
    mining_results : np.ndarray
        If report == 'a':
            Numpy array of all the pattern candidates (concepts) found in the
            `spiketrains`. Each pattern is represented as a tuple containing
            (spike IDs, discrete times (window position) of the  occurrences
            of the pattern). The spike IDs are defined as:
            `spike_id=neuron_id*bin_id` with `neuron_id` in
            `[0, len(spiketrains)]` and `bin_id` in `[0, winlen]`.
        If report == '#':
             The pattern spectrum is represented as a numpy array of triplets
             (pattern size, number of occurrences, number of patterns).
        If report == '3d#':
             The pattern spectrum is represented as a numpy array of
             quadruplets (pattern size, number of occurrences, difference
             between last and first spike of the pattern, number of patterns)
    rel_matrix : sparse.coo_matrix
        A binary matrix of shape (number of windows, winlen*len(spiketrains)).
        Each row corresponds to a window (order
        according to their position in time). Each column corresponds to one
        bin and one neuron and it is 0 if no spikes or 1 if one or more spikes
        occurred in that bin for that particular neuron. For example, the entry
        [0,0] of this matrix corresponds to the first bin of the first window
        position for the first neuron, the entry `[0,winlen]` to the first
        bin of the first window position for the second neuron.
    """
    if report not in ('a', '#', '3d#'):
        raise ValueError(
            "report has to assume of the following values:" +
            "  'a', '#' and '3d#,' got {} instead".format(report))
    # if spiketrains is list of neo.SpikeTrain convert to conv.BinnedSpikeTrain
    if isinstance(spiketrains, list) and \
            isinstance(spiketrains[0], neo.SpikeTrain):
        spiketrains = conv.BinnedSpikeTrain(
            spiketrains, bin_size=bin_size, tolerance=None)
    if not isinstance(spiketrains, conv.BinnedSpikeTrain):
        raise TypeError(
            'spiketrains must be either a list of neo.SpikeTrain or '
            'a conv.BinnedSpikeTrain object')
    # Clipping the spiketrains and (binary matrix)
    binary_matrix = spiketrains.to_sparse_bool_array().tocoo(copy=False)
    # Computing the context and the binary matrix encoding the relation between
    # objects (window positions) and attributes (spikes,
    # indexed with a number equal to  neuron idx*winlen+bin idx)
    context, transactions, rel_matrix = _build_context(binary_matrix, winlen)
    # By default, set the maximum pattern size to the maximum number of
    # spikes in a window
    if max_spikes is None:
        max_spikes = binary_matrix.shape[0] * winlen
    # By default, set maximum number of occurrences to number of non-empty
    # windows
    if max_occ is None:
        max_occ = int(np.sum(np.sum(rel_matrix, axis=1) > 0))
    # Check if fim.so available and use it
    if HAVE_FIM:
        # Return the output
        mining_results = _fpgrowth(
            transactions,
            rel_matrix=rel_matrix,
            min_c=min_occ,
            min_z=min_spikes,
            max_z=max_spikes,
            max_c=max_occ,
            winlen=winlen,
            min_neu=min_neu,
            report=report)
        return mining_results, rel_matrix
    # Otherwise use fast_fca python implementation
    warnings.warn(
        'Optimized C implementation of FCA (fim.so/fim.pyd) not found ' +
        'in elephant/spade_src folder, or not compatible with this ' +
        'Python version. You are using the pure Python implementation ' +
        'of fast fca.')
    # Return output
    mining_results = _fast_fca(
        context,
        min_c=min_occ,
        min_z=min_spikes,
        max_z=max_spikes,
        max_c=max_occ,
        winlen=winlen,
        min_neu=min_neu,
        report=report)
    return mining_results, rel_matrix


def _build_context(binary_matrix, winlen):
    """
    Building the context given a matrix (number of trains x number of bins) of
    binned spike trains

    Parameters
    ----------
    binary_matrix : sparse.coo_matrix
        Binary matrix containing the binned spike trains
    winlen : int
        Length of the bin_size used to bin the spiketrains

    Returns
    -------
    context : list of tuple
        List of tuples containing one object (window position idx) and one of
        the correspondent spikes idx (bin idx * neuron idx)
    transactions : list
        List of all transactions, each element of the list contains the
        attributes of the corresponding object.
    rel_matrix : sparse.coo_matrix
        A binary matrix with shape (number of windows,
        winlen*len(spiketrains)). Each row corresponds to a window (order
        according to their position in time).
        Each column corresponds to one bin and one neuron and it is 0 if no
        spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron.
        E.g. the entry [0,0] of this matrix corresponds to the first bin of the
        first window position for the first neuron, the entry [0,winlen] to the
        first bin of the first window position for the second neuron.
    """
    # Initialization of the outputs
    context = []
    transactions = []
    num_neurons, num_bins = binary_matrix.shape
    indices = np.argsort(binary_matrix.col)
    binary_matrix.row = binary_matrix.row[indices]
    binary_matrix.col = binary_matrix.col[indices]
    # out of all window positions
    # get all non-empty first bins
    unique_cols, unique_col_idx = np.unique(
        binary_matrix.col, return_index=True)
    unique_col_idx = np.concatenate((unique_col_idx, [len(binary_matrix.col)]))
    windows_row = []
    windows_col = []
    # all non-empty bins are starting positions for windows
    for idx, window_idx in enumerate(unique_cols):
        # find the end of the current window in unique_cols
        end_of_window = np.searchsorted(unique_cols, window_idx + winlen)
        # loop over all non-empty bins in the current window
        for rel_idx, col in enumerate(unique_cols[idx:end_of_window]):
            # get all occurrences of the current col in binary_matrix.col
            spike_indices_in_window = np.arange(
                unique_col_idx[idx + rel_idx],
                unique_col_idx[idx + rel_idx + 1])
            # get the binary_matrix.row entries matching the current col
            # prepare the row of rel_matrix matching the current window
            # spikes are indexed as (neuron_id * winlen + bin_id)
            windows_col.extend(
                binary_matrix.row[spike_indices_in_window] * winlen
                + (col - window_idx))
            windows_row.extend([window_idx] * len(spike_indices_in_window))
    # Shape of the rel_matrix:
    # (total number of bins,
    #  number of bins in one window * number of neurons)
    rel_matrix = sparse.coo_matrix(
        (np.ones((len(windows_col)), dtype=bool),
         (windows_row, windows_col)),
        shape=(num_bins, winlen * num_neurons),
        dtype=bool).A
    # Array containing all the possible attributes (each spike is indexed by
    # a number equal to neu idx*winlen + bin_idx)
    attributes = np.array(
        [s * winlen + t for s in range(binary_matrix.shape[0])
         for t in range(winlen)])
    # Building context and rel_matrix
    # Looping all the window positions w
    for window in unique_cols:
        # spikes in the current window
        times = rel_matrix[window]
        current_transactions = attributes[times]
        # adding to the context the window positions and the correspondent
        # attributes (spike idx) (fast_fca input)
        context.extend(
            (window, transaction) for transaction in current_transactions)
        # appending to the transactions spike idx (fast_fca input) of the
        # current window (fpgrowth input)
        transactions.append(list(current_transactions))
    # Return context and rel_matrix
    return context, transactions, rel_matrix


def _fpgrowth(transactions, min_c=2, min_z=2, max_z=None,
              max_c=None, rel_matrix=None, winlen=1, min_neu=1,
              target='c', report='a'):
    """
    Find frequent item sets with the fpgrowth algorithm.

    Parameters
    ----------
    transactions: tuple
                Transactions database to mine.
                The database must be an iterable of transactions;
                each transaction must be an iterable of items;
                each item must be a hashable object.
                If the database is a dictionary, the transactions are
                the keys, the values their (integer) multiplicities.
    target: str
            type of frequent item sets to find
            s/a   sets/all   all     frequent item sets
            c     closed     closed  frequent item sets
            m     maximal    maximal frequent item sets
            g     gens       generators
            Default:'c'
    min_c: int
        minimum support of an item set
        Default: 2
    min_z: int
         minimum number of items per item set
        Default: 2
    max_z: None/int
         maximum number of items per item set. If max_c==None no maximal
         size required
        Default: None
    max_c: None/int
         maximum support per item set. If max_c==None no maximal
         support required
        Default: None
    report: str
        'a': all the mined patterns
        '#': pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)
        Default: 'a'
    rel_matrix : None or sparse.coo_matrix
        A binary matrix with shape (number of windows,
        winlen*len(spiketrains)). Each row corresponds to a window (order
        according to their position in time).
        Each column corresponds to one bin and one neuron and it is 0 if no
        spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron.
        E.g. the entry [0,0] of this matrix corresponds to the first bin of the
        first window position for the first neuron, the entry [0,winlen] to the
        first bin of the first window position for the second neuron.
        If == None only the closed frequent itemsets (intent) are returned and
        not which the index of their occurrences (extent)
        Default: None
    The following parameters are specific to Massive parallel SpikeTrains
    winlen: int
        The size (number of bins) of the sliding window used for the
        analysis. The maximal length of a pattern (delay between first and
        last spike) is then given by winlen*bin_size
        Default: 1
    min_neu: int
         Minimum number of neurons in a sequence to considered a
         potential pattern.
         Default: 1

    Returns
    -------
    If report == 'a':
        numpy array of all the pattern candidates (concepts) found in the
        spiketrains.
        Each pattern is represented as a tuple containing
        (spike IDs, discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(spiketrains)] and
        bin_id in [0, winlen].
    If report == '#':
         The pattern spectrum is represented as a numpy array of triplets each
         formed by:
            (pattern size, number of occurrences, number of patterns)
    If report == '3d#':
         The pattern spectrum is represented as a numpy array of quadruplets
         each formed by:
            (pattern size, number of occurrences, difference between last
            and first spike of the pattern, number of patterns)

    """
    if min_neu < 1:
        raise ValueError('min_neu must be an integer >=1')
    # By default, set the maximum pattern size to the number of spiketrains
    if max_z is None:
        max_z = max(max(map(len, transactions)), min_z + 1)
    # By default set maximum number of data to number of bins
    if max_c is None:
        max_c = len(transactions)

    # Initializing outputs
    concepts = []
    if report == '#':
        spec_matrix = np.zeros((max_z + 1, max_c + 1))
    if report == '3d#':
        spec_matrix = np.zeros((max_z + 1, max_c + 1, winlen))
    spectrum = []
    # check whether all transactions are identical
    # in that case FIM would not find anything,
    # so we need to create the output manually
    # for optimal performance,
    # we do the check sequentially and immediately break
    # once we find a second unique transaction
    first_transaction = transactions[0]
    for transaction in transactions[1:]:
        if transaction != first_transaction:
            # Mining the spiketrains with fpgrowth algorithm
            fpgrowth_output = fim.fpgrowth(
                tracts=transactions,
                target=target,
                supp=-min_c,
                zmin=min_z,
                zmax=max_z,
                report='a',
                algo='s')
            break
    else:
        fpgrowth_output = [(tuple(transactions[0]), len(transactions))]
    # Applying min/max conditions and computing extent (window positions)
    fpgrowth_output = [concept for concept in fpgrowth_output
                       if _fpgrowth_filter(concept, winlen, max_c, min_neu)]
    # filter out subsets of patterns that are found as a side-effect
    # of using the moving window strategy
    fpgrowth_output = _filter_for_moving_window_subsets(
        fpgrowth_output, winlen)
    for (intent, supp) in fpgrowth_output:
        if report == 'a':
            if rel_matrix is not None:
                # Computing the extent of the concept (patterns
                # occurrences), checking in rel_matrix in which windows
                # the intent occurred
                extent = tuple(
                    np.nonzero(
                        np.all(rel_matrix[:, intent], axis=1)
                    )[0])
                concepts.append((intent, extent))
        # Computing 2d spectrum
        elif report == '#':
            spec_matrix[len(intent) - 1, supp - 1] += 1
        # Computing 3d spectrum
        elif report == '3d#':
            spec_matrix[len(intent) - 1, supp - 1, max(
                np.array(intent) % winlen)] += 1
    del fpgrowth_output
    if report == 'a':
        return concepts

    if report == '#':
        for (size, occurrences) in np.transpose(np.where(spec_matrix != 0)):
            spectrum.append(
                (size + 1, occurrences + 1,
                 int(spec_matrix[size, occurrences])))
    elif report == '3d#':
        for (size, occurrences, duration) in\
                np.transpose(np.where(spec_matrix != 0)):
            spectrum.append(
                (size + 1, occurrences + 1, duration,
                 int(spec_matrix[size, occurrences, duration])))
    del spec_matrix
    if len(spectrum) > 0:
        spectrum = np.array(spectrum)
    elif report == '#':
        spectrum = np.zeros(shape=(0, 3))
    elif report == '3d#':
        spectrum = np.zeros(shape=(0, 4))
    return spectrum


def _fpgrowth_filter(concept, winlen, max_c, min_neu):
    """
    Filter for selecting closed frequent items set with a minimum number of
    neurons and a maximum number of occurrences and first spike in the first
    bin position
    """
    intent = np.array(concept[0])
    keep_concept = (min(intent % winlen) == 0
                    and concept[1] <= max_c
                    and np.unique(intent // winlen).shape[0] >= min_neu
                    )
    return keep_concept


def _rereference_to_last_spike(transactions, winlen):
    """
    Converts transactions from the default format
    neu_idx * winlen + bin_idx (relative to window start)
    into the format
    neu_idx * winlen + bin_idx (relative to last spike)
    """
    len_transactions = len(transactions)
    neurons = np.zeros(len_transactions, dtype=int)
    bins = np.zeros(len_transactions, dtype=int)

    # extract neuron and bin indices
    for idx, attribute in enumerate(transactions):
        neurons[idx] = attribute // winlen
        bins[idx] = attribute % winlen

    # rereference bins to last spike
    bins = bins.max() - bins

    # calculate converted transactions
    converted_transactions = neurons * winlen + bins

    return converted_transactions


def _filter_for_moving_window_subsets(concepts, winlen):
    """
    Since we're using a moving window subpatterns starting from
    subsequent spikes after the first pattern spike will also be found.
    This filter removes them if they do not occur on their own in
    addition to the occurrences explained by their superset.
    Uses a reverse map with a set representation.
    """
    # don't do anything if the input list is empty
    if len(concepts) == 0:
        return concepts

    # don't do anything if winlen is one
    if winlen == 1:
        return concepts

    if hasattr(concepts[0], 'intent'):
        # fca format
        # sort the concepts by (decreasing) support
        concepts.sort(key=lambda c: -len(c.extent))

        support = np.array([len(c.extent) for c in concepts])

        # convert transactions relative to last pattern spike
        converted_transactions = [_rereference_to_last_spike(concept.intent,
                                                             winlen=winlen)
                                  for concept in concepts]
    else:
        # fim.fpgrowth format
        # sort the concepts by (decreasing) support
        concepts.sort(key=lambda concept: -concept[1])

        support = np.array([concept[1] for concept in concepts])

        # convert transactions relative to last pattern spike
        converted_transactions = [_rereference_to_last_spike(concept[0],
                                                             winlen=winlen)
                                  for concept in concepts]

    output = []

    for current_support in np.unique(support):
        support_indices = np.nonzero(support == current_support)[0]

        # construct reverse map
        reverse_map = defaultdict(set)
        for map_idx, i in enumerate(support_indices):
            for window_bin in converted_transactions[i]:
                reverse_map[window_bin].add(map_idx)

        for i in support_indices:
            intersection = reduce(
                operator.and_,
                (reverse_map[window_bin]
                 for window_bin in converted_transactions[i]))
            if len(intersection) == 1:
                output.append(concepts[i])

    return output


def _fast_fca(context, min_c=2, min_z=2, max_z=None,
              max_c=None, report='a', winlen=1, min_neu=1):
    """
    Find concepts of the context with the fast-fca algorithm.

    Parameters
    ----------
    context : list
        List of tuples containing one object and one the correspondent
        attribute
    min_c: int
        minimum support of an item set
        Default: 2
    min_z: int
         minimum number of items per item set
        Default: 2
    max_z: None/int
         maximum number of items per item set. If max_c==None no maximal
         size required
        Default: None
    max_c: None/int
         maximum support per item set. If max_c==None no maximal
         support required
        Default: None
    report: str
        'a': all the mined patterns
        '#': pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)
        Default: 'a'
    The following parameters are specific to Massive parallel SpikeTrains
    winlen: int
        The size (number of bins) of the sliding window used for the
        analysis. The maximal length of a pattern (delay between first and
        last spike) is then given by winlen*bin_size
        Default: 1
    min_neu: int
         Minimum number of neurons in a sequence to considered a
         potential pattern.
         Default: 1

    Returns
    -------
    If report == 'a':
        All the pattern candidates (concepts) found in the spiketrains. Each
        pattern is represented as a tuple containing
        (spike IDs, discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(spiketrains)] and
        bin_id in [0, winlen].
    If report == '#':
         The pattern spectrum is represented as a list of triplets each
         formed by:
            (pattern size, number of occurrences, number of patterns)
    If report == '3d#':
         The pattern spectrum is represented as a list of quadruplets each
         formed by:
            (pattern size, number of occurrences, difference between last
            and first spike of the pattern, number of patterns)
    """
    # Initializing outputs
    concepts = []
    # Check parameters
    if min_neu < 1:
        raise ValueError('min_neu must be an integer >=1')
    # By default set maximum number of attributes
    if max_z is None:
        max_z = len(context)
    # By default set maximum number of data to number of bins
    if max_c is None:
        max_c = len(context)
    if report == '#':
        spec_matrix = np.zeros((max_z, max_c))
    elif report == '3d#':
        spec_matrix = np.zeros((max_z, max_c, winlen))
    spectrum = []
    # Mining the spiketrains with fast fca algorithm
    fca_out = fast_fca.FormalConcepts(context)
    fca_out.computeLattice()
    fca_concepts = fca_out.concepts
    fca_concepts = [concept for concept in fca_concepts
                    if _fca_filter(concept, winlen, min_c, min_z, max_c, max_z,
                                   min_neu)]
    fca_concepts = _filter_for_moving_window_subsets(fca_concepts, winlen)
    # Applying min/max conditions
    for fca_concept in fca_concepts:
        intent = tuple(fca_concept.intent)
        extent = tuple(fca_concept.extent)
        concepts.append((intent, extent))
        # computing spectrum
        if report == '#':
            spec_matrix[len(intent) - 1, len(extent) - 1] += 1
        elif report == '3d#':
            spec_matrix[len(intent) - 1, len(extent) - 1, max(
                np.array(intent) % winlen)] += 1
    if report == 'a':
        return concepts

    del concepts
    # returning spectrum
    if report == '#':
        for (size, occurrence) in np.transpose(np.where(spec_matrix != 0)):
            spectrum.append(
                (size + 1, occurrence + 1, int(spec_matrix[size, occurrence])))

    if report == '3d#':
        for (size, occurrence, duration) in\
                np.transpose(np.where(spec_matrix != 0)):
            spectrum.append(
                (size + 1, occurrence + 1, duration,
                 int(spec_matrix[size, occurrence, duration])))
    del spec_matrix
    if len(spectrum) > 0:
        spectrum = np.array(spectrum)
    elif report == '#':
        spectrum = np.zeros(shape=(0, 3))
    elif report == '3d#':
        spectrum = np.zeros(shape=(0, 4))
    return spectrum


def _fca_filter(concept, winlen, min_c, min_z, max_c, max_z, min_neu):
    """
    Filter to select concepts with minimum/maximum number of spikes and
    occurrences and first spike in the first bin position
    """
    intent = tuple(concept.intent)
    extent = tuple(concept.extent)
    keep_concepts = \
        min_z <= len(intent) <= max_z and \
        min_c <= len(extent) <= max_c and \
        len(np.unique(np.array(intent) // winlen)) >= min_neu and \
        min(np.array(intent) % winlen) == 0
    return keep_concepts


@deprecated_alias(binsize='bin_size')
def pvalue_spectrum(
        spiketrains, bin_size, winlen, dither, n_surr, min_spikes=2, min_occ=2,
        max_spikes=None, max_occ=None, min_neu=1, spectrum='#',
        surr_method='dither_spikes', **surr_kwargs):
    """
    Compute the p-value spectrum of pattern signatures extracted from
    surrogates of parallel spike trains, under the null hypothesis of
    independent spiking.

    * n_surr surrogates are obtained from each spike train by spike dithering
    * pattern candidates (concepts) are collected from each surrogate data
    * the signatures (number of spikes, number of occurrences) of all patterns
      are computed, and their  occurrence probability estimated by their
      occurrence frequency (p-value spectrum)

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List containing the parallel spike trains to analyze
    bin_size : pq.Quantity
        The time precision used to discretize the `spiketrains` (binning).
    winlen : int
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by `winlen*bin_size`
    dither : pq.Quantity
        Amount of spike time dithering for creating the surrogates for
        filtering the pattern spectrum. A spike at time t is placed randomly
        within `[t-dither, t+dither]` (see also
        :func:`elephant.spike_train_surrogates.dither_spikes`).
        Default: 15*pq.s
    n_surr : int
        Number of surrogates to generate to compute the p-value spectrum.
        This number should be large (`n_surr>=1000` is recommended for 100
        spike trains in spiketrains). If `n_surr` is 0, then the p-value
        spectrum is not computed.
        Default: 0
    min_spikes : int, optional
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ : int, optional
        Minimum number of occurrences of a sequence to be considered as a
        pattern.
        Default: 2
    max_spikes : int, optional
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ : int, optional
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu : int, optional
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    spectrum : {'#', '3d#'}, optional
        Defines the signature of the patterns.

        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)

        Default: '#'
    surr_method : str
        Method that is used to generate the surrogates. You can use every
        method defined in
        :func:`elephant.spike_train_surrogates.dither_spikes`.
        Default: 'dither_spikes'
    surr_kwargs
        Keyword arguments that are passed to the surrogate methods.

    Returns
    -------
    pv_spec : list

        if spectrum == '#':
            A list of triplets (z,c,p), where (z,c) is a pattern signature
            and p is the corresponding p-value (fraction of surrogates
            containing signatures (z*,c*)>=(z,c)).
        if spectrum == '3d#':
            A list of triplets (z,c,l,p), where (z,c,l) is a pattern signature
            and p is the corresponding p-value (fraction of surrogates
            containing signatures (z*,c*,l*)>=(z,c,l)).

        Signatures whose empirical p-value is 0 are not listed.

    """
    # Initializing variables for parallel computing
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
        size = comm.Get_size()  # get tot number of MPI tasks
    else:
        rank = 0
        size = 1
    # Check on number of surrogates
    if n_surr <= 0:
        raise ValueError('n_surr has to be >0')
    if surr_method not in surr.SURR_METHODS:
        raise ValueError(
            'specified surr_method (=%s) not valid' % surr_method)
    if spectrum not in ('#', '3d#'):
        raise ValueError("Invalid spectrum: '{}'".format(spectrum))

    len_partition = n_surr // size  # length of each MPI task
    len_remainder = n_surr % size

    add_remainder = rank < len_remainder

    if max_spikes is None:
        # if max_spikes not defined, set it to the number of spiketrains times
        # number of bins per window.
        max_spikes = len(spiketrains) * winlen

    if spectrum == '#':
        max_occs = np.empty(shape=(len_partition + add_remainder,
                                   max_spikes - min_spikes + 1),
                            dtype=np.uint16)
    else:  # spectrum == '3d#':
        max_occs = np.empty(shape=(len_partition + add_remainder,
                                   max_spikes - min_spikes + 1, winlen),
                            dtype=np.uint16)

    for surr_id, binned_surrogates in _generate_binned_surrogates(
            spiketrains, bin_size=bin_size, dither=dither,
            surr_method=surr_method, n_surrogates=len_partition+add_remainder,
            **surr_kwargs):

        # Find all pattern signatures in the current surrogate data set
        surr_concepts = concepts_mining(
            binned_surrogates, bin_size, winlen, min_spikes=min_spikes,
            max_spikes=max_spikes, min_occ=min_occ, max_occ=max_occ,
            min_neu=min_neu, report=spectrum)[0]
        # The last entry of the signature is the number of times the
        # signature appeared. This entry is not needed here.
        surr_concepts = surr_concepts[:, :-1]

        max_occs[surr_id] = _get_max_occ(
            surr_concepts, min_spikes, max_spikes, winlen, spectrum)

    # Collecting results on the first PCU
    if size != 1:
        max_occs = comm.gather(max_occs, root=0)

        if rank != 0:  # pragma: no cover
            return []

        # The gather operator gives a list out. This is rearranged as a 2 resp.
        # 3 dimensional numpy-array.
        max_occs = np.vstack(max_occs)

    # Compute the p-value spectrum, and return it
    return _get_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ,
                            n_surr, winlen, spectrum)


def _generate_binned_surrogates(
        spiketrains, bin_size, dither, surr_method, n_surrogates,
        **surr_kwargs):
    if surr_method == 'bin_shuffling':
        binned_spiketrains = [
            conv.BinnedSpikeTrain(
                spiketrain, bin_size=bin_size, tolerance=None)
            for spiketrain in spiketrains]
        max_displacement = int(dither.rescale(pq.ms).magnitude /
                               bin_size.rescale(pq.ms).magnitude)
    elif surr_method in ('joint_isi_dithering', 'isi_dithering'):
        isi_dithering = surr_method == 'isi_dithering'
        joint_isi_instances = \
            [surr.JointISI(spiketrain, dither=dither,
                           isi_dithering=isi_dithering, **surr_kwargs)
             for spiketrain in spiketrains]
    for surr_id in range(n_surrogates):
        if surr_method == 'bin_shuffling':
            binned_surrogates = \
                [surr.bin_shuffling(binned_spiketrain,
                                    max_displacement=max_displacement,
                                    **surr_kwargs)[0]
                 for binned_spiketrain in binned_spiketrains]
            binned_surrogates = np.array(
                [binned_surrogate.to_bool_array()[0]
                 for binned_surrogate in binned_surrogates])
            binned_surrogates = conv.BinnedSpikeTrain(
                binned_surrogates,
                bin_size=bin_size,
                t_start=spiketrains[0].t_start,
                t_stop=spiketrains[0].t_stop)
        elif surr_method in ('joint_isi_dithering', 'isi_dithering'):
            surrs = [instance.dithering()[0]
                     for instance in joint_isi_instances]
        elif surr_method == 'dither_spikes_with_refractory_period':
            # The initial refractory period is set to the bin size in order to
            # prevent that spikes fall into the same bin, if the spike trains
            # are sparse (min(ISI)>bin size).
            surrs = \
                [surr.dither_spikes(
                    spiketrain, dither=dither, n_surrogates=1,
                    refractory_period=bin_size, **surr_kwargs)[0]
                 for spiketrain in spiketrains]
        else:
            surrs = \
                [surr.surrogates(
                    spiketrain, n_surrogates=1, method=surr_method,
                    dt=dither, **surr_kwargs)[0]
                 for spiketrain in spiketrains]

        if surr_method != 'bin_shuffling':
            binned_surrogates = conv.BinnedSpikeTrain(
                surrs, bin_size=bin_size, tolerance=None)
        yield surr_id, binned_surrogates


def _get_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ, n_surr, winlen,
                     spectrum):
    """
    This function converts the list of maximal occurrences into the
    corresponding p-value spectrum.

    Parameters
    ----------
    max_occs: np.ndarray
    min_spikes: int
    max_spikes: int
    min_occ: int
    n_surr: int
    winlen: int
    spectrum: {'#', '3d#'}

    Returns
    -------
    if spectrum == '#':
    List[List]:
        each entry has the form: [pattern_size, pattern_occ, p_value]
    if spectrum == '3d#':
    List[List]:
        each entry has the form:
        [pattern_size, pattern_occ, pattern_dur, p_value]
    """
    if spectrum not in ('#', '3d#'):
        raise ValueError("Invalid spectrum: '{}'".format(spectrum))

    pv_spec = []
    if spectrum == '#':
        max_occs = np.expand_dims(max_occs, axis=2)
        winlen = 1
    for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
        for dur in range(winlen):
            max_occs_size_dur = max_occs[:, size_id, dur]
            counts, occs = np.histogram(
                max_occs_size_dur,
                bins=np.arange(min_occ,
                               np.max(max_occs_size_dur) + 2))
            occs = occs[:-1].astype(np.uint16)
            pvalues = np.cumsum(counts[::-1])[::-1] / n_surr
            for occ_id, occ in enumerate(occs):
                if spectrum == '#':
                    pv_spec.append([pt_size, occ, pvalues[occ_id]])
                else:  # spectrum == '3d#':
                    pv_spec.append([pt_size, occ, dur, pvalues[occ_id]])
    return pv_spec


def _get_max_occ(surr_concepts, min_spikes, max_spikes, winlen, spectrum):
    """
    This function takes from a list of surrogate_concepts those concepts which
    have the highest occurrence for a given pattern size and duration.

    Parameters
    ----------
    surr_concepts: List[List]
    min_spikes: int
    max_spikes: int
    winlen: int
    spectrum: {'#', '3d#'}

    Returns
    -------
    np.ndarray
        Two-dimensional array. Each element corresponds to a highest occurrence
        for a specific pattern size (which range from min_spikes to max_spikes)
        and pattern duration (which range from 0 to winlen-1).
        The first axis corresponds to the pattern size the second to the
        duration.
    """
    if spectrum == '#':
        winlen = 1
    max_occ = np.zeros(shape=(max_spikes - min_spikes + 1, winlen))
    for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
        concepts_for_size = surr_concepts[
            surr_concepts[:, 0] == pt_size][:, 1:]
        for dur in range(winlen):
            if spectrum == '#':
                occs = concepts_for_size[:, 0]
            else:  # spectrum == '3d#':
                occs = concepts_for_size[concepts_for_size[:, 1] == dur][:, 0]
            max_occ[size_id, dur] = np.max(occs, initial=0)

    for pt_size in range(max_spikes - 1, min_spikes - 1, -1):
        size_id = pt_size - min_spikes
        max_occ[size_id] = np.max(max_occ[size_id:size_id + 2], axis=0)
    if spectrum == '#':
        max_occ = np.squeeze(max_occ, axis=1)

    return max_occ


def _stability_filter(concept, stability_thresh):
    """Criteria by which to filter concepts from the lattice"""
    # stabilities larger then stability_thresh
    keep_concept = \
        concept[2] > stability_thresh[0]\
        or concept[3] > stability_thresh[1]
    return keep_concept


def _mask_pvalue_spectrum(pv_spec, concepts, spectrum, winlen):
    """
    The function filters the pvalue spectrum based on the number of
    the statistical tests to be done. Only the entries of the pvalue spectrum
    that coincide with concepts found in the original data are kept.
    Moreover, entries of the pvalue spectrum with a value of 1 (all surrogates
    datasets containing at least one mined pattern with that signature)
    are discarded as well and considered trivial.
    Parameters
    ----------
    pv_spec: List[List]
    concepts: List[Tuple]
    spectrum: {'#', '3d#'}
    winlen: int

    Returns
    -------
    mask : np.array
        An array of boolean values, indicating if a signature of p-value
        spectrum is also in the mined concepts of the original data.
    """
    if spectrum == '#':
        signatures = {(len(concept[0]), len(concept[1]))
                      for concept in concepts}
    else:  # spectrum == '3d#':
        # third entry of signatures is the duration, fixed as the maximum lag
        signatures = {(len(concept[0]), len(concept[1]),
                       max(np.array(concept[0]) % winlen))
                      for concept in concepts}
    mask = np.zeros(len(pv_spec), dtype=bool)
    for index, pv_entry in enumerate(pv_spec):
        if tuple(pv_entry[:-1]) in signatures \
                and not np.isclose(pv_entry[-1], [1]):
            # select the highest number of occurrences for size and duration
            mask[index] = True
            if mask[index-1]:
                if spectrum == '#':
                    size = pv_spec[index][0]
                    prev_size = pv_spec[index-1][0]
                    if prev_size == size:
                        mask[index-1] = False
                else:
                    size, duration = pv_spec[index][[0, 2]]
                    prev_size, prev_duration = pv_spec[index-1][[0, 2]]
                    if prev_size == size and duration == prev_duration:
                        mask[index-1] = False

    return mask


def test_signature_significance(pv_spec, concepts, alpha, winlen,
                                corr='fdr_bh', report='spectrum',
                                spectrum='#'):
    """
    Compute the significance spectrum of a pattern spectrum.

    Given pvalue_spectrum `pv_spec` as a list of triplets (z,c,p), where z is
    pattern size, c is pattern support and p is the p-value of the signature
    (z,c), this routine assesses the significance of (z,c) using the
    confidence level alpha.

    Bonferroni or FDR statistical corrections can be applied.

    Parameters
    ----------
    pv_spec : list
        A list of triplets (z,c,p), where z is pattern size, c is pattern
        support and p is the p-value of signature (z,c)
    concepts : list of tuple
        Output of the concepts mining for the original data.
    alpha : float
        Significance level of the statistical test
    winlen : int
        Size (number of bins) of the sliding window used for the analysis
    corr : str, optional
        Method used for testing and adjustment of pvalues.
        Can be either the full name or initial letters.
        Available methods are:

        'bonferroni' : one-step correction

        'sidak' : one-step correction

        'holm-sidak' : step down method using Sidak adjustments

        'holm' : step-down method using Bonferroni adjustments

        'simes-hochberg' : step-up method (independent)

        'hommel' : closed method based on Simes tests (non-negative)

        'fdr_bh' : Benjamini/Hochberg (non-negative)

        'fdr_by' : Benjamini/Yekutieli (negative)

        'fdr_tsbh' : two stage fdr correction (non-negative)

        'fdr_tsbky' : two stage fdr correction (non-negative)

        '' or 'no': no statistical correction

        For further description see:
        https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
        Default: 'fdr_bh'

    report : {'spectrum', 'significant', 'non_significant'}, optional
        Format to be returned for the significance spectrum:

        'spectrum': list of triplets (z,c,b), where b is a boolean specifying
                    whether signature (z,c) is significant (True) or not
                    (False)

        'significant': list containing only the significant signatures (z,c) of
                       pvalue_spectrum

        'non_significant': list containing only the non-significant signatures

    spectrum : {'#', '3d#'}, optional
        Defines the signature of the patterns.

        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)

        Default: '#'

    Returns
    -------
    sig_spectrum : list
        Significant signatures of pvalue_spectrum, in the format specified
        by `report`
    """
    # If alpha == 1 all signatures are significant
    if alpha == 1:
        return []

    if spectrum not in ('#', '3d#'):
        raise ValueError("spectrum must be either '#' or '3d#', "
                         "got {} instead".format(spectrum))
    if report not in ('spectrum', 'significant', 'non_significant'):
        raise ValueError("report must be either 'spectrum'," +
                         "  'significant' or 'non_significant'," +
                         "got {} instead".format(report))
    if corr not in ('bonferroni', 'sidak', 'holm-sidak', 'holm',
                    'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                    'fdr_tsbh', 'fdr_tsbky', '', 'no'):
        raise ValueError("Parameter corr not recognized")

    pv_spec = np.array(pv_spec)
    mask = _mask_pvalue_spectrum(pv_spec, concepts, spectrum, winlen)
    pvalues = pv_spec[:, -1]

    pvalues_totest = pvalues[mask]

    # Initialize test array to False
    tests = [False] * len(pvalues)

    if len(pvalues_totest) > 0:

        # Compute significance for only the non trivial tests
        if corr in ['', 'no']:  # ...without statistical correction
            tests_selected = pvalues_totest <= alpha
        else:
            try:
                import statsmodels.stats.multitest as sm
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please run 'pip install statsmodels' if you "
                    "want to use multiple testing correction")

            tests_selected = sm.multipletests(pvalues_totest, alpha=alpha,
                                              method=corr)[0]

        # assign each corrected pvalue to its corresponding entry
        # this breaks
        for index, value in zip(mask.nonzero()[0], tests_selected):
            tests[index] = value

    # Return the specified results:
    if spectrum == '#':
        if report == 'spectrum':
            sig_spectrum = [(size, occ, test)
                            for (size, occ, pv), test in zip(pv_spec, tests)]
        elif report == 'significant':
            sig_spectrum = [(size, occ) for ((size, occ, pv), test)
                            in zip(pv_spec, tests) if test]
        else:  # report == 'non_significant'
            sig_spectrum = [(size, occ)
                            for ((size, occ, pv), test) in zip(pv_spec, tests)
                            if not test]

    else:  # spectrum == '3d#'
        if report == 'spectrum':
            sig_spectrum =\
                [(size, occ, l, test)
                 for (size, occ, l, pv), test in zip(pv_spec, tests)]
        elif report == 'significant':
            sig_spectrum = [(size, occ, l) for ((size, occ, l, pv), test)
                            in zip(pv_spec, tests) if test]
        else:  # report == 'non_significant'
            sig_spectrum =\
                [(size, occ, l)
                 for ((size, occ, l, pv), test) in zip(pv_spec, tests)
                 if not test]
    return sig_spectrum


def _pattern_spectrum_filter(concept, ns_signatures, spectrum, winlen):
    """
    Filter for significant concepts
    """
    if spectrum == '#':
        keep_concept = (len(concept[0]), len(concept[1])) not in ns_signatures
    else:   # spectrum == '3d#':
        # duration is fixed as the maximum lag
        duration = max(np.array(concept[0]) % winlen)
        keep_concept = (len(concept[0]), len(concept[1]),
                        duration) not in ns_signatures
    return keep_concept


def approximate_stability(concepts, rel_matrix, n_subsets=0,
                          delta=0., epsilon=0.):
    r"""
    Approximate the stability of concepts. Uses the algorithm described
    in Babin, Kuznetsov (2012): Approximating Concept Stability

    Parameters
    ----------
    concepts : list
        All the pattern candidates (concepts) found in the spiketrains. Each
        pattern is represented as a tuple containing (spike IDs,
        discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        `spike_id=neuron_id*bin_id` with `neuron_id` in `[0, len(spiketrains)]`
        and `bin_id` in `[0, winlen]`.
    rel_matrix : sparse.coo_matrix
        A binary matrix with shape (number of windows,
        winlen*len(spiketrains)). Each row corresponds to a window (order
        according to their position in time).
        Each column corresponds to one bin and one neuron and it is 0 if
        no spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron. For example, the entry [0,0] of this matrix
        corresponds to the first bin of the first window position for the first
        neuron, the entry `[0, winlen]` to the first bin of the first window
        position for the second neuron.
    n_subsets : int
        Number of subsets of a concept used to approximate its stability.
        If `n_subsets` is 0, it is calculated according to to the formula
        given in Babin, Kuznetsov (2012), proposition 6:

        .. math::
               n_{\text{subset}} = \frac{1}{2 \cdot \epsilon^2}
                \ln{\left( \frac{2}{\delta} \right)} +1

        Default: 0
    delta : float, optional
        delta: probability with at least :math:`1-\delta`
        Default: 0.0
    epsilon : float, optional
        epsilon: absolute error
        Default: 0.0

    Returns
    -------
    output : list
        List of all the pattern candidates (concepts) given in input, each with
        the correspondent intensional and extensional stability. Each
        pattern is represented as a tuple (spike IDs,
        discrete times of the  occurrences of the pattern, intensional
        stability of the pattern, extensional stability of the pattern).
        The spike IDs are defined as:
        `spike_id=neuron_id*bin_id` with `neuron_id` in `[0, len(spiketrains)]`
        and `bin_id` in `[0, winlen]`.

    Notes
    -----
    If n_subset is larger than the extent all subsets are directly
    calculated, otherwise for small extent size an infinite
    loop can be created while doing the recursion,
    since the random generation will always contain the same
    numbers and the algorithm will be stuck searching for
    other (random) numbers.

    """
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
        size = comm.Get_size()  # get tot number of MPI tasks
    else:
        rank = 0
        size = 1
    if not (isinstance(n_subsets, int) and n_subsets >= 0):
        raise ValueError('n_subsets must be an integer >=0')
    if n_subsets == 0 and not (isinstance(delta, float) and delta > 0. and
                               isinstance(epsilon, float) and epsilon > 0.):
        raise ValueError('delta and epsilon must be floats > 0., '
                         'given that n_subsets = 0')

    if len(concepts) == 0:
        return []
    if len(concepts) <= size:
        rank_idx = [0] * (size + 1) + [len(concepts)]
    else:
        rank_idx = list(
            range(0, len(concepts) - len(concepts) % size + 1,
                  len(concepts) // size)) + [len(concepts)]
    # Calculate optimal n
    if n_subsets == 0:
        n_subsets = int(round(np.log(2. / delta) / (2 * epsilon ** 2) + 1))

    if rank == 0:
        concepts_on_partition = concepts[rank_idx[rank]:rank_idx[rank + 1]] + \
            concepts[rank_idx[-2]:rank_idx[-1]]
    else:
        concepts_on_partition = concepts[rank_idx[rank]:rank_idx[rank + 1]]

    output = []
    for concept in concepts_on_partition:
        intent, extent = np.array(concept[0]), np.array(concept[1])
        stab_int = _calculate_single_stability_parameter(
            intent, extent, n_subsets, rel_matrix, look_at='intent')
        stab_ext = _calculate_single_stability_parameter(
            intent, extent, n_subsets, rel_matrix, look_at='extent')
        output.append((intent, extent, stab_int, stab_ext))

    if size != 1:
        recv_list = comm.gather(output, root=0)
        if rank == 0:
            for i in range(1, len(recv_list)):
                output.extend(recv_list[i])

    return output


def _calculate_single_stability_parameter(intent, extent,
                                          n_subsets, rel_matrix,
                                          look_at='intent'):
    """
    Calculates the stability parameter for extent or intent.

    For detailed describtion see approximate_stabilty

    Parameters
    ----------
    extent : np.array
        2nd element of concept
    intent : np.array
        1st element of concept
    n_subsets : int
        See approximate_stabilty
    rel_matrix : sparse.coo_matrix
        See approximate_stabilty
    look_at : {'extent', 'intent'}
        whether to determine stability for extent or intent.
        Default: 'intent'

    Returns
    -------
    stability : float
        Stability parameter for given extent, intent depending on which to look
    """
    if look_at == 'intent':
        element_1, element_2 = intent, extent
    else:  # look_at == 'extent':
        element_1, element_2 = extent, intent

    if n_subsets > 2 ** len(element_1):
        subsets = chain.from_iterable(
            combinations(element_1, subset_index)
            for subset_index in range(len(element_1) + 1))
    else:
        subsets = _select_random_subsets(element_1, n_subsets)

    stability = 0
    excluded_subsets = []
    for subset in subsets:
        if any([set(subset).issubset(excluded_subset)
                for excluded_subset in excluded_subsets]):
            continue

        # computation of the ' operator for the subset
        if look_at == 'intent':
            subset_prime = \
                np.where(np.all(rel_matrix[:, subset], axis=1) == 1)[0]
        else:  # look_at == 'extent':
            subset_prime = \
                np.where(np.all(rel_matrix[subset, :], axis=0) == 1)[0]

        # Condition holds if the closure of the subset of element_1 given in
        # element_2 is equal to element_2 given in input
        if set(subset_prime) == set(element_2):
            stability += 1
        else:
            excluded_subsets.append(subset)
    stability /= min(n_subsets, 2 ** len(element_1))
    return stability


def _select_random_subsets(element_1, n_subsets):
    """
    Creates a list of random_subsets of element_1.

    Parameters
    ----------
    element_1 : np.array
        intent or extent
    n_subsets : int
        see approximate_stability

    Returns
    -------
    subsets : list
        each element a subset of element_1
    """
    subsets_indices = [set()] * (len(element_1) + 1)
    subsets = []

    while len(subsets) < n_subsets:
        num_indices = np.random.binomial(n=len(element_1), p=1 / 2)
        random_indices = sorted(np.random.choice(
            len(element_1), size=num_indices, replace=False))

        random_tuple = tuple(random_indices)
        if random_tuple not in subsets_indices[num_indices]:
            subsets_indices[num_indices].add(random_tuple)
            subsets.append(element_1[random_indices])

    return subsets


def pattern_set_reduction(concepts, ns_signatures, winlen, spectrum,
                          h_subset_filtering=0, k_superset_filtering=0,
                          l_covered_spikes=0, min_spikes=2, min_occ=2):
    r"""
    Takes a list concepts and performs pattern set reduction (PSR).

    PSR determines which patterns in concepts_psf are statistically significant
    given any other pattern, on the basis of the pattern size and
    occurrence count ("support"). Only significant patterns are retained.
    The significance of a pattern A is evaluated through its signature
    :math:`(z_a, c_A)`, where :math:`z_A = |A|` is the size and :math:`c_A` -
    the support of A, by either of:

    * subset filtering: any pattern B is discarded if *concepts* contains a
      superset A of B such that
      :math:`(z_B, c_B - c_A + h) \in \text{ns}_{\text{signatures}}`
    * superset filtering: any pattern A is discarded if *concepts* contains a
      subset B of A such that
      :math:`(z_A - z_B + k, c_A) \in \text{ns}_{\text{signatures}}`
    * covered-spikes criterion: for any two patterns A, B with
      :math:`A \subset B`, B is discarded if
      :math:`(z_B-l) \cdot c_B \le c_A \cdot (z_A - l)`, A is discarded
      otherwise;
    * combined filtering: combines the three procedures above:
      takes a list concepts (see output psf function) and performs
      combined filtering based on the signature (z, c) of each pattern, where
      z is the pattern size and c the pattern support.

    For any two patterns A and B in concepts_psf such that :math:`B \subset A`,
    check:

    1) :math:`(z_B, c_B - c_A + h) \in \text{ns}_{\text{signatures}}`, and

    2) :math:`(z_A - z_B + k, c_A) \in \text{ns}_{\text{signatures}}`.

    Then:

    * if 1) and not 2): discard B
    * if 2) and not 1): discard A
    * if 1) and 2): discard B if
                    :math:`c_B \cdot (z_B - l) \le c_A \cdot (z_A - l)`,
                    otherwise discard A
    * if neither 1) nor 2): keep both patterns

    Assumptions/Approximations:

        * a pair of concepts cannot cause one another to be rejected
        * if two concepts overlap more than min_occ times, one of them can
          account for all occurrences of the other one if it passes the
          filtering

    Parameters
    ----------
    concepts : list
        List of concepts, each consisting in its intent and extent
    ns_signatures : list
        A list of non-significant pattern signatures (z, c)
    winlen : int
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by `winlen*bin_size`.
    spectrum : {'#', '3d#'}
        Define the signature of the patterns.

        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrences)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)
    h_subset_filtering : int, optional
        Correction parameter for subset filtering
        Default: 0
    k_superset_filtering : int, optional
        Correction parameter for superset filtering
        Default: 0
    l_covered_spikes : int, optional
        Correction parameter for covered-spikes criterion
        Default: 0
    min_spikes : int, optional
        Minimum pattern size
        Default: 2
    min_occ : int, optional
        Minimum number of pattern occurrences
        Default: 2

    Returns
    -------
    tuple
        A tuple containing the elements of the input argument
        that are significant according to combined filtering.
    """
    additional_measures = []
    # Extracting from the extent and intent the spike and window times
    for concept in concepts:
        intent = concept[0]
        extent = concept[1]
        additional_measures.append((len(extent), len(intent)))

    # by default, select all elements in conc to be returned in the output
    selected = [True] * len(concepts)
    # scan all conc and their subsets
    for id1, id2 in combinations(range(len(concepts)), r=2):
        # immediately continue if both concepts have already been rejected
        if not selected[id1] and not selected[id2]:
            continue

        intent1, extent1 = concepts[id1][:2]
        intent2, extent2 = concepts[id2][:2]
        occ1, size1 = additional_measures[id1]
        occ2, size2 = additional_measures[id2]
        dur1 = max(np.array(intent1) % winlen)
        dur2 = max(np.array(intent2) % winlen)
        intent2 = set(intent2)

        # Collecting all the possible distances between the windows
        # of the two concepts
        time_diff_all = np.array(
            [w2 - w1 for w2 in extent2 for w1 in extent1])
        # sort time differences by ascending absolute value
        time_diff_sorting = np.argsort(np.abs(time_diff_all))
        sorted_time_diff, sorted_time_diff_occ = np.unique(
            time_diff_all[time_diff_sorting],
            return_counts=True)
        # only consider time differences that are smaller than winlen
        # and that correspond to intersections that occur at least min_occ
        # times
        time_diff_mask = np.logical_and(
            np.abs(sorted_time_diff) < winlen,
            sorted_time_diff_occ >= min_occ)
        # Rescaling the spike times to realign to real time
        for time_diff in sorted_time_diff[time_diff_mask]:
            intent1_new = [t_old - time_diff for t_old in intent1]
            # from here on we will only need the intents as sets
            intent1_new = set(intent1_new)
            # if intent1 and intent2 are disjoint, skip this step
            if len(intent1_new & intent2) == 0:
                continue
            # Test the case intent1 is a superset of intent2
            if intent1_new.issuperset(intent2):
                reject1, reject2 = _perform_combined_filtering(
                    occ_superset=occ1,
                    size_superset=size1,
                    dur_superset=dur1,
                    occ_subset=occ2,
                    size_subset=size2,
                    dur_subset=dur2,
                    spectrum=spectrum,
                    ns_signatures=ns_signatures,
                    h_subset_filtering=h_subset_filtering,
                    k_superset_filtering=k_superset_filtering,
                    l_covered_spikes=l_covered_spikes,
                    min_spikes=min_spikes,
                    min_occ=min_occ)

            elif intent2.issuperset(intent1_new):
                reject2, reject1 = _perform_combined_filtering(
                    occ_superset=occ2,
                    size_superset=size2,
                    dur_superset=dur2,
                    occ_subset=occ1,
                    size_subset=size1,
                    dur_subset=dur1,
                    spectrum=spectrum,
                    ns_signatures=ns_signatures,
                    h_subset_filtering=h_subset_filtering,
                    k_superset_filtering=k_superset_filtering,
                    l_covered_spikes=l_covered_spikes,
                    min_spikes=min_spikes,
                    min_occ=min_occ)

            else:
                # none of the intents is a superset of the other one
                # we compare both concepts to the intersection
                # if one of them is not significant given the
                # intersection, it is rejected
                inter_size = len(intent1_new & intent2)
                reject1 = _superset_filter(
                    occ_superset=occ1,
                    size_superset=size1,
                    dur_superset=dur1,
                    size_subset=inter_size,
                    spectrum=spectrum,
                    ns_signatures=ns_signatures,
                    k_superset_filtering=k_superset_filtering,
                    min_spikes=min_spikes)
                reject2 = _superset_filter(
                    occ_superset=occ2,
                    size_superset=size2,
                    dur_superset=dur2,
                    size_subset=inter_size,
                    spectrum=spectrum,
                    ns_signatures=ns_signatures,
                    k_superset_filtering=k_superset_filtering,
                    min_spikes=min_spikes)
                # Reject accordingly:
                if reject1 and reject2:
                    reject1, reject2 = _covered_spikes_criterion(
                        occ_superset=occ1,
                        size_superset=size1,
                        occ_subset=occ2,
                        size_subset=size2,
                        l_covered_spikes=l_covered_spikes)

            selected[id1] &= not reject1
            selected[id2] &= not reject2

            # skip remaining time-shifts if both concepts have been rejected
            if (not selected[id1]) and (not selected[id2]):
                break

    # Return the selected concepts
    return [p for i, p in enumerate(concepts) if selected[i]]


def _perform_combined_filtering(occ_superset,
                                size_superset,
                                dur_superset,
                                occ_subset,
                                size_subset,
                                dur_subset,
                                spectrum,
                                ns_signatures,
                                h_subset_filtering,
                                k_superset_filtering,
                                l_covered_spikes,
                                min_spikes,
                                min_occ):
    """
    perform combined filtering
    (see pattern_set_reduction)
    """
    reject_subset = _subset_filter(
        occ_superset=occ_superset,
        occ_subset=occ_subset,
        size_subset=size_subset,
        dur_subset=dur_subset,
        spectrum=spectrum,
        ns_signatures=ns_signatures,
        h_subset_filtering=h_subset_filtering,
        min_occ=min_occ)
    reject_superset = _superset_filter(
        occ_superset=occ_superset,
        size_superset=size_superset,
        dur_superset=dur_superset,
        size_subset=size_subset,
        spectrum=spectrum,
        ns_signatures=ns_signatures,
        k_superset_filtering=k_superset_filtering,
        min_spikes=min_spikes)
    # Reject the superset and/or the subset accordingly:
    if reject_superset and reject_subset:
        reject_superset, reject_subset = _covered_spikes_criterion(
            occ_superset=occ_superset,
            size_superset=size_superset,
            occ_subset=occ_subset,
            size_subset=size_subset,
            l_covered_spikes=l_covered_spikes)
    return reject_superset, reject_subset


def _subset_filter(occ_superset, occ_subset, size_subset, dur_subset, spectrum,
                   ns_signatures=None, h_subset_filtering=0, min_occ=2):
    """
    perform subset filtering
    (see pattern_set_reduction)
    """
    if ns_signatures is None:
        ns_signatures = []
    occ_diff = occ_subset - occ_superset + h_subset_filtering
    if spectrum == '#':
        signature_to_test = (size_subset, occ_diff)
    else:  # spectrum == '3d#':
        signature_to_test = (size_subset, occ_diff, dur_subset)
    reject_subset = occ_diff < min_occ or signature_to_test in ns_signatures
    return reject_subset


def _superset_filter(occ_superset, size_superset, dur_superset, size_subset,
                     spectrum, ns_signatures=None, k_superset_filtering=0,
                     min_spikes=2):
    """
    perform superset filtering
    (see pattern_set_reduction)
    """
    if ns_signatures is None:
        ns_signatures = []
    size_diff = size_superset - size_subset + k_superset_filtering
    if spectrum == '#':
        signature_to_test = (size_diff, occ_superset)
    else:  # spectrum == '3d#':
        signature_to_test = (size_diff, occ_superset, dur_superset)
    reject_superset = \
        size_diff < min_spikes or signature_to_test in ns_signatures
    return reject_superset


def _covered_spikes_criterion(occ_superset,
                              size_superset,
                              occ_subset,
                              size_subset,
                              l_covered_spikes):
    """
    evaluate covered spikes criterion
    (see pattern_set_reduction)
    """
    reject_superset = True
    reject_subset = True
    score_superset = (size_superset - l_covered_spikes) * occ_superset
    score_subset = (size_subset - l_covered_spikes) * occ_subset
    if score_superset >= score_subset:
        reject_superset = False
    else:
        reject_subset = False
    return reject_superset, reject_subset


@deprecated_alias(binsize='bin_size')
def concept_output_to_patterns(concepts, winlen, bin_size, pv_spec=None,
                               spectrum='#', t_start=0 * pq.ms):
    """
    Construction of dictionaries containing all the information about a pattern
    starting from a list of concepts and its associated pvalue_spectrum.

    Parameters
    ----------
    concepts : tuple
        Each element of the tuple corresponds to a pattern which it turn is a
        tuple of (spikes in the pattern, occurrences of the patterns)
    winlen : int
        Length (in bins) of the sliding window used for the analysis.
    bin_size : pq.Quantity
        The time precision used to discretize the `spiketrains` (binning).
    pv_spec : None or tuple
        Contains a tuple of signatures and the corresponding p-value. If equal
        to None all p-values are set to -1.
    spectrum : {'#', '3d#'}
        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrences)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)

        Default: '#'
    t_start : pq.Quantity
        t_start of the analyzed spike trains

    Returns
    -------
    output : list
        List of dictionaries. Each dictionary corresponds to a pattern and
        has the following entries:

        'itemset':
            A list of the spikes in the pattern, expressed in theform of
            itemset, each spike is encoded by
            `spiketrain_id * winlen + bin_id`.
        'windows_ids':
            The ids of the windows in which the pattern occurred
            in discretized time (given byt the binning).
        'neurons':
            An array containing the idx of the neurons of the pattern.
        'lags':
            An array containing the lags (integers corresponding to the
            number of bins) between the spikes of the patterns. The first
            lag is always assumed to be 0 and corresponds to the first
            spike.
        'times':
            An array containing the times (integers corresponding to the
            bin idx) of the occurrences of the patterns.
        'signature':
            A tuple containing two integers (number of spikes of the
            patterns, number of occurrences of the pattern).
        'pvalue':
            The p-value corresponding to the pattern. If `n_surr==0`,
            all p-values are set to -1.

    """
    if pv_spec is not None:
        pvalue_dict = defaultdict(float)
        # Creating a dictionary for the pvalue spectrum
        for entry in pv_spec:
            if spectrum == '3d#':
                pvalue_dict[(entry[0], entry[1], entry[2])] = entry[-1]
            if spectrum == '#':
                pvalue_dict[(entry[0], entry[1])] = entry[-1]
    # Initializing list containing all the patterns
    t_start = t_start.rescale(bin_size.units)
    output = []
    for concept in concepts:
        itemset, window_ids = concept[:2]
        # Vocabulary for each of the patterns, containing:
        # - The pattern expressed in form of Itemset, each spike in the pattern
        # is represented as spiketrain_id * winlen + bin_id
        # - The ids of the windows in which the pattern occurred in discretized
        # time (clipping)
        output_dict = {'itemset': itemset, 'windows_ids': window_ids}
        # Bins relative to the sliding window in which the spikes of patt fall
        itemset = np.array(itemset)
        bin_ids_unsort = itemset % winlen
        order_bin_ids = np.argsort(bin_ids_unsort)
        bin_ids = bin_ids_unsort[order_bin_ids]
        # id of the neurons forming the pattern
        output_dict['neurons'] = list(itemset[order_bin_ids] // winlen)
        # Lags (in bin_sizes units) of the pattern
        output_dict['lags'] = bin_ids[1:] * bin_size
        # Times (in bin_size units) in which the pattern occurs
        output_dict['times'] = sorted(window_ids) * bin_size + t_start

        # pattern dictionary appended to the output
        if spectrum == '#':
            # Signature (size, n occ) of the pattern
            signature = (len(itemset), len(window_ids))
        else:  # spectrum == '3d#':
            # Signature (size, n occ, duration) of the pattern
            # duration is position of the last bin
            signature = (len(itemset), len(window_ids), bin_ids[-1])

        output_dict['signature'] = signature
        # If None is given in input to the pval spectrum the pvalue
        # is set to -1 (pvalue spectrum not available)
        if pv_spec is None:
            output_dict['pvalue'] = -1
        else:
            # p-value assigned to the pattern from the pvalue spectrum
            output_dict['pvalue'] = pvalue_dict[signature]

        output.append(output_dict)
    return output
