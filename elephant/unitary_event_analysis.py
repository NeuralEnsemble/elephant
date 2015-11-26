
"""
Unitary Event (UE) analysis is a statistical method that
 enables to analyze in a time resolved manner excess spike correlation
 between simultaneously recorded neurons by comparing the empirical
 spike coincidences (precision of a few ms) to the expected number 
 based on the firing rates of the neurons.

References:
-----------
Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods, 94(1): 67-79.
Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial nonstationarity 
on joint-spike events Biological Cybernetics 88(5):335-351. 
Gruen S (2009) Data-driven significance estimation of precise spike correlation. 
J Neurophysiology 101:1126-1140 (invited review)
"""





import numpy as np
import quantities as pq
import neo
import warnings
import elephant.conversion as conv
import scipy


def hash_from_pattern(m, N, base=2):
    """
    Calculate for a spike pattern or a matrix of spike patterns
    (provide each pattern as a column) composed of N neurons a
    unique number.


    Parameters:
    -----------
    m: list of integers
           matrix of 0-1 patterns as columns,
           shape: (number of neurons, number of patterns)
    N: integer
           number of neurons is required to be equal to the number
           of rows
    base: integer
           base for calculation of the number from binary
           sequences (= pattern).
           Default is 2

    Returns:
    --------
    list of integers:
           An array containing the hash values of each pattern,
           shape: (number of patterns)

    Raises:
    -------
       ValueError: if matrix m has wrong orientation

    Examples:
    ---------
    descriptive example:
    m = [0
         1
         1]
    N = 3
    base = 2
    hash = 0*2^2 + 1*2^1 + 1*2^0 = 3

    second example:
    >>> import numpy as np
    >>> m = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                         [0, 0, 1, 0, 1, 0, 1, 1],
                         [0, 0, 0, 1, 0, 1, 1, 1]])

    >>> hash_from_pattern(m,N=3)
        array([0, 4, 2, 1, 6, 5, 3, 7])
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(m)[0]:
        raise ValueError('patterns in the matrix should be column entries')

    # generate the representation for binary system
    v = np.array([base**x for x in range(N)])
    # reverse the order
    v = v[np.argsort(-v)]
    # calculate the binary number by use of scalar product
    return np.dot(v,m)


def inverse_hash_from_pattern(h, N, base=2):
    """
    Calculate the 0-1 spike patterns (matrix) from hash values

    Parameters:
    -----------
    h: list of integers
           list or array of hash values, length: number of patterns
    N: integer
           number of neurons
    base: integer
           base for calculation of the number from binary
           sequences (= pattern).
           Default is 2

    Raises:
    -------
       ValueError: if the hash is not compatible with the number
       of neurons hash value should not be larger than the biggest
       possible hash number with given number of neurons
       (e.g. for N = 2, max(hash) = 2^1 + 2^0 = 3
            , or for N = 4, max(hash) = 2^3 + 2^2 + 2^1 + 2^0 = 15)

    Returns:
    --------
       numpy.array:
           A matrix of shape: (N, number of patterns)

    Examples
    ---------
    >>> import numpy as np
    >>> h = np.array([3,7])
    >>> N = 4
    >>> inverse_hash_from_pattern(h,N)
        array([[1, 1],
               [1, 1],
               [0, 1],
               [0, 0]])
    """

    # check if the hash values are not bigger than possible hash value
    # for N neuron with basis = base
    if np.any(h > np.sum([base**x for x in range(N)])):
        raise ValueError(
            "hash value is not compatible with the number of neurons N")
    # check if the hash values are integer
    if not np.all(np.int64(h) == h):
        raise ValueError("hash values are not integers")

    m = np.zeros((N,len(h)), dtype=int)
    for j, hh in enumerate(h):
        i = N-1
        while i >= 0 and hh != 0:
            m[i, j] = hh % base
            hh /= base
            i -= 1
    return m


def n_emp_mat(mat, N, pattern_hash, base=2):
    """
    Calculates empirical number of observed patterns expressed
    by their hash values

    Parameters:
    -----------
    m: list of integers
           matrix of 0-1 patterns as columns,
           shape: (number of neurons N, number of patterns)
    N: integer
           number of neurons
    pattern_hash: list of integers
            array of hash values. Length defines number of patterns
    base: integer
           base for calculation of the number from binary
           sequences (= pattern).
           Default is 2

    Returns:
    --------
    N_emp: list of integers
           empirical number of each observed pattern.
           Same length as pattern_hash
    indices: list of list of integers
           list of indices of mat per entry of pattern_hash.
           indices[i] = N_emp[i] = pattern_hash[i]

    Raises:
    -------
       ValueError: if mat is not zero-one matrix

    Examples:
    ---------
    >>> mat = np.array([[1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 0]])
    >>> pattern_hash = np.array([1,3])
    >>> n_emp, n_emp_indices = N_emp_mat(mat, N,pattern_hash)
    >>> print n_emp
    [ 0.  2.]
    >>> print n_emp_indices
    [array([]), array([0, 3])]
    """
    # check if the mat is zero-one matrix
    if np.any(mat>1) or np.any(mat<0):
        raise ValueError("entries of mat should be either one or zero")
    h = hash_from_pattern(mat, N, base = base)
    N_emp = np.zeros(len(pattern_hash))
    indices = []
    for p_h_idx, p_h in enumerate(pattern_hash):
        indices_tmp = np.nonzero(h == p_h)[0]
        indices.append(indices_tmp)
        N_emp_tmp = len(indices_tmp)
        N_emp[p_h_idx] = N_emp_tmp
    return N_emp, indices


def n_emp_mat_sum_trial(mat, N, pattern_hash, method='analytic_TrialByTrial'):
    """
    Calculates empirical number of observed patterns summed across trials

    Parameters:
    -----------
    mat: 3d numpy array
            the entries are zero or one
            0-axis --> trials
            1-axis --> neurons
            2-axis --> time bins
    N: integer
            number of neurons
    pattern_hash: list of integers
            array of hash values, length: number of patterns
    method: string
            method with which the unitary events whould be computed
            'analytic_TrialByTrial' -- > calculate the expectency
            (analytically) on each trial, then sum over all trials.
            'analytic_TrialAverage' -- > calculate the expectency
            by averaging over trials.
            (cf. Gruen et al. 2003)
            'surrogate_TrialByTrial' -- > calculate the distribution 
            of expected coincidences by spike time randomzation in 
            each trial and sum over trials.
            Default is 'analytic_trialByTrial'



    Returns:
    --------
    N_empL list of integers
           empirical number of observed pattern summed across trials,
           length: number of patterns (i.e. len(patter_hash))
    idx_trials: list of list of integers
           list of indices of mat for each trial in which
           the specific pattern has been observed.
           0-axis --> trial
           1-axis --> list of indices for the chosen trial per
           entry of pattern_hash

    Raises:
    -------
       ValueError: if matrix mat has wrong orientation
       ValueError: if mat is not zero-one matrix

    Examples:
    ---------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([4,6])
    >>> N = 3
    >>> n_emp_sum_trial, n_emp_sum_trial_idx =
                             n_emp_mat_sum_trial(mat, N,pattern_hash)
    >>> n_emp_sum_trial
        array([ 1.,  3.])
    >>> n_emp_sum_trial_idx
        [[array([0]), array([3])], [array([], dtype=int64), array([2, 4])]]
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(mat)[1]:
        raise ValueError('the entries of mat should be a list of a'
                         'list where 0-axis is trials and 1-axis is neurons')

    num_patt = len(pattern_hash)
    N_emp = np.zeros(num_patt)

    idx_trials = []
    for mat_tr in mat:
        # check if the mat is zero-one matrix
        if np.any(np.array(mat_tr)>1):
            raise ValueError("entries of mat should be either one or zero")
        N_emp_tmp,indices_tmp = n_emp_mat(mat_tr, N, pattern_hash,base=2)
        idx_trials.append(indices_tmp)
        N_emp += N_emp_tmp
    if method == 'analytic_TrialByTrial' or method == 'surrogate_TrialByTrial':
        return N_emp, idx_trials
    elif method == 'analytic_TrialAverage':
        return N_emp/float(len(mat)), idx_trials


def _sts_overlap(sts, t_start=None, t_stop=None):
    """
    Find the internal range t_start, t_stop where all spike trains are
    defined; cut all spike trains taking that time range only
    """
    max_tstart = max([t.t_start for t in sts])
    min_tstop = min([t.t_stop for t in sts])

    if t_start is None:
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in sts]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        t_stop = min_tstop
        if not all([min_tstop == t.t_stop for t in sts]):
            warnings.warn(
                "Spiketrains have different t_stop values -- "
                "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in sts]
    return sts_cut


def n_exp_mat(mat, N, pattern_hash, method = 'analytic', **kwargs):
    """
    Calculates the expected joint probability for each spike pattern

    Parameters:
    -----------
    mat: 2d numpy array
            the entries are zero or one
            0-axis --> neurons
            1-axis --> time bins
    pattern_hash: list of integers
            array of hash values, length: number of patterns
    method: string
            method with which the expectency should be caculated
            'analytic' -- > analytically
            'surr' -- > with surrogates (spike time randomization)
            Default is 'analytic'
    kwargs:
    -------
    n_surr: integer
            number of surrogate to be used
            Default is 100

    Raises:
    -------
       ValueError: if matrix m has wrong orientation

    Returns:
    --------
    if method is analytic:
        numpy.array:
           An array containing the expected joint probability of each pattern,
           shape: (number of patterns,)
    if method is surr:
        numpy.ndarray, 0-axis --> different realizations,
                       length = number of surrogates
                       1-axis --> patterns

    Examples:
    ---------
    >>> mat = np.array([[1, 1, 1, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 0]])
    >>> pattern_hash = np.array([5,6])
    >>> N = 3
    >>> n_exp_anal = n_exp_mat(mat,N, pattern_hash, method = 'analytic')
    >>> n_exp_anal
        [ 0.5 1.5 ]
    >>>
    >>>
    >>> n_exp_surr = n_exp_mat(
                  mat, N,pattern_hash, method = 'surr', n_surr = 5000)
    >>> print n_exp_surr
    [[ 1.  1.]
     [ 2.  0.]
     [ 2.  0.]
     ...,
     [ 2.  0.]
     [ 2.  0.]
     [ 1.  1.]]

    """
    # check if the mat is zero-one matrix
    if np.any(mat > 1) or np.any(mat < 0):
        raise ValueError("entries of mat should be either one or zero")

    if method == 'analytic':
        marg_prob = np.mean(mat,1,dtype=float)
        # marg_prob needs to be a column vector, so we
        # build a two dimensional array with 1 column
        # and len(marg_prob) rows
        marg_prob = np.reshape(marg_prob,(len(marg_prob),1))
        m = inverse_hash_from_pattern(pattern_hash, N)
        nrep = np.shape(m)[1]
        # multipyling the marginal probability of neurons with regard to the pattern
        pmat = np.multiply(m,np.tile(marg_prob,(1,nrep))) +\
               np.multiply(1-m,np.tile(1-marg_prob,(1,nrep)))
        return np.prod(pmat,axis=0)*float(np.shape(mat)[1])
    if method == 'surr':
        if len(pattern_hash)>1:
                raise ValueError('surrogate method works only for one pattern!')            
        if 'n_surr' in kwargs:
            n_surr = kwargs['n_surr']
        else:
            n_surr = 100.
        N_exp_array = np.zeros(n_surr)
        for rz_idx, rz in enumerate(np.arange(n_surr)):
            # shuffling all elements of zero-one matrix
            mat_surr = np.array(mat)
            [np.random.shuffle(i) for i in mat_surr]
            N_exp_array[rz_idx] = n_emp_mat(mat_surr, N, pattern_hash)[0][0]
        return N_exp_array


def n_exp_mat_sum_trial(mat, N, pattern_hash, method = 'analytic_TrialByTrial', **kwargs):
    """
    Calculates the expected joint probability
    for each spike pattern sum over trials

    Parameters:
    -----------
    mat: 3d numpy array
            the entries are zero or one
            0-axis --> trials
            1-axis --> neurons
            2-axis --> time bins
    N: integer
           number of neurons
    pattern_hash: list of integers
            array of hash values, length: number of patterns
    method: string
            method with which the unitary events whould be computed
            'analytic_TrialByTrial' -- > calculate the expectency
            (analytically) on each trial, then sum over all trials.
            'analytic_TrialAverage' -- > calculate the expectency
            by averaging over trials.
            (cf. Gruen et al. 2003)
            'surrogate_TrialByTrial' -- > calculate the distribution 
            of expected coincidences by spike time randomzation in 
            each trial and sum over trials.
            Default is 'analytic_trialByTrial'

    kwargs:
    -------
    n_surr: integer
            number of surrogate to be used
            Default is 100

    Returns:
    --------
    numpy.array:
       An array containing the expected joint probability of
       each pattern summed over trials,shape: (number of patterns,)

    Examples:
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([5,6])
    >>> N = 3
    >>> n_exp_anal = n_exp_mat_sum_trial(mat, N, pattern_hash)
    >>> print n_exp_anal
        array([ 1.56,  2.56])
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(mat)[1]:
        raise ValueError('the entries of mat should be a list of a'
                         'list where 0-axis is trials and 1-axis is neurons')

    if method == 'analytic_TrialByTrial':
        n_exp = np.zeros(len(pattern_hash))
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, N, pattern_hash, method='analytic')
    elif method == 'analytic_TrialAverage':
        n_exp = n_exp_mat(
            np.mean(mat,0), N, pattern_hash, method='analytic')*np.shape(mat)[0]
    elif method == 'surrogate_TrialByTrial':
        if 'n_surr' in kwargs: 
            n_surr = kwargs['n_surr']
        else:
            n_surr = 100.
        n_exp = np.zeros(n_surr)
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, N, pattern_hash, method='surr', n_surr = n_surr)
    else:
        raise ValueError(
            "The method only works on the zero_one matrix at the moment")
    return n_exp


def gen_pval_anal(mat, N, pattern_hash, method='analytic_TrialByTrial',**kwargs):
    """
    computes the expected coincidences and a function to calculate
    p-value for given empirical coincidences

    this function generate a poisson distribution with the expected
    value calculated by mat. it returns a function which gets
    the empirical coincidences, `n_emp`,  and calculates a p-value
    as the area under the poisson distribution from `n_emp` to infinity

    Parameters:
    -----------
    mat: 3d numpy array
            the entries are zero or one
            0-axis --> trials
            1-axis --> neurons
            2-axis --> time bins
    N: integer
           number of neurons
    pattern_hash: list of integers
            array of hash values, length: number of patterns
    method: string
            method with which the unitary events whould be computed
            'analytic_TrialByTrial' -- > calculate the expectency
            (analytically) on each trial, then sum over all trials.
            ''analytic_TrialAverage' -- > calculate the expectency
            by averaging over trials.
            Default is 'analytic_trialByTrial'
            (cf. Gruen et al. 2003)

    kwargs:
    -------
    n_surr: integer
            number of surrogate to be used
            Default is 100


    Returns:
    --------
    pval_anal:
            a function which calculates the p-value for
            the given empirical coincidences
    n_exp: list of floats
            expected coincidences

    Examples:
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([5,6])
    >>> N = 3
    >>> pval_anal,n_exp = gen_pval_anal(mat, N,pattern_hash)
    >>> n_exp
        array([ 1.56,  2.56])
    """
    if method == 'analytic_TrialByTrial' or method == 'analytic_TrialAverage':
        n_exp = n_exp_mat_sum_trial(mat, N, pattern_hash, method = method)
        def pval(n_emp):
            p = 1.- scipy.special.gammaincc(n_emp, n_exp)
            return p
    elif method ==  'surrogate_TrialByTrial':
        if 'n_surr' in kwargs: 
            n_surr = kwargs['n_surr'] 
        else:
            n_surr = 100.
        n_exp = n_exp_mat_sum_trial(mat, N, pattern_hash, method = method, n_surr = n_surr)
        def pval(n_emp):
            hist = np.bincount(np.int64(n_exp))
            exp_dist = hist/float(np.sum(hist))
            if len(n_emp)>1:
                raise ValueError('in surrogate method the p_value can be calculated only for one pattern!')
            return np.sum(exp_dist[n_emp[0]:])

    return pval, n_exp



def jointJ(p_val):
    """Surprise measurement

    logarithmic transformation of joint-p-value into surprise measure
    for better visualization as the highly significant events are
    indicated by very low joint-p-values

    Parameters:
    -----------
    p_val: list of floats
        p-values of statistical tests for different pattern.

    Returns:
    --------
    J: list of floats
        list of surprise measure

    Examples:
    ---------
    >>> p_val = np.array([0.31271072,  0.01175031])
    >>> jointJ(p_val)
        array([0.3419968 ,  1.92481736])
    """
    p_arr = np.array(p_val)

    try:
        Js = np.log10(1-p_arr)-np.log10(p_arr)
    except RuntimeWarning:
        pass
    return Js


def _rate_mat_avg_trial(mat):
    """
    calculates the average firing rate of each neurons across trials
    """
    num_tr, N, nbins = np.shape(mat)
    psth = np.zeros(N)
    for tr,mat_tr in enumerate(mat):
        psth += np.sum(mat_tr, axis=1)
    return psth/float(nbins)/float(num_tr)


def _bintime(t, binsize):
    """
    change the real time to bintime
    """
    t_dl = t.rescale('ms').magnitude
    binsize_dl = binsize.rescale('ms').magnitude
    return np.floor(np.array(t_dl)/binsize_dl).astype(int)


def _winpos(t_start, t_stop, winsize, winstep,position='left-edge'):
    """
    Calculates the position of the analysis window
    """
    t_start_dl = t_start.rescale('ms').magnitude
    t_stop_dl = t_stop.rescale('ms').magnitude
    winsize_dl = winsize.rescale('ms').magnitude
    winstep_dl = winstep.rescale('ms').magnitude

    # left side of the window time
    if position == 'left-edge':
        ts_winpos = np.arange(t_start_dl, t_stop_dl - winsize_dl + winstep_dl, winstep_dl)*pq.ms
    else:
        raise ValueError('the current version only returns left-edge of the window')
    return ts_winpos


def _UE(mat, N, pattern_hash, method='analytic_TrialByTrial',**kwargs):
    """
    returns the default results of unitary events analysis
    (Surprise, empirical coincidences and index of where it happened
    in the given mat, n_exp and average rate of neurons)
    """
    rate_avg = _rate_mat_avg_trial(mat)
    n_emp, indices = n_emp_mat_sum_trial(mat, N, pattern_hash, method)
    if method == 'surrogate_TrialByTrial':
        if 'n_surr' in kwargs: 
            n_surr = kwargs['n_surr']
        else: n_surr = 100
        dist_exp, n_exp = gen_pval_anal(mat, N, pattern_hash, method, n_surr=n_surr)
    elif method == 'analytic_TrialByTrial' or method == 'analytic_TrialAverage':
        dist_exp, n_exp = gen_pval_anal(mat, N, pattern_hash, method)
    pval = dist_exp(n_emp)
    Js = jointJ(pval)
    return Js, rate_avg, n_exp, n_emp,indices


def jointJ_window_analysis(
        data, binsize, winsize, winstep, pattern_hash,
        method='analytic_TrialByTrial', t_start=None,
        t_stop=None, binary=True, **kwargs):
    """
    Calculates the joint surprise in a sliding window fashion

    Parameters:
    ----------
    data: list of neo.SpikeTrain objects
          list of spike trains in different trials
                                        0-axis --> Trials
                                        1-axis --> Neurons
                                        2-axis --> Spike times
    binsize: Qunatity scalar with dimension time
           size of bins for descritizing spike trains
    winsize: Qunatity scalar with dimension time
           size of the window of analysis
    winstep: Qunatity scalar with dimension time
           size of the window step
    pattern_hash: list of integers
           list of interested patterns in hash values
           (see hash_from_pattern and inverse_hash_from_pattern functions)
    method: string
            method with which the unitary events whould be computed
            'analytic_TrialByTrial' -- > calculate the expectency
            (analytically) on each trial, then sum over all trials.
            ''analytic_TrialAverage' -- > calculate the expectency
            by averaging over trials.
            (cf. Gruen et al. 2003)
            'surrogate_TrialByTrial' -- > calculate the distribution 
            of expected coincidences by spike time randomzation in 
            each trial and sum over trials.
            Default is 'analytic_trialByTrial'

    kwargs:
    -------
    n_surr: integer
            number of surrogate to be used
            Default is 100


    Returns:
    -------
    result: dictionary
          Js: list of float
                 JointSurprise of different given patterns within each window
                 shape: different pattern hash --> 0-axis
                        different window --> 1-axis
          indices: list of list of integers
                 list of indices of pattern whithin each window
                 shape: different pattern hash --> 0-axis
                        different window --> 1-axis
          n_emp: list of integers
                 empirical number of each observed pattern.
                 shape: different pattern hash --> 0-axis
                        different window --> 1-axis
          n_exp: list of floats
                 expeced number of each pattern.
                 shape: different pattern hash --> 0-axis
                        different window --> 1-axis
          rate_avg: list of floats
                 average firing rate of each neuron
                 shape: different pattern hash --> 0-axis
                        different window --> 1-axis

    """
    if not isinstance(data[0][0],neo.SpikeTrain):
        raise ValueError("structure of the data is not correct: 0-axis should be trials, 1-axis units and 2-axis neo spike trains")

    if t_start is None: t_start = data[0][0].t_start.rescale('ms')
    if t_stop is None: t_stop = data[0][0].t_stop.rescale('ms')

    # position of all windows (left edges)
    t_winpos = _winpos(t_start, t_stop, winsize, winstep,position = 'left-edge')
    t_winpos_bintime = _bintime(t_winpos, binsize)

    winsize_bintime = _bintime(winsize, binsize)
    winstep_bintime = _bintime(winstep, binsize)

    if winsize_bintime * binsize != winsize:
        warnings.warn(
            "ratio between winsize and binsize is not integer -- "
            "the actual number for window size is "+str (winsize_bintime * binsize))

    if winstep_bintime * binsize != winstep:
        warnings.warn(
            "ratio between winsize and binsize is not integer -- "
            "the actual number for window size is" + str(winstep_bintime * binsize))

    num_tr, N = np.shape(data)[:2]

    n_bins = int((t_stop - t_start)/binsize)

    mat_tr_unit_spt = np.zeros((len(data), N, n_bins))
    for tr, sts in enumerate(data):
        bs = conv.BinnedSpikeTrain(sts, t_start=t_start, t_stop=t_stop, binsize=binsize)
        if binary is True:
            mat = bs.to_bool_array()
        else:
            raise ValueError(
                "The method only works on the zero_one matrix at the moment")
        mat_tr_unit_spt[tr] = mat

    num_win = len(t_winpos)
    Js_win, n_exp_win, n_emp_win = (np.zeros(num_win) for _ in xrange(3))
    rate_avg = np.zeros((num_win,N))
    indices_win = {}
    for i in range(num_tr):
        indices_win['trial'+str(i)]= []

    for i, win_pos in enumerate(t_winpos_bintime):
        mat_win = mat_tr_unit_spt[:,:,win_pos:win_pos + winsize_bintime]
        if method == 'surrogate_TrialByTrial':
            if 'n_surr' in kwargs: 
                n_surr = kwargs['n_surr']
            else: n_surr = 100
            Js_win[i], rate_avg[i], n_exp_win[i], n_emp_win[i], indices_lst = _UE(mat_win, N,pattern_hash,method,n_surr=n_surr)
        else:
            Js_win[i], rate_avg[i], n_exp_win[i], n_emp_win[i], indices_lst = _UE(mat_win, N,pattern_hash,method)
        for j in range(num_tr):
            if len(indices_lst[j][0]) > 0:
                indices_win['trial'+str(j)] = np.append(indices_win['trial'+str(j)], indices_lst[j][0] + win_pos)
    return {'Js': Js_win, 'indices':indices_win, 'n_emp': n_emp_win,'n_exp': n_exp_win,'rate_avg':rate_avg/binsize}
