# -*- coding: utf-8 -*-

"""
This algorithm determines if a spike train `spk` can be considered as stationary
process (constant firing rate) or not as stationary process (i.e. presence of
one or more points at which the rate increases or decreases). In case of
non-stationarity, the output is a list of detected Change Points (CPs).
Essentially, a det of  two-sided window of width `h` (`_filter(t, h, spk)`)
slides over the spike train within the time `[h, t_final-h]`. This generates a
`_filter_process(dt, h, spk)` that assigns at each time `t` the difference
between a spike lying in the right and left window. If at any time `t` this
difference is large 'enough' is assumed the presence of a rate Change Point in
a neighborhood of `t`. A threshold `test_quantile` for the maximum of
the filter_process (max difference of spike count between the left and right
window) is derived based on asymptotic considerations. The procedure is
repeated for an arbitrary set of windows, with different size `h`.


Examples
--------
The following applies multiple_filter_test to a spike trains.

    >>> import quantities as pq
    >>> import neo
    >>> from elephant.change_point_detection import multiple_filter_test


    >>> test_array = [1.1,1.2,1.4,   1.6,1.7,1.75,1.8,1.85,1.9,1.95]
    >>> st = neo.SpikeTrain(test_array, units='s', t_stop = 2.1)
    >>> window_size = [0.5]*pq.s
    >>> t_fin = 2.1*pq.s
    >>> alpha = 5.0
    >>> num_surrogates = 10000
    >>> change_points = multiple_filter_test(window_size, st, t_fin, alpha,
                        num_surrogates, dt = 0.5*pq.s)



References
----------
Messer, M., Kirchner, M., Schiemann, J., Roeper, J., Neininger, R., & Schneider,
G. (2014). A multiple filter test for the detection of rate changes in renewal
processes with varying variance. The Annals of Applied Statistics, 8(4),2027-2067.


Original code
-------------
Adapted from the published R implementation:
DOI: 10.1214/14-AOAS782SUPP;.r

"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import quantities as pq


def multiple_filter_test(window_sizes, spiketrain, t_final, alpha,
                         n_surrogates, test_quantile=None, test_param=None,
                         dt=None):
    """
    Detects change points.

    This function returns the detected change points, that correspond to the
    maxima of the `_filter_processes`. These are the processes generated by
    sliding the windows of step `dt`; at each step the difference between spike
    on the right and left window is calculated.

    Parameters
    ----------
        window_sizes : list of quantity objects
                    list that contains windows sizes
        spiketrain : neo.SpikeTrain, numpy array or list
            spiketrain objects to analyze
        t_final : quantity
            final time of the spike train which is to be analysed
        alpha : float
            alpha-quantile in range [0, 100] for the set of maxima of the limit
            processes
        n_surrogates : integer
            numbers of simulated limit processes
        test_quantile : float
            threshold for the maxima of the filter derivative processes, if any
            of these maxima is larger than this value, it is assumed the
            presence of a cp at the time corresponding to that maximum
        dt : quantity
          resolution, time step at which the windows are slided
        test_param : np.array of shape (3, num of window),
            first row: list of `h`, second and third rows: empirical means and
            variances of the limit process correspodning to `h`. This will be
            used to normalize the `filter_process` in order to give to the
            every maximum the same impact on the global statistic.

    Returns
    -------
    cps : list of list
       one list for each window size `h`, containing the points detected with
       the corresponding `filter_process`. N.B.: only cps whose h-neighborhood
       does not include previously detected cps (with smaller window h) are
       added to the list.
    """

    if (test_quantile is None) and (test_param is None):
        test_quantile, test_param = empirical_parameters(window_sizes, t_final,
                                                         alpha, n_surrogates,
                                                         dt)
    elif test_quantile is None:
        test_quantile = empirical_parameters(window_sizes, t_final, alpha,
                                             n_surrogates, dt)[0]
    elif test_param is None:
        test_param = empirical_parameters(window_sizes, t_final, alpha,
                                          n_surrogates, dt)[1]

    spk = spiketrain

    #  List of lists of detected change points (CPs), to be returned
    cps = []

    for i, h in enumerate(window_sizes):
        # automatic setting of dt
        dt_temp = h / 20 if dt is None else dt
        # filter_process for window of size h
        t, differences = _filter_process(dt_temp, h, spk, t_final, test_param)
        time_index = np.arange(len(differences))
        # Point detected with window h
        cps_window = []
        while np.max(differences) > test_quantile:
            cp_index = np.argmax(differences)
            # from index to time
            cp = cp_index * dt_temp + h
            #print("detected point {0}".format(cp), "with filter {0}".format(h))
            # before repeating the procedure, the h-neighbourgs of detected CP
            # are discarded, because rate changes into it are alrady explained
            mask_fore = time_index > cp_index - int((h / dt_temp).simplified)
            mask_back = time_index < cp_index + int((h / dt_temp).simplified)
            differences[mask_fore & mask_back] = 0
            # check if the neighbourhood of detected cp does not contain cps
            # detected with other windows
            neighbourhood_free = True
            # iterate on lists of cps detected with smaller window
            for j in range(i):
                # iterate on CPs detected with the j-th smallest window
                for c_pre in cps[j]:
                    if c_pre - h < cp < c_pre + h:
                        neighbourhood_free = False
                        break
            # if none of the previously detected CPs falls in the h-
            # neighbourhood
            if neighbourhood_free:
                # add the current CP to the list
                cps_window.append(cp)
        # add the present list to the grand list
        cps.append(cps_window)

    return cps


def _brownian_motion(t_in, t_fin, x_in, dt):
    """
    Generate a Brownian Motion.

    Parameters
    ----------
        t_in : quantities,
            initial time
        t_fin : quantities,
             final time
        x_in : float,
            initial point of the process: _brownian_motio(0) = x_in
        dt : quantities,
          resolution, time step at which brownian increments are summed
    Returns
    -------
    Brownian motion on [t_in, t_fin], with resolution dt and initial state x_in
    """

    u = 1 * pq.s
    try:
        t_in_sec = t_in.rescale(u).magnitude
    except ValueError:
        raise ValueError("t_in must be a time quantity")
    try:
        t_fin_sec = t_fin.rescale(u).magnitude
    except ValueError:
        raise ValueError("t_fin must be a time quantity")
    try:
        dt_sec = dt.rescale(u).magnitude
    except ValueError:
        raise ValueError("dt must be a time quantity")

    x = np.random.normal(0, np.sqrt(dt_sec),
                         size=int((t_fin_sec - t_in_sec) / dt_sec))
    s = np.cumsum(x)
    return s + x_in


def _limit_processes(window_sizes, t_final, dt):
    """
    Generate the limit processes (depending only on t_final and h), one for
    each window size `h` in H. The distribution of maxima of these processes
    is used to derive threshold `test_quantile` and parameters `test_param`.

    Parameters
    ----------
        window_sizes : list of quantities
            set of windows' size
        t_final : quantity object
            end of limit process
        dt : quantity object
            resolution, time step at which the windows are slided

    Returns
    -------
        limit_processes : list of numpy array
            each entries contains the limit processes for each h,
            evaluated in [h,T-h] with steps dt
    """

    limit_processes = []

    u = 1 * pq.s
    try:
        window_sizes_sec = window_sizes.rescale(u).magnitude
    except ValueError:
        raise ValueError("window_sizes must be a list of times")
    try:
        dt_sec = dt.rescale(u).magnitude
    except ValueError:
        raise ValueError("dt must be a time quantity")

    w = _brownian_motion(0 * u, t_final, 0, dt)

    for h in window_sizes_sec:
        # BM on [h,T-h], shifted in time t-->t+h
        brownian_right = w[int(2 * h / dt_sec):]
        # BM on [h,T-h], shifted in time t-->t-h
        brownian_left = w[:int(-2 * h / dt_sec)]
        # BM on [h,T-h]
        brownian_center = w[int(h / dt_sec):int(-h / dt_sec)]

        modul = np.abs(brownian_right + brownian_left - 2 * brownian_center)
        limit_process_h = modul / (np.sqrt(2 * h))
        limit_processes.append(limit_process_h)

    return limit_processes


def empirical_parameters(window_sizes, t_final, alpha, n_surrogates, dt=None):
    """
    This function generates the threshold and the null parameters.
    The`_filter_process_h` has been proved to converge (for t_fin, h-->infinity)
    to a continuous functional of a Brownaian motion ('limit_process').
    Using a MonteCarlo technique, maxima of these limit_processes are
    collected.

    The threshold is defined as the alpha quantile of this set of maxima.
    Namely:
    test_quantile := alpha quantile of {max_(h in window_size)[
                                 max_(t in [h, t_final-h])_limit_process_h(t)]}

    Parameters
    ----------
        window_sizes : list of quantity objects
            set of windows' size
        t_final : quantity object
            final time of the spike
        alpha : float
            alpha-quantile in range [0, 100]
        n_surrogates : integer
            numbers of simulated limit processes
        dt : quantity object
            resolution, time step at which the windows are slided

    Returns
    -------
        test_quantile : float
            threshold for the maxima of the filter derivative processes, if any
            of these maxima is larger than this value, it is assumed the
            presence of a cp at the time corresponding to that maximum

        test_param : np.array 3 * num of window,
            first row: list of `h`, second and third rows: empirical means and
            variances of the limit process correspodning to `h`. This will be
            used to normalize the `filter_process` in order to give to the every
            maximum the same impact on the global statistic.
    """

    # try:
    #     window_sizes_sec = window_sizes.rescale(u)
    # except ValueError:
    #     raise ValueError("H must be a list of times")
    # window_sizes_mag = window_sizes_sec.magnitude
    # try:
    #     t_final_sec = t_final.rescale(u)
    # except ValueError:
    #     raise ValueError("T must be a time quantity")
    # t_final_mag = t_final_sec.magnitude

    if not isinstance(window_sizes, pq.Quantity):
        raise ValueError("window_sizes must be a list of time quantities")
    if not isinstance(t_final, pq.Quantity):
        raise ValueError("t_final must be a time quantity")
    if not isinstance(n_surrogates, int):
        raise TypeError("n_surrogates must be an integer")
    if not (isinstance(dt, pq.Quantity) or (dt is None)):
        raise ValueError("dt must be a time quantity")

    if t_final <= 0:
        raise ValueError("t_final needs to be strictly positive")
    if alpha * (100.0 - alpha) < 0:
        raise ValueError("alpha needs to be in (0,100)")
    if np.min(window_sizes) <= 0:
        raise ValueError("window size needs to be strictly positive")
    if np.max(window_sizes) >= t_final / 2:
        raise ValueError("window size too large")
    if dt is not None:
        for h in window_sizes:
            if int(h.rescale('us')) % int(dt.rescale('us')) != 0:
                raise ValueError(
                    "Every window size h must be a multiple of dt")

    # Generate a matrix M*: n X m where n = n_surrogates is the number of
    # simulated limit processes and m is the number of chosen window sizes.
    # Elements are: M*(i,h) = max(t in T)[`limit_process_h`(t)],
    # for each h in H and surrogate i
    maxima_matrix = []

    for i in range(n_surrogates):
        # mh_star = []
        simu = _limit_processes(window_sizes, t_final, dt)
        # for i, h in enumerate(window_sizes_mag):
        #     # max over time of the limit process generated with window h
        #     m_h = np.max(simu[i])
        #     mh_star.append(m_h)
        # max over time of the limit process generated with window h
        mh_star = [np.max(x) for x in simu]
        maxima_matrix.append(mh_star)

    maxima_matrix = np.asanyarray(maxima_matrix)

    # these parameters will be used to normalize both the limit_processes (H0)
    # and the filter_processes
    null_mean = maxima_matrix.mean(axis=0)
    null_var = maxima_matrix.var(axis=0)

    # matrix normalization by mean and variance of the limit process, in order
    # to give, for every h, the same impact on the global maximum
    matrix_normalized = (maxima_matrix - null_mean) / np.sqrt(null_var)

    great_maxs = np.max(matrix_normalized, axis=1)
    test_quantile = np.percentile(great_maxs, 100.0 - alpha)
    null_parameters = [window_sizes, null_mean, null_var]
    test_param = np.asanyarray(null_parameters)

    return test_quantile, test_param


def _filter(t, h, spk):
    """
    This function calculates the difference of spike counts in the left and right
    side of a window of size h centered in t and normalized by its variance.
    The variance of this count can be expressed as a combination of mean and var
    of the I.S.I. lying inside the window.

    Parameters
    ----------
        h : quantity
            window's size
        t : quantity
            time on which the window is centered
        spk : list, numpy array or SpikeTrain
            spike train to analyze

    Returns
    -------
        difference : float,
          difference of spike count normalized by its variance
    """

    u = 1 * pq.s
    try:
        t_sec = t.rescale(u).magnitude
    except AttributeError:
        raise ValueError("t must be a quantities object")
    # tm = t_sec.magnitude
    try:
        h_sec = h.rescale(u).magnitude
    except AttributeError:
        raise ValueError("h must be a time quantity")
    # hm = h_sec.magnitude
    try:
        spk_sec = spk.rescale(u).magnitude
    except AttributeError:
        raise ValueError(
            "spiketrain must be a list (array) of times or a neo spiketrain")

    # cut spike-train on the right
    train_right = spk_sec[(t_sec < spk_sec) & (spk_sec < t_sec + h_sec)]
    # cut spike-train on the left
    train_left = spk_sec[(t_sec - h_sec < spk_sec) & (spk_sec < t_sec)]
    # spike count in the right side
    count_right = train_right.size
    # spike count in the left side
    count_left = train_left.size
    # form spikes to I.S.I
    isi_right = np.diff(train_right)
    isi_left = np.diff(train_left)

    if isi_right.size == 0:
        mu_ri = 0
        sigma_ri = 0
    else:
        # mean of I.S.I inside the window
        mu_ri = np.mean(isi_right)
        # var of I.S.I inside the window
        sigma_ri = np.var(isi_right)

    if isi_left.size == 0:
        mu_le = 0
        sigma_le = 0
    else:
        mu_le = np.mean(isi_left)
        sigma_le = np.var(isi_left)

    if (sigma_le > 0) & (sigma_ri > 0):
        s_quad = (sigma_ri / mu_ri**3) * h_sec + (sigma_le / mu_le**3) * h_sec
    else:
        s_quad = 0

    if s_quad == 0:
        difference = 0
    else:
        difference = (count_right - count_left) / np.sqrt(s_quad)

    return difference


def _filter_process(dt, h, spk, t_final, test_param):
    """
    Given a spike train `spk` and a window size `h`, this function generates
    the `filter derivative process` by evaluating the function `_filter`
    in steps of `dt`.

    Parameters
    ----------
        h : quantity object
         window's size
        t_final : quantity,
           time on which the window is centered
        spk : list, array or SpikeTrain
           spike train to analyze
        dt : quantity object, time step at which the windows are slided
          resolution
        test_param : matrix, the means of the first row list of `h`,
                    the second row Empirical and the third row variances of
                    the limit processes `Lh` are used to normalize the number
                    of elements inside the windows

    Returns
    -------
        time_domain : numpy array
                   time domain of the `filter derivative process`
        filter_process : array,
                      values of the `filter derivative process`
    """

    u = 1 * pq.s

    try:
        h_sec = h.rescale(u).magnitude
    except AttributeError:
        raise ValueError("h must be a time quantity")
    try:
        t_final_sec = t_final.rescale(u).magnitude
    except AttributeError:
        raise ValueError("t_final must be a time quanity")
    try:
        dt_sec = dt.rescale(u).magnitude
    except AttributeError:
        raise ValueError("dt must be a time quantity")
    # domain of the process
    time_domain = np.arange(h_sec, t_final_sec - h_sec, dt_sec)
    filter_trajectrory = []
    # taken from the function used to generate the threshold
    emp_mean_h = test_param[1][test_param[0] == h]
    emp_var_h = test_param[2][test_param[0] == h]

    for t in time_domain:
        filter_trajectrory.append(_filter(t * u, h, spk))

    filter_trajectrory = np.asanyarray(filter_trajectrory)
    # ordered normalization to give each process the same impact on the max
    filter_process = (
        np.abs(filter_trajectrory) - emp_mean_h) / np.sqrt(emp_var_h)

    return time_domain, filter_process
