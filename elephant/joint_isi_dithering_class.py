"""
Module to generate surrogates of spike trains by using the joint-ISI dithering.
Features are provided to separate the preprocessing from the main process.

Original implementation by: Peter Bouss [p.bouss@fz-juelich.de]
:copyright: Copyright 2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
import quantities as pq
import neo

import elephant.spike_train_surrogates as surr
import elephant.statistics as stats
from scipy.ndimage import gaussian_filter


class JointISISpace:
    """
    The class :class:`Joint_ISI_Space` is implemented for Joint-ISI dithering
    as a continuation of the ideas of Louis et al. (2010) and Gerstein (2004).

    When creating an class instance all necessary preprocessing steps are done,
    to use the method dithering().

    To make the dithering-procedure stable, the spiketrain needs to pass two
    thresholds, that the Joint-ISI dithering is applied, if not uniform
    dithering is used.

    Attributes
    ----------
    st: neo.SpikeTrain
        For this spiketrain the surrogates will be created

    n_surr: int, optional
        Number of surrogates to be created.
        Default: 1
    dither: pq.Quantity
        The range of the dithering for the uniform dithering,
        which is also used for the method 'window'.
        Default: 15.*pq.ms
    unit: pq.unit
        The unit of the spiketrain in the output.
        Default: pq.s
    window_length: pq.Quantity
        The Joint-ISI distribution is as such defined on a range for ISI_i and
        ISI_(i+1) from 0 to inf. Since this is computationally not feasible,
        the Joint-ISI distribution is truncated for high ISI. The Joint-ISI
        histogram is calculated for ISI_i, ISI_(i+1) from 0 to window_length.
        Default: 120*pq.ms
    num_bins: int
        The size of the joint-ISI-distribution will be num_bins*num_bins.
        Default: 120
    sigma: pq.Quantity
        The standard deviation of the Gaussian kernel, with which
        the data is convoluted.
        Default: 0.001*pq.s
    isi_median_threshold: pq.Quantity
        Only if the median of the ISI distribution is smaller than
        isi_median_threshold the Joint-ISI dithering is applied, if not the
        uniform dithering is used.
        Default: 30*pq.ms
    alternate: boolean
        If alternate == True: then first all even and then all odd spikes are
        dithered. Else: in ascending order from the first to the last spike,
        all spikes are moved.
        Default: True.
    print_mode: boolean
        If True, also the way of how the dithered spikes are evaluated
        is returned so 'uniform' for uniform and dithering and 'jisid' for
        joint-ISI-dithering
        Default: False
    use_sqrt: boolean
        if use_sqrt == True a sqrt is applied to the joint-ISI histogram,
        following Gerstein et al. 2004
        Default: False
    method: string
        if 'window': the spike movement is limited to the parameter dither.
        if 'fast': the spike can move in all the range between the previous
            spike and the subsequent spike. This is computationally much faster
            and thus is called 'fast'.
        Default: 'fast'
    cutoff: boolean
        if True then the Filtering of the Joint-ISI histogram is
        limited to the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain refractory
        period, which would be destroyed by the convolution with the
        2d-Gaussian function.
        Default: True
    min_spikes: int
        if the number of spikes is lower than this number, the spiketrain
        is directly passed to the uniform dithering.
        Default: 10

    Methods
    ----------
    preprocessing()
        The preprocessing function is called in the initialization process.
        Outside of it is only necessary if the attributes of the
        :class:`Joint_ISI_Space` were changed after the initialization, than it
        prepares the class again to create dithered spiketrains.
    dithering()
        Returns a list of dithered spiketrains and if print_mode it returns
        also a string 'uniform' or 'jisid' indicating the way, how the dithered
        spiketrains were obtained.
    """

    def __init__(self,
                 st,
                 n_surr=1,
                 dither=15. * pq.ms,
                 unit=pq.s,
                 window_length=120. * pq.ms,
                 num_bins=120,
                 sigma=1. * pq.ms,
                 isi_median_threshold=30 * pq.ms,
                 alternate=True,
                 print_mode=False,
                 use_sqrt=False,
                 method='fast',
                 cutoff=True,
                 min_spikes=10
                 ):
        self.st = st

        self.n_surr = n_surr

        self.dither = dither
        self.window_length = window_length
        self.sigma = sigma
        self.isi_median_threshold = isi_median_threshold

        self.num_bins = num_bins

        self.alternate = alternate
        self.print_mode = print_mode
        self.use_sqrt = use_sqrt
        self.method = method
        self.cutoff = cutoff
        self.min_spikes = min_spikes

        self.preprocessing()

    def preprocessing(self):
        """
        To perform the Joint-ISI dithering a preprocessing procedure for each
        spiketrain is necessary. This is part of the initializer (__init___).
        If after calling the class for the first time, a parameter is changed,
        the preprocessing needs to be done again.

        First, two checks are done. If they are not passed, self.method is
        set to 'uniform'. The first one asks for the number of spikes.
        The second compares the median of the ISI-distribution against a
        threshold.

        If the method is not 'uniform' the cumulative distribution functions
        for the Joint-ISI dither process are evaluated.

        If method is 'fast':
        For each slice of the joint-ISI
        distribution (parallel to the anti-diagonal) a cumulative distribution
        function is calculated.

        If method is 'window':
        For each point in the joint-ISI distribution a on the line parallel to
        the anti-diagonal all points up to the dither-parameter are included,
        to calculate the cumulative distribution function.

        The function has no output, but stores its result inside the class.
        """
        if len(self.st) < self.min_spikes:
            self.method = 'uniform'
            return None

        self._unit = self.st.units

        self._isi = stats.isi(self.st.rescale(self._unit).magnitude)
        isi_median = np.median(self._isi)

        if isi_median > self.isi_median_threshold.rescale(
                self._unit).magnitude:
            self.method = 'uniform'
            return None

        if isinstance(self.dither, pq.Quantity):
            self.dither = self.dither.rescale(self._unit).magnitude
        if isinstance(self.window_length, pq.Quantity):
            self.window_length = self.window_length.rescale(
                self._unit).magnitude
        if isinstance(self.sigma, pq.Quantity):
            self.sigma = self.sigma.rescale(self._unit).magnitude

        self._sampling_rhythm = self.alternate + 1

        self._bin_width = self.window_length / self.num_bins

        def isi_to_index(isi):
            return np.rint(isi / self._bin_width - 0.5).astype(int)

        self._isi_to_index = isi_to_index

        self._number_of_isis = len(self._isi)
        self._first_spike = self.st[0].rescale(self._unit).magnitude
        self._t_stop = self.st.t_stop.rescale(self._unit).magnitude

        self._get_joint_isi_histogram()

        # Gives an array, taking an element with an index of the Joint-ISI
        # distribution gives back the corresponding ISI.
        self._indices_to_isi = (np.arange(self.num_bins)
                                + 0.5) * self._bin_width

        flipped_jisih = np.flip(self.jisih.T, 0)

        def normalize(v):
            if v[-1] - v[0] > 0.:
                return (v - v[0]) / (v[-1] - v[0])
            return np.zeros_like(v)

        self._normalize = normalize

        if self.method == 'fast':
            self._jisih_cumulatives = [normalize(
                np.cumsum(np.diagonal(flipped_jisih,
                                      -self.num_bins + double_index + 1)))
                for double_index in range(self.num_bins)]
            return None

        if self.method == 'window':
            self._jisih_cumulatives = self._window_cumulatives(flipped_jisih)
            return None

        error_message = ('method must can only be \'uniform\' or \'fast\' '
                         'or \'window\', but not \'' + self.method + '\' .')
        raise ValueError(error_message)

    def dithering(self):
        """
        Implementation of Joint-ISI-dithering for spiketrains that pass the
        threshold of the dense rate, if not a uniform dithered spiketrain is
        given back. The implementation continued the ideas of Louis et al.
        (2010) and Gerstein (2004).

        Returns
        ----------
        dithered_sts: list
            list of spiketrains, that are dithered versions of the given
            spiketrain
        if print_mode == True
        mode: string
            Indicates, which method was used to dither the spikes.
            'jisid' if joint-ISI was used,
            'uniform' if the ISI median was too low and uniform dithering was
            used.
        """
        if self.method == 'uniform':
            if self.print_mode:
                return surr.dither_spikes(
                    self.st, self.dither,
                    n=self.n_surr), 'uniform'
            return surr.dither_spikes(
                self.st, self.dither,
                n=self.n_surr)

        if self.method == 'fast' or self.method == 'window':
            if self.print_mode:
                return self._dithering_process(), 'jisid'
            return self._dithering_process()

        error_message = ('method must can only be \'uniform\' or \'fast\' '
                         'or \'window\', but not \'' + self.method + '\' .')
        raise ValueError(error_message)

    def _get_joint_isi_histogram(self):
        """
        This function calculates the joint-ISI histogram.
        """
        jisih = np.histogram2d(self._isi[:-1], self._isi[1:],
                               bins=[self.num_bins, self.num_bins],
                               range=[[0., self.window_length],
                                      [0., self.window_length]])[0]

        if self.use_sqrt:
            jisih = np.sqrt(jisih)

        if self.cutoff:
            minimal_isi = np.min(self._isi)
            start_index = self._isi_to_index(minimal_isi)
            jisih[start_index:, start_index:] = gaussian_filter(
                jisih[start_index:, start_index:],
                self.sigma / self._bin_width)

            jisih[:start_index + 1, :] = np.zeros_like(
                jisih[:start_index + 1, :])
            jisih[:, :start_index + 1] = np.zeros_like(
                jisih[:, :start_index + 1])

        else:
            jisih = gaussian_filter(jisih, self.sigma / self._bin_width)
        self.jisih = jisih
        return None

    def _window_diagonal_cumulatives(self, flipped_jisih):
        self.max_change_index = self._isi_to_index(self.dither)
        self.max_change_isi = self._indices_to_isi[self.max_change_index]

        jisih_diag_cums = np.zeros((self.num_bins,
                                    self.num_bins
                                    + 2 * self.max_change_index))

        for double_index in range(self.num_bins):
            cum_diag = np.cumsum(np.diagonal(flipped_jisih,
                                             - self.num_bins
                                             + double_index + 1))
            jisih_diag_cums[double_index,
                            self.max_change_index:
                            double_index
                            + self.max_change_index + 1] = cum_diag

            cum_bound = np.repeat(jisih_diag_cums[double_index,
                                                  double_index +
                                                  self.max_change_index],
                                  self.max_change_index)

            jisih_diag_cums[double_index,
                            double_index + self.max_change_index + 1:
                            double_index
                            + 2 * self.max_change_index + 1] = cum_bound
        return jisih_diag_cums

    def _window_cumulatives(self, flipped_jisih):
        jisih_diag_cums = self._window_diagonal_cumulatives(flipped_jisih)
        jisih_cumulatives = np.zeros(
            (self.num_bins, self.num_bins,
             2 * self.max_change_index + 1))
        for back_index in range(self.num_bins):
            for for_index in range(self.num_bins - back_index):
                double_index = for_index + back_index
                cum_slice = jisih_diag_cums[double_index,
                                            back_index:
                                            back_index +
                                            2 * self.max_change_index + 1]
                normalized_cum = self._normalize(cum_slice)
                jisih_cumulatives[back_index][for_index] = normalized_cum
        return jisih_cumulatives

    def _dithering_process(self):
        """
        Dithering process for the Joint-ISI dithering.

        Returns
        --------
        dithered_sts
            list of neo.SpikeTrain: A list of len n_surr,
            each entry is one dithered spiketrain.
        """

        dithered_sts = []
        for surr_number in range(self.n_surr):
            dithered_isi = self._get_dithered_isi()

            dithered_st = self._first_spike + np.hstack(
                (np.array(0.), np.cumsum(dithered_isi)))
            dithered_st = neo.SpikeTrain(dithered_st * self._unit,
                                         t_stop=self._t_stop)
            dithered_sts.append(dithered_st)
        return dithered_sts

    def _get_dithered_isi(self):
        dithered_isi = self._isi
        random_list = np.random.random(self._number_of_isis)
        if self.method == 'fast':
            for start in range(self._sampling_rhythm):
                dithered_isi_indices = self._isi_to_index(dithered_isi)
                for i in range(start, self._number_of_isis - 1,
                               self._sampling_rhythm):
                    self._update_dithered_isi_fast(dithered_isi,
                                                   dithered_isi_indices,
                                                   random_list[i],
                                                   i)
        else:
            for start in range(self._sampling_rhythm):
                dithered_isi_indices = self._isi_to_index(dithered_isi)
                for i in range(start, self._number_of_isis - 1,
                               self._sampling_rhythm):
                    self._update_dithered_isi_window(dithered_isi,
                                                     dithered_isi_indices,
                                                     random_list[i],
                                                     i)
        return dithered_isi

    def _update_dithered_isi_fast(self,
                                  dithered_isi,
                                  dithered_isi_indices,
                                  random_number,
                                  i):
        back_index = dithered_isi_indices[i]
        for_index = dithered_isi_indices[i + 1]
        double_index = back_index + for_index
        if double_index < self.num_bins:
            if self._jisih_cumulatives[double_index][-1]:
                cond = (self._jisih_cumulatives[double_index]
                        > random_number)
                new_index = np.where(
                    cond,
                    self._jisih_cumulatives[double_index],
                    np.inf).argmin()
                step = (self._indices_to_isi[new_index]
                        - self._indices_to_isi[back_index])
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step
        return None

    def _update_dithered_isi_window(self,
                                    dithered_isi,
                                    dithered_isi_indices,
                                    random_number,
                                    i):
        back_index = dithered_isi_indices[i]
        for_index = dithered_isi_indices[i + 1]
        if back_index + for_index < self.num_bins:
            cum_dist_func = self._jisih_cumulatives[
                back_index][for_index]
            if cum_dist_func[-1]:
                cond = cum_dist_func > random_number
                new_index = np.where(
                    cond,
                    cum_dist_func,
                    np.inf).argmin()
                step = (self._indices_to_isi[new_index]
                        - self.max_change_isi)
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step
        return None
