'''
Module to generate surrogates of spike trains by using the joint-ISI dithering.
Features are provided to separate the preprocessing from the main process.

Original implementation by: Peter Bouss [p.bouss@fz-juelich.de]
:copyright: Copyright 2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
'''


import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo

import elephant.spike_train_surrogates as surr
import elephant.statistics as stats
from scipy.ndimage import gaussian_filter


class Joint_ISI_Space:
    '''
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
        The extent in which the joint-ISI-distribution is calculated.
        Default: 120*pq.ms
    num_bins: int
        The size of the joint-ISI-ditribution will be num_bins*num_bins.
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
        dithered. Else: in acending order from the first to the last spike, all
        spikes are moved.
        Default: True.
    show_plot: boolean
        if show_plot == True the joint-ISI distribution will be plotted
        Default: False
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
        if 'fast' the entire diagonals of the joint-ISI histograms are
        used if 'window' only the values of the diagonals are used, whose
        distance is lower than dither
        Default: 'fast'
    cutoff: boolean
        if True than the Filtering of the Joint-ISI histogram is
        limited to the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain dead time,
        which would be destroyed by the convolution with the 2d-Gaussian
        function.
        Default: True
    min_spikes: int
        if the number of spikes is lower than this number, the spiketrain
        is directly passed to the uniform dithering.
        Default: 10

    Methods
    ----------
    preprocessing()
        The preprocessing function is called in the initialization process.
        Wutside of it is only necessary if the attributes of the
        :class:`Joint_ISI_Space` were changed after the initialization, than it
        prepares the class again to create dithered spiketrains.
    dithering()
        Returns a list of dithered spiketrains and if print_mode it returns
        also a string 'uniform' or 'jisid' indicating the way, how the dithered
        spiketrains were obtained.

    Output:
    dithered_sts
        List of spiketrains, dithered with Joint-ISI-dithering
    mode (only if print_mode=True)
        string: Indicates, which method was used to dither the spikes.
            'jisih' if joint-ISI was used,
            'uniform' if the dense_rate was too low and uniform dithering was
             used.
    '''

    def __init__(self,
                 st,
                 n_surr=1,
                 dither=15.*pq.ms,
                 unit=pq.s,
                 window_length=120.*pq.ms,
                 num_bins=120,
                 sigma=1.*pq.ms,
                 isi_median_threshold=30*pq.ms,
                 alternate=True,
                 show_plot=False,
                 print_mode=False,
                 use_sqrt=False,
                 method='fast',
                 cutoff=True,
                 min_spikes=10
                 ):
        self.st = st

        self.n_surr = n_surr

        self.unit = unit
        self.dither = dither
        self.window_length = window_length
        self.sigma = sigma
        self.isi_median_threshold = isi_median_threshold

        self.num_bins = num_bins

        self.alternate = alternate
        self.show_plot = show_plot
        self.print_mode = print_mode
        self.use_sqrt = use_sqrt
        self.method = method
        self.cutoff = cutoff
        self.min_spikes = min_spikes

        self.preprocessing()

    def preprocessing(self):
        '''
        All preprocessing steps for the joint-ISI dithering are done here.

        So first to checks are done. If they are not passed, self.method is
        set to 'uniform'. The first one asks for the number of spikes.
        The second compares the median of the ISI-distribution against a
        threshold.

        If the method is not 'uniform' the cumulative distribution functions
        for the Joint-ISI dither process are evaluated.

        If method is 'fast':
        For each slice of the joint-ISI
        distribution (parallel to the antidiagonal) a cumulative distr.
        function is calculated.

        If method is 'window':
        For each point in the joint-ISI distribution a on the line parallel to
        the antidiagonal all points up to the dither-parameter are included, to
        calculate the cumulative distribution function.

        The function has no output, but stores its result inside the class.
        '''
        if len(self.st) < self.min_spikes:
            self.method = 'uniform'
            return None

        self.isi = stats.isi(self.st.rescale(self.unit).magnitude)
        isi_median = np.median(self.isi)

        if isi_median > self.isi_median_threshold.rescale(
                self.unit).magnitude:
            self.method = 'uniform'
            return None

        if isinstance(self.dither, pq.Quantity):
            self.dither = self.dither.rescale(self.unit).magnitude
        if isinstance(self.window_length, pq.Quantity):
            self.window_length = self.window_length.rescale(
                self.unit).magnitude
        if isinstance(self.sigma, pq.Quantity):
            self.sigma = self.sigma.rescale(self.unit).magnitude

        self.sampling_rhythm = self.alternate + 1

        self.bin_width = self.window_length/self.num_bins

        def index_to_isi(ind):
            return (ind+0.5)*self.bin_width
        self.index_to_isi = index_to_isi

        def isi_to_index(isi):
            return np.rint(isi/self.bin_width-0.5).astype(int)
        self.isi_to_index = isi_to_index

        self.number_of_isis = len(self.isi)
        self.first_spike = self.st[0].rescale(self.unit).magnitude
        self.t_stop = self.st.t_stop.rescale(self.unit).magnitude

        self._get_joint_isi_histogram()

        self.indices_to_isi = self.index_to_isi(np.arange(self.num_bins))

        flipped_jisih = np.flip(self.jisih.T, 0)

        def normalize(v):
            if v[-1]-v[0] > 0.:
                return (v-v[0])/(v[-1]-v[0])
            return np.zeros_like(v)
        self.normalize = normalize

        if self.method == 'fast':
            self.jisih_cumulatives = [normalize(
                np.cumsum(np.diagonal(flipped_jisih,
                                      -self.num_bins+double_index+1)))
                for double_index in range(self.num_bins)]
            return None

        if self.method == 'window':
            self.jisih_cumulatives = self._window_cumulatives(flipped_jisih)
            return None

        error_message = ('method must can only be \'uniform\' or \'fast\' '
                         'or \'window\', but not \''+self.method+'\' .')
        raise ValueError(error_message)

    def dithering(self):
        '''
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
        '''
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
                         'or \'window\', but not \''+self.method+'\' .')
        raise ValueError(error_message)

    def _get_joint_isi_histogram(self):
        '''
        This function calculates the joint-ISI histogram.
        '''
        jisih = np.histogram2d(self.isi[:-1], self.isi[1:],
                               bins=[self.num_bins, self.num_bins],
                               range=[[0., self.window_length],
                                      [0., self.window_length]])[0]

        if self.use_sqrt:
            jisih = np.sqrt(jisih)

        if self.cutoff:
            minimal_isi = np.min(self.isi)
            start_index = self.isi_to_index(minimal_isi)
            jisih[start_index:, start_index:] = gaussian_filter(
                jisih[start_index:, start_index:],
                self.sigma/self.bin_width)

            jisih[:start_index+1, :] = np.zeros_like(jisih[:start_index+1, :])
            jisih[:, :start_index+1] = np.zeros_like(jisih[:, :start_index+1])

        else:
            jisih = gaussian_filter(jisih, self.sigma/self.bin_width)
        self.jisih = jisih

        if self.show_plot:
            plt.figure(figsize=[12.8, 9.6])
            plt.imshow(jisih, origin='lower',
                       extent=(0., self.window_length,
                               0., self.window_length))
            plt.xlabel('ISI(i+1) in s')
            plt.ylabel('ISI(i) in s')
            if self.use_sqrt:
                plt.title('Joint-ISI-distribution (sqrt)')
            else:
                plt.title('Joint-ISI-distribution')
            plt.colorbar()
            plt.show()
        return None

    def _window_diagonal_cumulatives(self, flipped_jisih):
        self.max_change_index = self.isi_to_index(self.dither)
        self.max_change_isi = self.indices_to_isi[self.max_change_index]

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
                            double_index
                            + self.max_change_index + 1:
                            double_index
                            + 2 * self.max_change_index + 1] = cum_bound
        return jisih_diag_cums

    def _window_cumulatives(self, flipped_jisih):
        jisih_diag_cums = self._window_diagonal_cumulatives(flipped_jisih)
        jisih_cumulatives = np.zeros(
            (self.num_bins, self.num_bins,
             2*self.max_change_index+1))
        for back_index in range(self.num_bins):
            for for_index in range(self.num_bins-back_index):
                double_index = for_index+back_index
                cum_slice = jisih_diag_cums[double_index,
                                            back_index:
                                            back_index +
                                            2*self.max_change_index + 1]
                normalized_cum = self.normalize(cum_slice)
                jisih_cumulatives[back_index][for_index] = normalized_cum
        return jisih_cumulatives

    def _dithering_process(self):
        '''
        Dithering process for the Joint-ISI dithering.

        Returns
        --------
        dithered_sts
            list of neo.SpikeTrain: A list of len n_surr,
            each entry is one dithered spiketrain.
        '''

        dithered_sts = []
        for surr_number in range(self.n_surr):
            dithered_isi = self._get_dithered_isi()

            dithered_st = self.first_spike+np.hstack(
                (np.array(0.), np.cumsum(dithered_isi)))
            dithered_st = neo.SpikeTrain(dithered_st*self.unit,
                                         t_stop=self.t_stop)
            dithered_sts.append(dithered_st)
        return dithered_sts

    def _get_dithered_isi(self):
        dithered_isi = self.isi
        random_list = np.random.random(self.number_of_isis)
        if self.method == 'fast':
            for start in range(self.sampling_rhythm):
                dithered_isi_indices = self.isi_to_index(dithered_isi)
                for i in range(start, self.number_of_isis-1,
                               self.sampling_rhythm):
                    self._update_dithered_isi_fast(dithered_isi,
                                                   dithered_isi_indices,
                                                   random_list[i],
                                                   i)
        else:
            for start in range(self.sampling_rhythm):
                dithered_isi_indices = self.isi_to_index(dithered_isi)
                for i in range(start, self.number_of_isis-1,
                               self.sampling_rhythm):
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
        for_index = dithered_isi_indices[i+1]
        double_index = back_index+for_index
        if double_index < self.num_bins:
            if self.jisih_cumulatives[double_index][-1]:
                cond = (self.jisih_cumulatives[double_index]
                        > random_number)
                new_index = np.where(
                    cond,
                    self.jisih_cumulatives[double_index],
                    np.inf).argmin()
                step = (self.indices_to_isi[new_index]
                        - self.indices_to_isi[back_index])
                dithered_isi[i] += step
                dithered_isi[i+1] -= step
        return None

    def _update_dithered_isi_window(self,
                                    dithered_isi,
                                    dithered_isi_indices,
                                    random_number,
                                    i):
        back_index = dithered_isi_indices[i]
        for_index = dithered_isi_indices[i+1]
        if back_index + for_index < self.num_bins:
            cum_dist_func = self.jisih_cumulatives[
                back_index][for_index]
            if cum_dist_func[-1]:
                cond = cum_dist_func > random_number
                new_index = np.where(
                    cond,
                    cum_dist_func,
                    np.inf).argmin()
                step = (self.indices_to_isi[new_index]
                        - self.max_change_isi)
                dithered_isi[i] += step
                dithered_isi[i+1] -= step
        return None
