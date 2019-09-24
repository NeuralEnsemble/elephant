'''
Module to generate surrogates of spike trains by using the joint-ISI dithering.
Features are provided to separate the preprocessing from the main process, also
some plot routines to analyze the results.

Original implementation by: Peter Bouss [p.bouss@fz-juelich.de]
:copyright: Copyright 2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
'''


import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import time

import elephant.spike_train_surrogates as spike_train_surrogates
import elephant.statistics as ele_statistics
from scipy.ndimage import gaussian_filter


def joint_isi_dithering(st, **kwargs):
    '''
    Implementation of Joint-ISI-dithering for spiketrains that pass the
    threshold of the dense rate, if not a uniform dithered spiketrain is given
    back. The implementation continued the ideas of Louis et al. (2010) and
    Gerstein (2004).

    Inputs:
    st
        neo.SpikeTrain: For this spiketrain the surrogates will be created

    **kwargs non-necessary parameters
    n_surr
        int: Number of surrogates to be created.
            Default is 1.
    dither
        pq.Quantity: The range of the dithering for the uniform dithering,
            which is also used for the method 'window'.
            Default is 0.015*pq.s.
    unit
        pq.unit: The unit of the spiketrain in the output.
            Default is pq.s
    window_length
        pq.Quantity: The extent in which the joint-ISI-distribution is
                    calculated.
            Default is 0.06*pq.s
    jisih_bins
        int: The size of the joint-ISI-ditribution will be
            jisih_bins*jisih_bins.
            Default is 120
    sigma
        pq.Quantity: The standard deviation of the Gaussian kernel, with which
            the data is convoluted
            Default is 0.001*pq.s
    dense_rate
        float: Percentage of isis, that has to be lower then 2*dither, so the
            the joint-ISI-dithering is applied. Else the uniform dithering
             would be applied.
             Default is 0.5
    alternate
        int: determines in which order the ISIs are changed, i.e. if alternate
            is 2 first the ISIs with even indices are changed, than those with odd
            indices.
            Default is 2.
    show_plot
        boolean: if show_plot=True the joint-ISI distribution will be plotted
            Default is False
    print_mode
        boolean: If True, also the way of how the dithered spikes are evaluated
            is returned so 'dither' for uniform and dithering and 'jisih' for
            joint-ISI-dithering
            Default is False
    use_sqrt
        boolean: if use_sqrt=True a sqrt is applied to the joint-ISI histogram,
            following Gerstein et al. 2004
            Default is False
    method
        string: if 'fast' the entire diagonals of the joint-ISI histograms are
            used if 'window' only the values of the diagonals are used, whose
            distance is lower than dither
            Default is 'fast'
    cutoff
        boolean: if True than the Filtering of the Joint-ISI histogram is
            limited to the lower side by the minimal ISI.
            This can be necessary, if in the data there is a certain dead time,
            which would be destroyed by the convolution with the 2d-Gaussian
            function.
            Default is True
    number_of_trials
        int: if the number of spikes is lower than this number, the spiketrain
            is directly passed to the uniform dithering.
            Set such that in average at least on spike in each trial is needed.
            Default is 35

    Output:
    dithered_sts
        List of spiketrains, dithered with Joint-ISI-dithering
    mode (only if print_mode=True)
        string: Indicates, which method was used to dither the spikes.
            'jisih' if joint-ISI was used,
            'dither' if the dense_rate was too low and uniform dithering was
             used.

    '''

    isi, jisih_cumulatives, params = preprocessing_joint_isi_dithering(st,
                                                                    **kwargs)

    return processing_joint_isi_dithering(st, isi, jisih_cumulatives, **params)

def preprocessing_joint_isi_dithering(st, **kwargs):
    '''
    All preprocessing steps for the joint-ISI dithering are done here.

    So first to checks are done. If they are not passed, params['method'] is
    set to 'dither'. The first one asks for the number of spikes. The second
    for a value of the cumulative distribution function of the ISI.

    If the method is not 'dither' the cumulative distribution functions for the
    joint-ISI dither process are evaluated.

    If method is 'fast':
    For each slice of the joint-ISI
    distribution (parallel to the antidiagonal) a cumulative distr. function is
    calculated.

    If method is 'window':
    For each point in the joint-ISI distribution a on the line parallel to the
    antidiagonal all points up to the dither-parameter are included, to
    calculate the cumulative distribution function.

    Inputs:
    Same as for joint_isi_dithering(st, **kwargs)

    Outputs:
    isi
        np.ndarray: The interspike intervals of the spiketrain.
                    Values are in the unit of params['unit'].
                    [None] if params['method']='dither'
    jisih_cumulatives:
        np.ndarray: The cumulatives distribution functions as described above.
                    [None] if params['method']='dither'
    params:
        dict: Dicticionary of parameters.
    '''
    params = _jisih_add_default_parameters(**kwargs)

    if len(st) < params['number_of_trials']:
        params['method'] = 'dither'
        return [None], [None], params

    isi=ele_statistics.isi(st.rescale(params['unit']).magnitude)
    if np.sum(np.where(
                isi<2*params['dither'], 1., 0.
                ))/len(isi) < params['dense_rate']:
        params['method'] = 'dither'
        return [None], [None], params

    params['number_of_isis'] = len(isi)
    params['first_spike'] = st[0].rescale(params['unit']).magnitude
    params['t_stop'] = st.t_stop.rescale(params['unit']).magnitude

    jisih = _get_joint_isi_histogram(isi, **params)

    jisih_bins = params['jisih_bins']
    params['indices_to_isi'] = params['index_to_isi'](np.arange(jisih_bins))

    flipped_jisih=np.flip(jisih.T,0)
    def normalize(v):
        if v[-1]-v[0] > 0.:
            return (v-v[0])/(v[-1]-v[0])
        else:
            return np.zeros_like(v)

    if params['method'] == 'fast':
        jisih_cumulatives = [normalize(
            np.cumsum(np.diagonal(flipped_jisih, -jisih_bins+double_index+1)))
            for double_index in range(jisih_bins)]
        return isi, jisih_cumulatives, params

    elif params['method'] == 'window':
        max_change_index = params['isi_to_index'](params['dither'])
        max_change_isi = params['indices_to_isi'][max_change_index]
        params['max_change_isi'] = max_change_isi
        jisih_bins = params['jisih_bins']

        jisih_diagonals_cumulatives = np.zeros(
                                    (jisih_bins,jisih_bins+2*max_change_index))

        for double_index in range(jisih_bins):
            jisih_diagonals_cumulatives[double_index,
                max_change_index:
                double_index+max_change_index+1
                ]=np.cumsum(np.diagonal(
                flipped_jisih, -jisih_bins+double_index+1))
            jisih_diagonals_cumulatives[double_index,
                double_index+max_change_index+1:
                double_index+2*max_change_index+1
                ]=np.repeat(jisih_diagonals_cumulatives[double_index,
                double_index+max_change_index], max_change_index)

        jisih_cumulatives=np.zeros(
                            (jisih_bins, jisih_bins, 2*max_change_index+1))
        for back_index in range(jisih_bins):
            for for_index in range(jisih_bins-back_index):
                double_index=for_index+back_index
                jisih_cumulatives[back_index][for_index]=normalize(
                    jisih_diagonals_cumulatives[
                    double_index,back_index:back_index+2*max_change_index+1])

        return isi, jisih_cumulatives, params

def processing_joint_isi_dithering(st, isi, jisih_cumulatives, **params):
    '''
    The main processing function of the joint-ISI dithering. Shall be only
    used, with the output of the preprocessing function.
    Essentially for each of the three methods, the corresponding dither
    function is called.
    Input:
    See Output arguments of preprocessing_joint_isi_dithering(st, **kwargs)

    Output:
    See Output arguments of joint_isi_dithering(st, **kwargs)
    '''
    if params['method'] == 'dither':
        if params['print_mode']:
            return spike_train_surrogates.dither_spikes(
                    st, params['dither']*params['unit'],
                    n=params['n_surr']), 'dither'
        else:
            return spike_train_surrogates.dither_spikes(
                    st, params['dither']*params['unit'],
                    n=params['n_surr'])

    elif params['method'] == 'fast':
        if params['print_mode']:
            return _joint_isi_dithering_fast(
                    isi, jisih_cumulatives, **params), 'jisih'
        else:
            return _joint_isi_dithering_fast(
                    isi,jisih_cumulatives,**params)

    elif params['method'] == 'window':
        isi,jisih_cumulatives = args
        if params['print_mode']:
            return _joint_isi_dithering_window(
                isi, jisih_cumulatives, **params), 'jisih'
        else:
            return _joint_isi_dithering_window(
                isi, jisih_cumulatives, **params)

def plot_difference_in_joint_isi_distributions(st, dithered_sts, **kwargs):
    '''
    A function made to compare the joint_isi_distribution of the original
    spiketrain against the ones of the dithered spikestrains. The function
    shows three plots:
    First, the joint-ISI distribution of the original spiketrain,
    which is convoluted with a 2d-Gaussian depending on the sigma in kwargs.
    Second, the mean of the joint-ISI distribution of dithered spiketrains.
    In this case, no convolution is applied.
    Third, the difference between these teo distributions is shown, weighted by
    the standard deviation of the second case.
    '''
    params=_jisih_add_default_parameters(**kwargs)
    window_length=params['window_length']

    isi=np.diff(st.rescale(params['unit']).magnitude)
    minimal_isi=np.min(isi)
    start_index=params['isi_to_index'](minimal_isi)


    jisih=_get_joint_isi_histogram(isi, **params)

    params_dithered=params
    params_dithered['sigma']=0.
    params_dithered['cutoff']=False

    dithered_jisih=[_get_joint_isi_histogram(np.diff(dithered_st.magnitude),
                    **params_dithered)
                    for dithered_st in dithered_sts]
    mean_jisih=np.mean(dithered_jisih, axis=0)
    std_jisih=np.std(dithered_jisih, axis=0)

    plt.figure(figsize=[12.8, 9.6])
    plt.imshow(jisih, origin='lower',
                extent=(0., window_length, 0., window_length))
    plt.xlabel('ISI(i+1) in s')
    plt.ylabel('ISI(i) in s')
    plt.title('Joint-ISI-distribution \n Original')
    #plt.xlim(left=0.00,right=0.03)
    #plt.ylim(bottom=0.00,top=0.03)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=[12.8, 9.6])
    plt.imshow(mean_jisih, origin='lower',
                extent=(0., window_length, 0., window_length))
    plt.xlabel('ISI(i+1) in s')
    plt.ylabel('ISI(i) in s')
    plt.title('Joint-ISI-distribution \n Mean Dithered')
    #plt.xlim(left=0.00,right=0.03)
    #plt.ylim(bottom=0.00,top=0.03)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=[12.8, 9.6])
    plt.imshow((jisih-mean_jisih)/std_jisih, vmin=-3., vmax=3.,
                cmap='nipy_spectral', origin='lower',
                extent=(0.,window_length,0.,window_length))
    plt.xlabel('ISI(i+1) in s')
    plt.ylabel('ISI(i) in s')
    plt.title('Joint-ISI-distribution '+
                '\n (Original - Mean Dithered) / Std Dithered')
    #plt.xlim(left=0.00, right=0.03)
    #plt.ylim(bottom=0.00, top=0.03)
    plt.colorbar()
    plt.show()
    return None

def _jisih_add_default_parameters(**kwargs):
    '''
    In these function we have all default parameters, which are changed
    according to kwargs.

    Input:
    **kwargs: as in joint_isi_dithering(st, **kwargs)

    Output:
    params
        dict: Dictionary of parameters.
    '''
    #Default Parameters
    params = {  'n_surr':1,
                'dither':0.015*pq.s,
                'unit':pq.s,
                'window_length':0.06*pq.s,
                'jisih_bins':120,
                'sigma':0.001*pq.s,
                'dense_rate':0.5,
                'alternate':2,
                'show_plot':False,
                'print_mode':False,
                'use_sqrt':False,
                'method':'fast',
                'cutoff':True,
                'number_of_trials':35}

    for key in kwargs.keys():
        params[key] = kwargs[key]
    for key in params.keys():
        if isinstance(params[key], pq.Quantity):
            if key != 'unit':
                params[key] = params[key].rescale(params['unit']).magnitude

    params['bin_width'] = params['window_length']/params['jisih_bins']

    def index_to_isi(ind,bin_width = params['bin_width']):
        return (ind+0.5)*bin_width
    def isi_to_index(isi,bin_width = params['bin_width']):
        return np.rint(isi/bin_width-0.5).astype(int)

    params['index_to_isi'] = index_to_isi
    params['isi_to_index'] = isi_to_index
    return params

def _get_joint_isi_histogram(isi, **params):
    '''
    This function calculates the joint-ISI histogram.

    Inputs:
    isi
        np.ndarray: The interspike intervals.

    **params
    corresponding to the output of _jisih_add_default_parameters(**kwargs)
    Important here:
    use_sqrt
        boolean: applies a square root to the Joint-ISI histogram corresponding
            to Gerstein (2004) and Louis et al. (2010)
    cutoff
        boolean: the minimal ISI is fixed, as it is in the initial ISI.
            A very important feature, if the Low-ISI regime is clipped.
    show_plot
        boolean: A plot of the joint-ISI histogram is shown

    Output
    jisih
        np.ndarray: the joint-ISI histogram.
    '''
    jisih=np.histogram2d(isi[:-1], isi[1:],
                        bins=[params['jisih_bins'], params['jisih_bins']],
                        range=[[0., params['window_length']],
                        [0., params['window_length']]])[0]

    if params['use_sqrt']:
        jisih = np.sqrt(jisih)

    if params['cutoff']:
        minimal_isi = np.min(isi)
        start_index = params['isi_to_index'](minimal_isi)
        jisih[start_index:, start_index:] = gaussian_filter(
            jisih[start_index:, start_index:],
            params['sigma']/params['bin_width'])

        jisih[:start_index+1, :] = np.zeros_like(jisih[:start_index+1, :])
        jisih[:, :start_index+1] = np.zeros_like(jisih[:, :start_index+1])

    else:
        jisih = gaussian_filter(jisih,params['sigma']/params['bin_width'])

    if params['show_plot']:
        plt.figure(figsize = [12.8, 9.6])
        plt.imshow(jisih, origin = 'lower',
                    extent = (0., params['window_length'],
                    0., params['window_length']))
        plt.xlabel('ISI(i+1) in s')
        plt.ylabel('ISI(i) in s')
        if params['use_sqrt']:
            plt.title('Joint-ISI-distribution (sqrt)')
        else:
            plt.title('Joint-ISI-distribution')
        plt.colorbar()
        plt.show()
    return jisih

def _joint_isi_dithering_fast(isi, jisih_cumulatives, **params):
    '''
    Dithering process for the fast version of the joint-ISI dithering.

    Inputs:
    isi
        np.ndarray: The interspike-intervals
    jisih_cumulatives:
        np.ndarray: The cumulatives distribution functions as calculated
                    in preprocessing_joint_isi_dithering(st, **kwargs).
    params
        see output of preprocessing_joint_isi_dithering(st,**kwargs)

    Output:
    dithered_sts
        list of neo.SpikeTrain: A list of len n_surr,
            each entry is one dithered spiketrain.
    '''
    jisih_bins = params['jisih_bins']
    first_spike = params['first_spike']
    t_stop = params['t_stop']
    unit = params['unit']
    isi_to_index = params['isi_to_index']
    alternate = params['alternate']
    indices_to_isi = params['indices_to_isi']
    number_of_isis = params['number_of_isis']

    ###counter_isi = 0
    dithered_sts = []
    for surr_number in range(params['n_surr']):
        dithered_isi = isi
        random_list = np.random.random(number_of_isis)
        for k in range(alternate):
            dithered_isi_indices = isi_to_index(dithered_isi)
            for i in range(k, number_of_isis-1, alternate):
                back_index = dithered_isi_indices[i]
                for_index = dithered_isi_indices[i+1]
                double_index = back_index+for_index
                if double_index<jisih_bins:
                    if jisih_cumulatives[double_index][-1]>0.:
                        step = indices_to_isi[np.where(
                            jisih_cumulatives[double_index]>random_list[i],
                            jisih_cumulatives[double_index],
                            np.inf).argmin()]-indices_to_isi[back_index]
                        dithered_isi[i] += step
                        dithered_isi[i+1] -= step
                        ###counter_isi+ = 1

        dithered_st = first_spike+np.hstack(
                        (np.array(0.), np.cumsum(dithered_isi)))
        dithered_st = neo.SpikeTrain(dithered_st*unit, t_stop=t_stop)
        if len(dithered_st) != number_of_isis + 1:
            dithered_st = _correct_artefacts_of_neo(dithered_st,
                                                    number_of_isis+1)
        dithered_sts.append(dithered_st)
    ###print('Percentage of spikes moved: {:.4}'.format(counter_isi/
    ###                                 ((number_of_isis+1)*params['n_surr'])))
    ###print(number_of_isis+1)
    return dithered_sts


def _joint_isi_dithering_window(isi,jisih_cumulatives,**params):
    '''
    Dithering process for the window version of the joint-ISI dithering.

    Inputs:
    isi
        np.ndarray: The interspike-intervals
    jisih_cumulatives:
        np.ndarray: The cumulatives distribution functions as calculated in
            preprocessing_joint_isi_dithering(st, **kwargs).
    params
        see output of preprocessing_joint_isi_dithering(st,**kwargs)

    Output:
    dithered_sts
        list of neo.SpikeTrain: A list of len n_surr,
            each entry is one dithered spiketrain.
    '''
    jisih_bins = params['jisih_bins']
    first_spike = params['first_spike']
    t_stop = params['t_stop']
    unit = params['unit']
    isi_to_index = params['isi_to_index']
    alternate = params['alternate']
    indices_to_isi = params['indices_to_isi']
    max_change_isi = params['max_change_isi']
    number_of_isis = params['number_of_isis']

    ###counter_isi = 0
    dithered_sts = []
    for surr_number in range(params['n_surr']):
        dithered_isi = isi
        random_list = np.random.random(number_of_isis)
        for k in range(alternate):
            dithered_isi_indices = isi_to_index(dithered_isi)
            for i in range(k, number_of_isis-1, alternate):
                back_index = dithered_isi_indices[i]
                for_index = dithered_isi_indices[i+1]
                if back_index+for_index<jisih_bins:
                    if jisih_cumulatives[back_index][for_index][-1]>0.:
                        step = indices_to_isi[np.where(
                            jisih_cumulatives[back_index][
                            for_index] > random_list[i],
                            jisih_cumulatives[back_index][for_index],
                            np.inf).argmin()]-max_change_isi
                        dithered_isi[i] += step
                        dithered_isi[i+1] -= step
                        ###counter_isi+=1
        dithered_st = first_spike + np.hstack(
                        (np.array(0.), np.cumsum(dithered_isi)))
        dithered_st = neo.SpikeTrain(dithered_st*unit, t_stop = t_stop)
        if len(dithered_st) != number_of_isis + 1:
            dithered_st = _correct_artefacts_of_neo(dithered_st,
                                                    number_of_isis+1)
        dithered_sts.append(dithered_st)
    ###print('Percentage of spikes moved: {:.4}'.format(counter_isi/
    ###                                           ((number_of_isis+1)*n_surr)))
    return dithered_sts

def _correct_artefacts_of_neo(st,number_of_spikes):
    '''
    For strange reasons, somtimes building a neo.SpikeTrain, a small amounts of
    spikes get lost. This are here thrown back into the spiketrain according to
    a uniform distribution.

    Input:
    st
        neo.SpikeTrain
    number_of_spikes
        int: Number of Spiketrain that the spiketrain should have.


    Output:
    st
        neo.SpikeTrain: SpikeTrain with the correct number of spikes
    '''
    if len(st) < number_of_spikes:
        t_start = st.t_start
        t_stop = st.t_stop
        unit = st.unit
        st_pure = st.magnitude
        st_pure.append(np.random.random()*
            (t_stop.magnitude-t_start.magnitude)+t_start.magnitude)
        st_pure.sort()
        st = neo.SpikeTrain(st_pure*unit, t_start = t_start, t_stop = t_stop)
        if len(st) < number_of_spikes:
            return _correct_artefacts_of_neo(st, number_of_spikes)
        return st
    raise ValueError('SpikeTrain already long enough')
