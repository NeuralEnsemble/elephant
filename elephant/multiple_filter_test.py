# -*- coding: utf-8 -*-

"""
This algorithm is to determine if a spike train can be considerate stationary 
(costant firing rate) or not stationary (i.e. presence of one or more points at 
which the rate increases or decreases). In case of nonstationarity, the output 
is a list of detected  Change Points (CPs).
Essentialy, a two-side window of width 'h' ('_filter(t, h, spk)') slide over the 
time of the spike train '[h,t_final-h]'. This generates a '_filter_process' that 
at each time t assigns the difference between  spike lying in the right and left 
window. If at any time t this difference is large 'enough', it is assumed the 
presence of a rate Change Point in a neighborhood of t. A treshold 'test_quantile' 
for the maximum of the filter_process (max difference of spike count between the
left and right window) is derived based on asymptotic considerations.
The procedure is repeated for an arbitrary set of windows, with different size h.

Example 

Reference:
--------- 
Michael Messer, Marietta Kirchner, Julia Schiemann, Jochen Roeper,
Ralph Neininger, and Gaby Schneider. 'A multiple filter test for the
detection of rate changes in renewal processes with varying variance.'
Annals of Applied Statistics, 8(4):2027–2067, 12 2014.

### Adapted from published R implementation... ###
"""

import numpy as np
import quantities as pq


def multiple_filter_test(window_sizes, spk, t_final, alfa, n_surrogates,
                             test_quantile = None, test_param = None, dt = None):
 
    '''      
    Parameter
    ---------
        window_sizes : list of quantities,
                    set of windows' size 
        spk : list, array or SpikeTrain,
           spike train in analisys
        t_final : quantity, 
               final time of the spike train to be analysed
        alfa : integer, 
            alfa-quantile
        n_surrogates : integer, 
                    numbers of simulated limit processes
        dt : quantity, 
          resolution
        n_surrogates : scalar,
                    asyntotic treshold
        test_param : matrix, frist row list of h, second row empirical means and 
                 third row variances of the limit processes. They will be used to 
                 normalize the filter_processes for different h
                      
    Returns:
    --------
        cps : list of list
           one list for each h in increasing order of size 
           contaning the points detected with each window h.
    '''
    
    if (test_quantile is None)&(test_param is None):
        test_quantile, test_param = null_parameters(window_sizes, t_final,
                                                        alfa, n_surrogates, dt)
    elif test_quantile is None:
        test_quantile = null_parameters(window_sizes, t_final, alfa, 
                                                           n_surrogates, dt)[0]
    elif test_param is None:
        test_param = null_parameters(window_sizes, t_final, alfa,
                                                           n_surrogates, dt)[1]
        
    cps = []      # List of list of dected point, to be returned
  
    for i, h in enumerate(window_sizes):
        dt_temp = dt
        if dt_temp is None:  # automatic setting of dt
            dt_temp = h/20.
        # filter_process for window of size h    
        t, differences = _filter_process(dt_temp, h, spk, t_final, test_param) 
        time_index = np.arange(len(differences))
     
        cps_window = [] #Point detected by window h
        while (np.max(differences) > test_quantile):
            cp_index = np.argmax(differences)
            cp = cp_index*dt_temp + h #from index to time
            print "detected point {0}".format(cp), "with filter {0}".format(h)
            # before to repet the procedure the h-neighbourg of 'cp' detected is
            # cut, because rate changes within it are alredy explained by cp 
            differences[np.where((time_index > cp_index-int(h/dt_temp.rescale(h.units))) 
                    & (time_index < cp_index+int(h/dt_temp.rescale(h.units))))[0]] = 0 
                 
     
# The output consist in a list of lists: first being a list of deteced cps with 
# the first window h, then appending list of cps detected with other windows of 
# different size h. N.B.: only cps whose h-neighborhood does not include previously
# detected cps (with smaller window h) are added to the list. Same reason as l:101
            neighbourhood_free = True
            if i == 0:
                cps_window.append(cp)
            else: 
                # iterate on lists of cps detected with smaller window
                for j in range(i):   
                    # iterating on cps detected with the j^th samllest window
                    for c_pre in cps[j]: 
                        if (c_pre-h< cp < c_pre+h):
                            #ok = 0
                            neighbourhood_free = False
                            break
                #if none of the previous detected cp falls in the h neighbourhood
                if neighbourhood_free: 
                    cps_window.append(cp) # add the point to the list      
        cps.append(cps_window) # add the list to the list
                 
    return cps
    
    
def _brownian_motion(t_in, t_fin, x_in, dt):
    
    '''
    Generate a Brownian Motion.

    Parameter
    ---------
        t_in:    quantities, 
                initial time
        t_fin:   qiantities,
                final time
        x_in:    quantities,
                initial point of the process: _brownian_motio(0) = x_in
        dt:     quantities, 
                resolution
    Returns:
    --------
    Browniam motion on t_in-t_fin, with resolution dt and initial state x_in
    '''
    
    u = 1*pq.s
    try:
        t_in_sec = t_in.rescale(u)
    except:
        raise ValueError ("t_in must be a time quantity")
    t_in_m = t_in_sec.magnitude
    try:
        t_fin_sec = t_fin.rescale(u)
    except:
        raise ValueError ("t_fin must be a time quantity")
    t_fin_m = t_fin_sec.magnitude
    try:
        dt_sec = dt.rescale(u)
    except:
        raise ValueError ("dt must be a time quantity")
    t_fin_m = t_fin_sec.magnitude
    
    x = []
    for i in range(int((t_fin_m - t_in_m)/dt_sec)):
        x.append(np.random.normal(0,  np.sqrt(dt_sec)))
        
    s = np.cumsum(x)
    return s + x_in


def _limit_processes(window_sizes, t_final, dt):
    
    '''
    Generate the limit processes (depending only on t_final and h), one for each 
    h in H. The distribution of maxima of these processes is used to derive the
    threshold 'test_quantile' and the parameters 'test_param'.

    Parameter
    ---------
        window_sizes:      list of quantities, 
                           set of windows' size 
        T:                 quantity, F
                           end of limit process
        dt:                quantities, 
                           resolution 
    
    Returns:
    --------
        limit_processes : list of array
                        each entries contains the limit processes for each h,
                        evaluated in [h,T-h] with steps dt
    '''    
    
    limit_processes = []
    
    u = 1*pq.s
    try:
        window_sizes_sec = window_sizes.rescale(u)
    except:
        raise ValueError ("H must be a list of times")
    window_sizes_mag = window_sizes_sec.magnitude
    try:
        t_final_sec = t_final.rescale(u)
    except:
        raise ValueError ("t_fin must be a time scalar")
    if dt is not None:
        try:
            dt_sec = dt.rescale(u)
            dtm = dt.magnitude
        except:
            raise ValueError ("dt must be a time scalar")
    
    for h in window_sizes_mag : 
        if dt is None:   #automatic setting of dt
            dtm = h/20.
            dt_sec = dtm * pq.s
        else:
            dtm = dt.magnitude
                    
        T = t_final_sec-t_final_sec%(dt_sec)
        w =  _brownian_motion(0*pq.s, T, 0, dtm*u)    
        
        brownian_right = w[int(2*h/dtm):] #BM on [h,T-h], shifted in time t-->t+h
        brownian_left = w[:int(-2*h/dtm)] #BM on [h,T-h], shifted in time t-->t-h
        brownian_center = w[int(h/dtm):int(-h/dtm)]    #BM on [h,T-h]
      
        modul = (brownian_right + brownian_left - 2*brownian_center)
        limit_process_h = modul/(np.sqrt(2*h))
        limit_processes.append(limit_process_h) 
  
    return limit_processes
    

def null_parameters(window_sizes, t_final, alfa, n_surrogates, dt):    
    
    '''
    This function generates the threshold and the null parameters.
    The '_filter_process' has been proved to converge (for t_fin, h-->infinity) 
    to a continuous funcional of a Brownaian motion ('limit_process').
    Using a MonteCarlo techinique, maxima of these limit_processes are collected.
  
    The threshold is defined as the alfa quantile of this set of maxima. Namely:
    test_quantile := alpha quantile of {max_(h in window_size)[
                                 max_(t in [h, t_final-h])_limit_process_h(t)]}
                                 
    Parameter
    ---------
        window_sizes : list of quantities,
                    set of windows' size 
        t_final : quantity, 
               final time of the spike
        alfa : integer, 
            alfa-quantile
        n_surrogates : integer,
                    numbers of simulated limit processes
        dt : quantity,
          resolution
    
    Returns:
    --------
        test_quantile : scalar,
                    threshold for the maximum of the filter derivative process
        test_param : matrix, frist row list of h, second row Empirical means and third
                 row variances of the limit processes Lh. It will be used to 
                 normalize the number of elements inside the windows of differnt width h
 
    '''
    
    u = 1*pq.s
    try:
        window_sizes_sec = window_sizes.rescale(u)
    except:
        raise ValueError ("H must be a list of times")
    window_sizes_mag = window_sizes_sec.magnitude
    try:
        t_final_sec = t_final.rescale(u)
    except:
        raise ValueError ("T must be a time scalar")
    t_final_mag = t_final_sec.magnitude
        
    if t_final_mag <= 0:
        raise ValueError ("T needs to be stricktly poisitive")
    if alfa*(100 - alfa) < 0:
        raise ValueError ("alfa needs to be in (0,100)")
    if not isinstance(n_surrogates,int):
        raise TypeError ("numba of simulation needs to be integer")
    if np.min(window_sizes_mag) <= 0:
        raise ValueError ("window's size needs to be stricktly poisitive")
    if np.max(window_sizes_mag) >= t_final/2:
        raise ValueError ("window's size tooo large")
    if dt != None:
        try:
            dt = dt.rescale(u)
        except:
            raise ValueError ("dt must be a time scalar")        
        for h in window_sizes_mag:
            if (h/dt).magnitude - int(h/dt)!=0:     
                raise ValueError("Every window size h must be a multiple of dt")

    maxima_matrix = [] 
    #Generate a matrix: n X m where n = n_surrogates is the number of simulated 
    # limit processes and m is the number of choosen window size. Entrances are: 
    # M*(n,h) = max(t in T)[limit_process_h(t)], for each h in H and surrogate n. 
 
    for i in range(n_surrogates):
            mh_star = []
            simu = _limit_processes(window_sizes, t_final, dt)
            for i,h in enumerate(window_sizes_mag):
                m_h = np.max(simu[i]) #max over time of the i-th limit process
                mh_star.append(m_h)
            maxima_matrix .append(mh_star)
               
    maxima_matrix = np.asanyarray(maxima_matrix)
    matrix = maxima_matrix.T

    # matrix normalization by mean and variance of the limit process, in order
    # to give, for every h, the same impact on the global maximum
    matrix_normalized = []
    null_mean = [] # these parameters will be used to normalize both the +
    null_var = [] # limit_processes (H0) and the filter_rocesses (H1)
    
    for i,h in enumerate(window_sizes_mag):
        mean = np.mean(matrix, axis = 1)[i]
        var = np.var(matrix, axis = 1)[i]
        matrix_normalized.append((matrix [i] - mean)/np.sqrt(var))
        null_mean.append(mean)
        null_var.append(var)
        #print("Window {0}: Emp_mean {1}".format(h, mean))
        #print("Window {0}: Emp_var {1}".format(h, var))
    matrix_normalized = np.asanyarray(matrix_normalized) 
    
    great_maxs = np.max(matrix_normalized, axis = 0) # max over row   
    test_quantile = np.percentile(great_maxs, 100 - alfa)
    null_parameters= [window_sizes, null_mean, null_var]    
    test_param = np.asanyarray(null_parameters)
    
    return test_quantile, test_param


def _filter(t, h, spk):
    '''
    This function calculates the difference of spikes count in the left and right
    side of a window of size h centered in t. Normalized by its variance. 
    The variance of this count can be expressed as a combination of mean and var 
    of the I.S.I. lying inside the window.

    Parameter
    ---------
        h : quantity,
         window's size 
        t : quantity,
         time on which the window is centered
        spk : ist, array or SpikeTrain,
         spike train in analisys
    
    Returns
    -------
        difference : scalar, 
                  difference of spike count normalized by its variance

    '''
    
    u = 1*pq.s
    try:
        t_sec = t.rescale(u)
    except:
        raise ValueError ("t must be a time scalar")
    tm = t_sec.magnitude
    try:
        h_sec = h.rescale(u)
    except:
        raise ValueError ("h must be a time scalar")
    hm = h_sec.magnitude 
    try:
        spk = spk.rescale(u)
    except:
        raise ValueError ("Spk must be a list (array) of times or a neo spiketrain")
    
    # cut spike-train on the right
    train_right = spk[np.where((tm < spk) & (spk < tm+hm))]
    # cut spike-train on the left
    train_left = spk[np.where((tm-hm < spk) & (spk < tm)) ]
    # spike count in the right side
    count_right = spk[np.where((tm < spk) & (spk < tm+hm))].size
    # spike count in the left side
    count_left = spk[np.where((tm-hm < spk) & (spk < tm))].size

    isi_right = np.diff(train_right) # form spikes to I.S.I
    isi_left = np.diff(train_left) 
    
    mu_le=0
    mu_ri=0
    
    if isi_right.size == 0:
        mu_ri = 0
        sigma_ri = 0
    else:
        mu_ri = np.mean(isi_right) #mean of I.S.I inside the window
        sigma_ri = np.var(isi_right) #var of I.S.I inside the windo
        
    if isi_left.size==0:
        mu_le = 0
        sigma_le= 0
    else:     
        mu_le = np.mean(isi_left)
        sigma_le = np.var(isi_left)
    
      #if ((mu_le>0) & (mu_ri>0)):  
    if ((sigma_le > 0) & (sigma_ri > 0)): #sigma>0 imply also mu>0
        s_quad = [(sigma_ri/mu_ri**(3))*h + (sigma_le/mu_le**(3))*h]
    else:
        s_quad = 0
    
    if s_quad <= 0:
        difference = 0
    else:
        difference = (count_right - count_left) /  np.sqrt(s_quad)
    
    return difference
    
 
def _filter_process(dt, h, spk, t_final, test_param): 
    
    '''
    Given a spike train ìspk' and a window size h, this function generates the 
    'filter derivative process', by evaluating the function '_filter, in steps dt.

    Parameter
    ---------
        h : quantiy,
         window's size 
        t_final : quantity,
               time on which the window is centered
        spk : list, array or SpikeTrain,
           spike train in analisys
        dt : quantity,
          resolution
        test_param : matrix, frist row list of h, second row Empirical means and third
                  row variances of the limit processes Lh, 
                  used to normalize the number of elements inside the windows
                
    Returns:
    --------
        time_domain : array, 
                   time domain of the 'filter derivative process'
        filter_process : array, 
                      values of the 'filter derviative process'
    '''
    
    u = 1*pq.s 
    
    try:
        h_sec = h.rescale(u)
    except:
        raise ValueError ("h must be a time scalar")
    hm = h_sec.magnitude 
    try:
        t_final_sec = t_final.rescale(u)
    except:
        raise ValueError ("t_final must be a time scalar")   
    try:
        dt_sec = dt.rescale(u)
    except:
        raise ValueError ("dt must be a time scalar")
    ''' Control on lower level: _filter
    try:
        spk = spk.rescale(u)
    except:
        raise ValueError ("Spk must be a list (array) of times or a neo spiketrain")
    '''
    time_domain = np.arange(h_sec, t_final_sec - h_sec, dt_sec) #domain of the process
    time_domain_sec = time_domain*pq.s
    filter_trajectrory = []
   
    emp_mean_h = test_param[1][np.where(test_param[0]==hm)] #taken from the function
    emp_var_h =  test_param[2][np.where(test_param[0]==hm)] #used to generate the threshold
    
    for t in time_domain_sec:
        filter_trajectrory.append(_filter(t, h, spk))
        
    filter_trajectrory = np.asanyarray(filter_trajectrory)
    #Normailztion in orded to give each window the same impact on the max
    # which will be used as statistic
    filter_process = (np.abs(filter_trajectrory) - emp_mean_h) / np.sqrt(emp_var_h)
    
    return  time_domain, filter_process 
    



