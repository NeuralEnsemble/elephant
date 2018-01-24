# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:20:45 2016

@author: emanuele
"""

import numpy as np
import quantities as pq

'''
This algorithm is to determinate if a spike train can be considerate stationary
(costant rate) or not stationary (i.e. presence of one or more points at which the 
the rate increases or decreases). In case of nonstationary, the output is a list of detected 
Changing Points (CP).
Essentialy, a two-side window of width h (Rh(t)) slide over the time of the spike train [h,T-h].
This generates a 'filter derivative process' that at each time t assigns the difference 
between  spike lying in the right and left window. If at any time t this difference is large 
'enough', it is assumed the presence of a rate Change Point in a neighborhood of t.
A treshold Q for the maximum of the filter dervative process (max difference of spike
count between the left and right window) is derived based on asymptotic consideration.
The procedure is repeated for an arbitrary set of windows, with different size h.

Reference:
---------
Michael Messer, Marietta Kirchner, Julia Schiemann, Jochen Roeper,
Ralph Neininger, and Gaby Schneider. 'A multiple filter test for the
detection of rate changes in renewal processes with varying variance.'
Annals of Applied Statistics, 8(4):2027–2067, 12 2014.
'''

### Based on the published R implementation ###

def Brownian(Tin, Tfin, Xin, dt):
    '''
    Generate a BM.

    Parameters
    ----------
        Tin:    quantities, 
                initial time
        Tfin:   qiantities,
                final time
        Xin:    quantities,
                initial point
        dt:     quantities, 
                resolution
        Return
        ------
        Browniam motion on Tin-Tfin, with time step dt
    '''
    u = 1*pq.s
    try:
        Tin_sec = Tin.rescale(u)
    except:
        raise ValueError ("Tin must be a time quantity")
    Tin_m = Tin_sec.magnitude
    try:
        Tfin_sec = Tfin.rescale(u)
    except:
        raise ValueError ("Tfin must be a time quantity")
    Tfin_m = Tfin_sec.magnitude
    try:
        dt_sec = dt.rescale(u)
    except:
        raise ValueError ("dt must be a time quantity")
    Tfin_m = Tfin_sec.magnitude
    
    x = []
    for i in range(int((Tfin_m - Tin_m)/dt_sec)):
        x.append(np.random.normal(0,  np.sqrt(dt_sec)))
        
    s = np.cumsum(x)
    return s + Xin


def LtH(H, T, dt):
    '''
    Generate the limit processes (depending only on T and h), one for each h in H.
    The distribution of the maxima of these processes is used to derive the threshold Q.

    Parameters
    ---------
        H:      list of quantities, 
                set of windows' size 
        T:      quantity, F
                end of limit process
        dt:     quantities, 
                resolution 
    
    Return
    ------
        Lp:    list of array
               each entries contains the limit processes for each h,
               evaluated in [h,T-h] with steps dt
    '''    
    Lp = []
    
    u = 1*pq.s
    try:
        H_sec = H.rescale(u)
    except:
        raise ValueError ("H must be a list of times")
    Hm = H_sec.magnitude
    try:
        T_sec = T.rescale(u)
    except:
        raise ValueError ("T must be a time scalar")
    if dt is not None:
        try:
            dt_sec = dt.rescale(u)
            dtm = dt.magnitude
        except:
            raise ValueError ("dt must be a time scalar")
    
    for h in Hm: 
        if dt is None:   #automatic setting of dt
            dtm = h/20.
            dt_sec = dtm * pq.s
        else:
            dtm = dt.magnitude
                    
        T = T_sec-T_sec%(dt_sec)
        w = Brownian(0*pq.s,T,0,dtm)    
        
        w_plus = w[int(2*h/dtm):]   #BM on [h,T-h], shifted in time t-->t+h
        w_meno = w[:int(-2*h/dtm)]  #BM on [h,T-h], shifted in time t-->t-h
        ww = w[int(h/dtm):int(-h/dtm)]    #BM on [h,T-h]
      
        modul = (w_plus + w_meno - 2*ww)
        L = modul/(np.sqrt(2*h))
        Lp.append(L) 
#        
#        if dt == h/20.:
#            dt = None    
    return Lp

def CalcolateTresh(H, T, alfa, Num_sim, dt):    
    '''
    Generate the threshold.
    The Rh process has been proved to converge (for T,h-->infinity) to a continuous 
    funcional of a BM, called Lh.
    Simulating many of these limit processes (Num_sim), the maxima are collected:
        - fixed h in H, the maximum of Lh,t over t in [h,T-h] (called m_h) is taken
        - for each simulation the maximum of m_h over h in H is then taken

    The threshold will be considered the alfa quantile of this set of maxima.
    ----- Q := alfa quantile of {max(h in H)[max(t in T)Lh,t]} --------

    Parameters
    ----------
        H:        list of quantities,
                  set of windows' size 
        T:        quantity, 
                  final time of the spike
        alfa:     integer, 
                  alfa-quantile
        Num_sim:  integer,
                  numbers of simulated limit processes
        dt:       quantity,
                  resolution
    
    Return
    ------
        Q:          scalar,
                    Asyntotic treshold for the maximum of the filter derivative process
        Emp_param:  matrix, frist row list of h, second row Empirical means and third
                    row variances of the limit processes Lh. It will be used to 
                    normalize the number of elements inside the windows of differnt width h
 
    '''
    u = 1*pq.s
    try:
        H_sec = H.rescale(u)
    except:
        raise ValueError ("H must be a list of times")
    Hm = H_sec.magnitude
    try:
        T_sec = T.rescale(u)
    except:
        raise ValueError ("T must be a time scalar")
    Tm = T_sec.magnitude
        
    if Tm <= 0:
        raise ValueError ("T needs to be stricktly poisitive")
    if alfa*(100 - alfa) < 0:
        raise ValueError ("alfa needs to be in (0,100)")
    if not isinstance(Num_sim,int):
        raise TypeError ("numba of simulation needs to be integer")
    if np.min(Hm) <= 0:
        raise ValueError ("window's size needs to be stricktly poisitive")
    if np.max(Hm) >= T/2:
        raise ValueError ("window's size tooo large")
    if dt != None:
        try:
            dt = dt.rescale(u)
        except:
            raise ValueError ("dt must be a time scalar")        
        for h in Hm:
            if (h/dt).magnitude - int(h/dt)!=0:     
                raise ValueError("Every window size h must be a multiple of dt")

    a = []
#Generate a matrix with rows represent n = Num_sim realization 
# of the random variable M*_h, i.e. the variable max(t in T)Lh,t
#each row for each h in H. 
    for i in range(Num_sim):
            mh_star = []
            simu = LtH(H,T,dt)
            for i,h in enumerate(Hm):
                m_h = np.max(simu[i]) #max over T of the i-th limit process
                mh_star.append(m_h)
            a.append(mh_star)
                
    Mt = np.asanyarray(a)
    M = Mt.T
    
###Generate a matrix with rows represent n=Num_sim realization 
#  of the process M*_h, each row for each h in H.
#NORMALIZED for every h by the mean and the variance of its limit process 
    b = []
    Emp_mean = []  #These will be needed to normalize Rth
    Emp_var = []    
    
    for i,h in enumerate(Hm):
        mean = np.mean(M, axis = 1)[i]
        var = np.var(M, axis = 1)[i]
        b.append((M[i]-mean)/np.sqrt(var))
        Emp_mean.append(mean)
        Emp_var.append(var)
        print("Window {0}: Emp_mean {1}".format(h, mean))
        print("Window {0}: Emp_var {1}".format(h, var))
    M_norm = np.asanyarray(b) 
    
### n=Num_sim REALIZATION of the random variable M* 
#i.e. max(h in H)[max(t in T)Lh,t]!
    
    MSTAR = np.max(M_norm, axis = 0) #max over row   
    Q = np.percentile(MSTAR,100 - alfa)
    Emp = [H,Emp_mean,Emp_var]    
    Emp_param = np.asanyarray(Emp)
    
    return Q, Emp_param


def Gth(t ,h ,spk):
    '''
    Calculates the difference of spikes count in the left and right side of the window;
    normalized by the variance of this count. This variance is obtained as a combination
    of mean and var of the I.S.I.(Internal-spike-interval) lying inside the window.

    Parameters
    ---------
        h:     quantity,
               window's size 
        t:     quantity,
               time on which the window is centered
        spk:   list, array or SpikeTrain,
               spike train in analisys
    
    Return
    ------
        g:  scalar, 
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
    
#TODO: remove useless np.where    
    N_ri = spk[np.where((tm < spk) & (spk < tm+hm))]#cut spike-train on the right
    N_le = spk[np.where((tm-hm < spk) & (spk < tm))]#cut spike-train on the left
    
    n_ri = spk[np.where((tm < spk) & (spk < tm+hm))].size#count spikes in the right side
    n_le = spk[np.where((tm-hm < spk) & (spk < tm))].size#'''

    gamma_ri = np.diff(N_ri)#form spikes to I.S.I
    gamma_le = np.diff(N_le)
    
    mu_le=0
    mu_ri=0
    
    if gamma_ri.size==0:
        mu_ri = 0
        sigma_ri = 0
    else:
        mu_ri = np.mean(gamma_ri) #mean of I.S.I inside the window
        sigma_ri = np.var(gamma_ri) #var of I.S.I inside the windo
        
    if gamma_le.size==0:
        mu_le = 0
        sigma_le= 0
    else:     
        mu_le = np.mean(gamma_le)
        sigma_le = np.var(gamma_le)
    
      #if ((mu_le>0) & (mu_ri>0)):  
    if ((sigma_le > 0) & (sigma_ri > 0)): #sigma>0 imply also mu>0
        Squad = [(sigma_ri/mu_ri**(3))*h + (sigma_le/mu_le**(3))*h]
    else:
        Squad = 0
    
    if Squad <= 0:
        g = 0
    else:
        g = (n_ri - n_le) /  np.sqrt(Squad)
    
    return g
    
    
def Rth(dt, h, spk, T, Emp_param): 
    '''
    Given a spike train and a window size h generate the 'filter derivative process',
    by evaluating the function Gth above, at steps dt.

    Parameters
    ---------
        h:          quantiy,
                    window's size 
        T:          quantity,
                    time on which the window is centered
        spk:        list, array or SpikeTrain,
                    spike train in analisys
        dt:         quantity,
                    resolution
        Emp_param:  matrix, frist row list of h, second row Empirical means and third
                    row variances of the limit processes Lh, 
                    used to normalize the number of elements inside the windows
                
    Return
    ------
        tauh: array, time domain of the 'filter derivative process'
        Rh: array, values of the 'filter derviative process'
    '''
    u = 1*pq.s 
    try:
        h_sec = h.rescale(u)
    except:
        raise ValueError ("h must be a time scalar")
    hm = h_sec.magnitude 
    try:
        T_sec = T.rescale(u)
    except:
        raise ValueError ("T must be a time scalar")   
    try:
        dt_sec = dt.rescale(u)
    except:
        raise ValueError ("dt must be a time scalar")
    try:
        spk = spk.rescale(u)
    except:
        raise ValueError ("Spk must be a list (array) of times or a neo spiketrain")
    
    tauh = np.arange(h_sec,T_sec-h_sec,dt_sec) #domain of the process
    tauh_sec = tauh*pq.s
    traiet_gh = []
   
    Emp_meanh = Emp_param[1][np.where(Emp_param[0]==hm)] #taken from the function
    Emp_varh =  Emp_param[2][np.where(Emp_param[0]==hm)] #used to generate the threshold
    
    for t in tauh_sec:
        traiet_gh.append(Gth(t,h,spk))
        
    gh = np.asanyarray(traiet_gh)
    Rh = (np.abs(gh) - Emp_meanh) / np.sqrt(Emp_varh)#Normailztion in orded
                     #to give each window the same impact on the max

    return  tauh, Rh
    
### °*°*°*° $|$ Multiple Filter Algorithm $|$ °*°*°*° ###

def MultipleFilterAlgorithm(H, spk, T, alfa, Num_sim, Q = None, Emp_param = None, dt = None):
    '''      
    Parameters
    ----------
        H:          list of quantities,
                    set of windows' size 
        spk:        list, array or SpikeTrain,
                    spike train in analisys
        T:          quantity, 
                    Final time of the spike train to be analysed
        alfa:       integer, 
                    alfa-quantile
        Num_sim:    integer, numbers of simulated limit processes
        dt:         quantity, 
                    resolution
        Q:          scalar,
                    Asyntotic treshold
        Emp_param:  matrix, frist row list of h, second row Empirical means and third
                    row variances of the limit processes Lh. It will be used to 
                    normalize the number of elements inside the windows of differnt width h
    Return
    ------
        C:          list of lists: one list for each h in increasing order contaning
                    the points detected with each window h.
    '''
    if (Q is None)&(Emp_param is None):
        Q,Emp_param = CalcolateTresh(H,T,alfa,Num_sim,dt)
    elif Q is None:
        Q = CalcolateTresh(H,T,alfa,Num_sim,dt)[0]
    elif Emp_param is None:
        Emp_param = CalcolateTresh(H,T,alfa,Num_sim,dt)[1]
        
    C = []      # List of list of dected point, to be returned
  
    for i,h in enumerate(H):
        if dt is None:   #automatic setting of dt     
            dt = h/20.
            
        t,r = Rth(dt,h,spk,T,Emp_param) #Process of the sliding window h
        times = np.where(r<1000)[0] #just to take ALL times
     
        Ch = [] #Point detected by window h
        while (np.max(r) > Q):
            cci = np.argmax(r)
            ci = cci*dt + h #from index to time
            print "detected point {0}".format(ci), "with filter {0}".format(h)
            r[np.where((times > cci-int(h/dt.rescale(h.units))) & (times < cci+int(h/dt.rescale(h.units))))[0]] = 0 
            #before to repet the procedure cut the h-neighbourg of the point  
            #just detected 'cause the change in rate inside it is alredy explained            
            
# The list of Change Points detected with the smallest window is first given as output.
# Then for the list of increasing window size h, only those whose h-neighborhood does
# not include other CPs already listed (with smaller hs) are given as output
            ok = 1
            if i == 0:
                Ch.append(ci)
            else: 
                for j in range(i):   #control on previous filter h
                    for c_pre in C[j]: #control on prev.det.point with that h above
                        if (c_pre-h < ci < c_pre+h):
                            ok = 0
                            break
                if (ok == 1):
                    Ch.append(ci)         
        C.append(Ch)
        
        if dt == h/20.:
            dt = None
# TODO: decide whether to change shape of output                  
    return C
