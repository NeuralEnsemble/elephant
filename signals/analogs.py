"""
NeuroTools.signals.analogs
==================

A collection of functions to create, manipulate and play with analog signals. 

Classes
-------

None; uses neo classes.  

Functions
---------

load_vmarray          - function to load a VmArray object (inherits from 
                        AnalogSignalArray) from a file.
                        Same comments on format as previously.
load_currentarray     - function to load a CurrentArray object (inherits 
                        from AnalogSignalArray) from a file.
                        Same comments on format as previously.
load_conductancearray - function to load a ConductanceArray object (inherits 
                        from AnalogSignalArray) from a file.
                        Same comments on format as previously. 
                        load_conductancearray returns two ConductanceArrays, 
                        one for the excitatory conductance and one for the 
                        inhibitory conductance.
load                  - a generic loader for all the previous load methods.

See also NeuroTools.signals.spikes
"""

import os, re, numpy # Base libraries.
import neo # Other libraries.  
from NeuroTools import check_dependency, check_numpy_version
from NeuroTools.io import *
from NeuroTools.plotting import get_display, set_axis_limits, set_labels
from NeuroTools.plotting import SimpleMultiplot

if check_dependency('psyco'):
    import psyco
    psyco.full()

from NeuroTools import check_dependency
HAVE_PYLAB = check_dependency('pylab')
HAVE_MATPLOTLIB = check_dependency('matplotlib')
if HAVE_PYLAB:
    import pylab
else:
    PYLAB_ERROR = "The pylab package was not detected"
if not HAVE_MATPLOTLIB:
    MATPLOTLIB_ERROR = "The matplotlib package was not detected"

PLOT = False # Assume that we are not doing any plotting in these functions.  

newnum = check_numpy_version()
from pairs import *
from intervals import *
    
# Are we including plotting support?
def plot(analog_signal, 
         ylabel="Analog Signal", 
         display=True, 
         kwargs={}):
    """
    Plot the AnalogSignal
    
    Inputs:
        ylabel  - A string to sepcify the label on the yaxis.
        display - if True, a new figure is created. Could also be a subplot
        kwargs  - dictionary contening extra parameters that will be sent 
                  to the plot function
    
    Examples:
        >> z = subplot(221)
        >> signal.plot(ylabel="Vm", display=z, kwargs={'color':'r'})
    """
    subplot   = get_display(display)
    time_axis = analog_signal.time_axis()  
    if not subplot or not HAVE_PYLAB:
        print PYLAB_ERROR
    else:
        xlabel = "Time (ms)"
        set_labels(subplot, xlabel, ylabel)
        subplot.plot(time_axis, analog_signal.signal, **kwargs)
        pylab.draw()

def threshold_detection(analog_signal, 
                        threshold=None, 
                        format=None, 
                        sign='above'):
    """
    Returns the times when the analog signal crosses a threshold.
    The times can be returned as a numpy.array or a neo.SpikeTrain object
    (default)

    Inputs:
         analog_signal - A neo.AnalogSignal instance.  
         threshold     - Threshold
         format        - when 'raw' the raw events array is returned, 
                         otherwise this is a neo.SpikeTrain object by default
         sign          - 'above' detects when signal gets above the threshold, 
                         'below when it gets below the threshold'
            
    Examples:
        >> aslist.threshold_detection(-55, 'raw')
            [54.3, 197.4, 206]
    """
    
    assert threshold is not None, "A threshold must be provided"
    # Why is this an optional parameter if we forcing it to be not None?  

    if sign is 'above':
        cutout = numpy.where(analog_signal.signal > threshold)[0]
    elif sign in 'below':
        cutout = numpy.where(analog_signal.signal < threshold)[0]
        
    if len(cutout) <= 0:
        events = numpy.zeros(0)
    else:
        take = numpy.where(numpy.diff(cutout)>1)[0]+1
        take = numpy.append(0,take)
        
        time = analog_signal.time_axis()
        events = time[cutout][take]

    if format is 'raw':
        return events
    else:
        return neo.SpikeTrain(events,t_start=analog_signal.t_start,
                                 t_stop=analog_signal.t_stop)
        
                
def event_triggered_average(analog_signal, 
                            events, 
                            average = True, 
                            t_min = 0, 
                            t_max = 100, 
                            display = False, 
                            with_time = False, 
                            kwargs={}):
    """
    Return the spike triggered averaged of an analog signal according to 
    selected events, on a time window t_spikes - tmin, t_spikes + tmax
    Can return either the averaged waveform (average = True), or an array of 
    all the waveforms triggered by all the spikes.
    
    Inputs:
        spike_train - A neo.AnalogSignal instance.  
        events  - Can be a neo.SpikeTrain object (and events will be the spikes) 
                  or just a list of times
        average - If True, return a single vector of the averaged waveform. 
                  If False, return an array of all the waveforms.
        t_min   - Time (>0) to average the signal before an event, in ms 
                  (default 0)
        t_max   - Time (>0) to average the signal after an event, in ms 
                  (default 100)
        display - if True, a new figure is created. Could also be a subplot.
        kwargs  - dictionary contening extra parameters that will be sent to 
                  the plot function
        
    Examples:
        >> vm.event_triggered_average(spktrain, average=False, t_min = 50, 
                                                               t_max = 150)
        >> vm.event_triggered_average(spktrain, average=True)
        >> vm.event_triggered_average(range(0,1000,10), average=False, 
                                                        display=True)
    """
    
    if isinstance(events, neo.SpikeTrain):
        events = events.spike_times
        ylabel = "Spike Triggered Average"
    else:
        assert numpy.iterable(events), "events should be a neo.SpikeTrain \
                                        object or an iterable object"
        ylabel = "Event Triggered Average"
    assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater \
                                           than 0"
    assert len(events) > 0, "events should not be empty and should contained \
                             at least one element"
    time_axis = numpy.linspace(-t_min, t_max, (t_min+t_max)/analog_signal.dt)
    N         = len(time_axis)
    Nspikes   = 0.
    subplot   = get_display(display)
    if average:
        result = numpy.zeros(N, float)
    else:
        result = []
    
    # recalculate everything into timesteps, is more stable against rounding 
    # errors and subsequent cutouts with different sizes
    events = numpy.floor(numpy.array(events)/analog_signal.dt)
    t_min_l = numpy.floor(t_min/analog_signal.dt)
    t_max_l = numpy.floor(t_max/analog_signal.dt)
    t_start = numpy.floor(analog_signal.t_start/analog_signal.dt)
    t_stop = numpy.floor(analog_signal.t_stop/analog_signal.dt)
    
    for spike in events:
        if ((spike-t_min_l) >= t_start) and ((spike+t_max_l) < t_stop):
            spike = spike - t_start
            if average:
                result += analog_signal.signal[(spike-t_min_l):(spike+t_max_l)]
            else:
                result.append(\
                    analog_signal.signal[(spike-t_min_l):(spike+t_max_l)])
            Nspikes += 1
    if average:
        result = result/Nspikes
    else:
        result = numpy.array(result)
        
    if PLOT:
        if not subplot or not HAVE_PYLAB:
            if with_time:
                return result, time_axis
            else:
                return result
        else:
            xlabel = "Time (ms)"
            set_labels(subplot, xlabel, ylabel)
            if average:
                subplot.plot(time_axis, result, **kwargs)
            else:
                for idx in xrange(len(result)):
                    subplot.plot(time_axis, result[idx,:], c='0.5', **kwargs)
                    subplot.hold(1)
                result = numpy.sum(result, axis=0)/Nspikes
                subplot.plot(time_axis, result, c='k', **kwargs)
            xmin, xmax, ymin, ymax = subplot.axis()
                        
            subplot.plot([0,0],[ymin, ymax], c='r')
            set_axis_limits(subplot, -t_min, t_max, ymin, ymax)
            pylab.draw()

def slice_by_events(analog_signal,events,t_min=100,t_max=100):
    """
    Returns a dict containing new AnalogSignals cutout around events.

    Inputs:
        analog_signal - a neo.AnalogSignal instance.  
        events  - Can be a neo.SpikeTrain object (and events will be the 
                  spikes) or just a list of times
        t_min   - Time (>0) to cut the signal before an event, in ms
                  (default 100)
        t_max   - Time (>0) to cut the signal after an event, in ms
                  (default 100)
    
    Examples:
        >> res = aslist.slice_by_events(analog_signal, 
                                        [100,200,300], 
                                        t_min=0, 
                                        t_max =100)
        >> print len(res)
            3
    """
    if isinstance(events, neo.SpikeTrain):
        events = events.spike_times
    else:
        assert numpy.iterable(events), "events should be a SpikeTrain object \
                                       or an iterable object"
    assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should \
                                           be greater than 0"
    assert len(events) > 0, "events should not be empty and should contain \
                             at least one element"
    
    result = {}
    for index, spike in enumerate(events):
        if ((spike-t_min) >= analog_signal.t_start) and \
        ((spike+t_max) < analog_signal.t_stop):
            spike = spike - analog_signal.t_start
            t_start_new = (spike-t_min)
            t_stop_new = (spike+t_max)
            result[index] = analog_signal.time_slice(t_start_new, t_stop_new)
    return result

def mask_events(analog_signal,events,t_min=100,t_max=100):
    """
    Returns a new Analog signal which has analog_signal.signal of 
    numpy.ma.masked_array, where the internals (t_i-t_min, t_i+t_max) for 
    events={t_i} have been masked out.

    Inputs:
        analog_signal - a neo.AnalogSignal instance. 
        events  - Can be a SpikeTrain object (and events will be the spikes) or 
                  just a list of times
        t_min   - Time (>0) to cut the signal before an event, in ms 
                  (default 100)
        t_max   - Time (>0) to cut the signal after an event, in ms 
                  (default 100)
    
    Examples:
        >> res = signal.mask_events(analog_signal,[100,200,300], 
                                    t_min=0, 
                                    t_max =100)


    Author: Eilif Muller
    """
    from numpy import ma
    
    if isinstance(events, neo.SpikeTrain):
        events = events.spike_times
    else:
        assert numpy.iterable(events), "events should be a SpikeTrain object \
                                        or an iterable object"
    assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater \
                                          than 0"
    assert len(events) > 0, "events should not be empty and should contained \
                            at least one element"
    
    result = neo.AnalogSignal(analog_signal.signal, 
                              analog_signal.dt, 
                              analog_signal.t_start, 
                              analog_signal.t_stop)
    result.signal = ma.masked_array(result.signal, None)

    for index, spike in enumerate(events):
        t_start_new = numpy.max([spike-t_min, analog_signal.t_start])
        t_stop_new = numpy.min([spike+t_max, analog_signal.t_stop])
        
        i_start = int(round(t_start_new/analog_signal.dt))
        i_stop = int(round(t_stop_new/analog_signal.dt))
        result.signal.mask[i_start:i_stop]=True
                  
    return result

def slice_exclude_events(analog_signal,events,t_min=100,t_max=100):
    """
    yields new AnalogSignals with events cutout (useful for removing spikes).

    Events should be sorted in chronological order

    Inputs:
        analog_signal - a neo.AnalogSignal instance. 
        events  - Can be a neo.SpikeTrain object (and events will be the 
                  spikes) or just a list of times
        t_min   - Time (>0) to cut the signal before an event, in ms 
                  (default 100)
        t_max   - Time (>0) to cut the signal after an event, in ms 
                  (default 100)
    
    Examples:
        >> res = aslist.slice_exclude_events(analog_signal,
                                             [100,200,300], 
                                             t_min=100, 
                                             t_max=100)
        >> print len(res)
            4

    Author: Eilif Muller
    """
    if isinstance(events, neo.SpikeTrain):
        events = events.spike_times
    else:
        assert numpy.iterable(events), "events should be a neo.SpikeTrain \
                                        object or an iterable object"
    assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater \
                                           than 0"

    # if no events to remove, return analog_signal
    if len(events)==0:
        yield analog_signal
        return
    
    t_last = analog_signal.t_start
    for spike in events:
        # skip spikes which aren't close to the signal interval
        if spike+t_min<analog_signal.t_start or \
        spike-t_min>analog_signal.t_stop:
            continue
        
        t_min_local = numpy.max([t_last, spike-t_min])
        t_max_local = numpy.min([analog_signal.t_stop, spike+t_max])

        if t_last<t_min_local:
            yield analog_signal.time_slice(t_last, t_min_local)

        t_last = t_max_local

    if t_last<analog_signal.t_stop:
        yield analog_signal.time_slice(t_last, analog_signal.t_stop)

def cov(signal1,signal2):
    """

    Returns the covariance of two neo.AnalogSignals (signal1, signal2),

    i.e. mean(signal1*signal2)-mean(signal1)*(signal2)

    Inputs:
        signal1 - A neo.AnalogSignal instance.  
        signal2 - Another AnalogSignal instance.  
                  It should have the same temporal dimension and dt.
    
    Examples:
        >> a1 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
        >> a2 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
        >> print cov(a1,a2) # This is the covariance of a1 and a2.  
        -0.043763817072107143
        >> print cov(a1,a1) # This is the variance of a1.  
        1.0063757246782141

    See also:
        NeuroTools.analysis.ccf
        http://en.wikipedia.org/wiki/Covariance

    Author: Eilif Muller

    """

    from numpy import mean

    assert signal1.dt == signal2.dt
    assert signal1.signal.shape==signal2.signal.shape

    return mean(signal1.signal*signal2.signal) - \
           mean(signal1.signal)*mean(signal2.signal)

def load_conductance_array(user_file, 
                           id_list=None, 
                           dt=None, 
                           t_start=None, 
                           t_stop=None, 
                           dims=None):
    """
    Returns TWO neo.ConductanceArray objects from a file. One for the 
    excitatory and the other for the inhibitory conductance.
    If the file has been generated by PyNN, 
    a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text 
    file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, 
                    if a string is provided, a StandardTextFile object is 
                    created
        id_list  - the list of the recorded ids. Can be an int (meaning cells 
                   in the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the 
                   dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from 
    either the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    
    Examples:
        >> gexc, ginh = load_conductance_array("mydata.dat")
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="conductance", 
                                      id_list=id_list, 
                                      dt=dt, 
                                      t_start=t_start, 
                                      t_stop=t_stop, 
                                      dims=dims)

def load_vm_array(user_file, 
                  id_list=None, 
                  dt=None, 
                  t_start=0, 
                  t_stop=None, 
                  dims=None):
    """
    Returns a neo.VmArray object from a file. If the file has been generated by 
    PyNN, a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text 
    file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, 
                    if a string is provided, a StandardTextFile object is 
                    created
        id_list  - the list of the recorded ids. Can be an int (meaning cells 
                   in the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the 
                   dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from 
    either the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="vm", 
                                      id_list=id_list, 
                                      dt=dt, 
                                      t_start=t_start, 
                                      t_stop=t_stop, 
                                      dims=dims)

def load_current_array(user_file, 
                       id_list=None, 
                       dt=None, 
                       t_start=None, 
                       t_stop=None, 
                       dims=None):
    """
    Returns a neo.CurrentArray object from a file. If the file has been 
    generated by PyNN, a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text 
    file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, 
                    if a string is provided, a StandardTextFile object is 
                    created
        id_list  - the list of the recorded ids. Can be an int (meaning cells 
                   in the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the 
                   dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from 
    either the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="current", 
                                      id_list=id_list, 
                                      dt=dt, 
                                      t_start=t_start, 
                                      t_stop=t_stop, 
                                      dims=dims)


