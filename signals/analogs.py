"""
NeuroTools.signals.analogs
==================

A collection of functions to create, manipulate and play with analog signals. 

Classes
-------

AnalogSignal     - object representing an analog signal, with its data. Can be used to do 
                   threshold detection, event triggered averages, ...
AnalogSignalList - list of AnalogSignal objects, again with methods such as mean, std, plot, 
                   and so on
VmList           - AnalogSignalList object used for Vm traces
ConductanceList  - AnalogSignalList object used for conductance traces
CurrentList      - AnalogSignalList object used for current traces

Functions
---------

load_vmlist          - function to load a VmList object (inherits from AnalogSignalList) from a file.
                       Same comments on format as previously.
load_currentlist     - function to load a CurrentList object (inherits from AnalogSignalList) from a file.
                       Same comments on format as previously.
load_conductancelist - function to load a ConductanceList object (inherits from AnalogSignalList) from a file.
                       Same comments on format as previously. load_conductancelist returns two 
                       ConductanceLists, one for the excitatory conductance and one for the inhibitory conductance
load                 - a generic loader for all the previous load methods.

See also NeuroTools.signals.spikes
"""

import os, re, numpy
from NeuroTools import check_dependency, check_numpy_version
from NeuroTools.io import *
from NeuroTools.plotting import get_display, set_axis_limits, set_labels, SimpleMultiplot

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


newnum = check_numpy_version()
from spikes import SpikeList, SpikeTrain
from pairs import *
from intervals import *


class AnalogSignal(object):
    """
    AnalogSignal(signal, dt, t_start=0, t_stop=None)
    
    Return a AnalogSignal object which will be an analog signal trace

    Inputs:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - begining of the signal, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be inferred from the data
    
    Examples:
        >> s = AnalogSignal(range(100), dt=0.1, t_start=0, t_stop=10)

    See also
        AnalogSignalList, load_currentlist, load_vmlist, load_conductancelist, load
    """
    def __init__(self, signal, dt, t_start=0, t_stop=None):
        logging.debug("Creating AnalogSignal. len(signal)=%d, dt=%g, t_start=%g, t_stop=%s" % (len(signal), dt, t_start, t_stop))
        self.signal  = numpy.array(signal, float)
        self.dt      = float(dt)
        if t_start is None:
            t_start = 0
        self.t_start = float(t_start)
        self.t_stop  = t_stop
        # If t_stop is not None, we test that the signal has the correct number
        # of elements
        if self.t_stop is not None:
            if abs(self.t_stop-self.t_start - self.dt * len(self.signal)) > 0.1*self.dt:
                raise Exception("Inconsistent arguments: t_start=%g, t_stop=%g, dt=%g implies %d elements, actually %d" % (
                                    t_start, t_stop, dt, int(round((t_stop-t_start)/float(dt))), len(signal)))
        else:
            self.t_stop = self.t_start + len(self.signal)*self.dt

        # TODO raise an error if some data is outside [t_start, t_stop] ?
        # TODO return an exception if self.t_stop < self.t_start (when not empty)
        if self.t_start >= self.t_stop:
            raise Exception("Incompatible time interval for the creation of the AnalogSignal. t_start=%s, t_stop=%s" % (self.t_start, self.t_stop))

    def __getslice__(self, i, j):
        """
        Return a sublist of the signal vector of the AnalogSignal
        """
        return self.signal[i:j]

    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self.t_stop - self.t_start

    def __str__(self):
        return str(self.signal)

    def __len__(self):
        return len(self.signal)

    def max(self):
        return self.signal.max()

    def min(self):
        return self.signal.min()
    
    def mean(self):
        return numpy.mean(self.signal)

    def copy(self):
        """
        Return a copy of the AnalogSignal object
        """
        return AnalogSignal(self.signal, self.dt, self.t_start, self.t_stop)

    def time_axis(self, normalized=False):
        """
        Return the time axis of the AnalogSignal
        """
        if normalized:
            norm = self.t_start
        else:
            norm = 0.
        return numpy.arange(self.t_start-norm, self.t_stop-norm, self.dt)
    
    def time_offset(self, offset):
        """
        Add a time offset to the AnalogSignal object. t_start and t_stop are
        shifted from offset.
         
        Inputs:
            offset - the time offset, in ms
        
        Examples:
            >> as = AnalogSignal(arange(0,100,0.1),0.1)
            >> as.t_stop
                100
            >> as.time_offset(1000)
            >> as.t_stop
                1100
        """
        self.t_start += offset
        self.t_stop  += offset

    
    def time_parameters(self):
        """
        Return the time parameters of the AnalogSignal (t_start, t_stop, dt)
        """
        return (self.t_start, self.t_stop, self.dt)
    
    
    def plot(self, ylabel="Analog Signal", display=True, kwargs={}):
        """
        Plot the AnalogSignal
        
        Inputs:
            ylabel  - A string to sepcify the label on the yaxis.
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> signal.plot(ylabel="Vm", display=z, kwargs={'color':'r'})
        """
        subplot   = get_display(display)
        time_axis = self.time_axis()  
        if not subplot or not HAVE_PYLAB:
            print PYLAB_ERROR
        else:
            xlabel = "Time (ms)"
            set_labels(subplot, xlabel, ylabel)
            subplot.plot(time_axis, self.signal, **kwargs)
            pylab.draw()
    
    
    def time_slice(self, t_start, t_stop):
        """ 
        Return a new AnalogSignal obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.
        
        See also:
            interval_slice
        """
        assert t_start >= self.t_start
        assert t_stop <= self.t_stop
        assert t_stop > t_start
        
        t = self.time_axis()
        i_start = int(round((t_start-self.t_start)/self.dt))
        i_stop = int(round((t_stop-self.t_start)/self.dt))
        signal = self.signal[i_start:i_stop]
        result = AnalogSignal(signal, self.dt, t_start, t_stop)
        return result

    def interval_slice(self, interval):
        """
        Return only the parts of the AnalogSignal that are defined in the range of the interval. 
        The output is therefor a list of signal segments
        
        Inputs:
            interval - The Interval to slice the AnalogSignal with
        
        Examples:
            >> as.interval_slice(Interval([0,100],[50,150]))

        See also:
            time_slice
        """
        result = []
        for itv in interval.interval_data :
            result.append(self.signal[intv[0]/self.dt:intv[1]/self.dt])


    def threshold_detection(self, threshold=None, format=None,sign='above'):
        """
        Returns the times when the analog signal crosses a threshold.
        The times can be returned as a numpy.array or a SpikeTrain object
        (default)

        Inputs:
             threshold - Threshold
             format    - when 'raw' the raw events array is returned, 
                         otherwise this is a SpikeTrain object by default
             sign      - 'above' detects when the signal gets above the threshodl, 'below when it gets below the threshold'
                
        Examples:
            >> aslist.threshold_detection(-55, 'raw')
                [54.3, 197.4, 206]
        """
        
        assert threshold is not None, "threshold must be provided"

        if sign is 'above':
            cutout = numpy.where(self.signal > threshold)[0]
        elif sign in 'below':
            cutout = numpy.where(self.signal < threshold)[0]
            
        if len(cutout) <= 0:
            events = numpy.zeros(0)
        else:
            take = numpy.where(numpy.diff(cutout)>1)[0]+1
            take = numpy.append(0,take)
            
            time = self.time_axis()
            events = time[cutout][take]

        if format is 'raw':
            return events
        else:
            return SpikeTrain(events,t_start=self.t_start,t_stop=self.t_stop)
            
                    
    def event_triggered_average(self, events, average = True, t_min = 0, t_max = 100, display = False, with_time = False, kwargs={}):
        """
        Return the spike triggered averaged of an analog signal according to selected events, 
        on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.
        
        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list 
                      of times
            average - If True, return a single vector of the averaged waveform. If False, 
                      return an array of all the waveforms.
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)
            display - if True, a new figure is created. Could also be a subplot.
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
            
        Examples:
            >> vm.event_triggered_average(spktrain, average=False, t_min = 50, t_max = 150)
            >> vm.event_triggered_average(spktrain, average=True)
            >> vm.event_triggered_average(range(0,1000,10), average=False, display=True)
        """
        
        if isinstance(events, SpikeTrain):
            events = events.spike_times
            ylabel = "Spike Triggered Average"
        else:
            assert numpy.iterable(events), "events should be a SpikeTrain object or an iterable object"
            ylabel = "Event Triggered Average"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        assert len(events) > 0, "events should not be empty and should contained at least one element"
        time_axis = numpy.linspace(-t_min, t_max, (t_min+t_max)/self.dt)
        N         = len(time_axis)
        Nspikes   = 0.
        subplot   = get_display(display)
        if average:
            result = numpy.zeros(N, float)
        else:
            result = []
        
        # recalculate everything into timesteps, is more stable against rounding errors
        # and subsequent cutouts with different sizes
        events = numpy.floor(numpy.array(events)/self.dt)
        t_min_l = numpy.floor(t_min/self.dt)
        t_max_l = numpy.floor(t_max/self.dt)
        t_start = numpy.floor(self.t_start/self.dt)
        t_stop = numpy.floor(self.t_stop/self.dt)
        
        for spike in events:
            if ((spike-t_min_l) >= t_start) and ((spike+t_max_l) < t_stop):
                spike = spike - t_start
                if average:
                    result += self.signal[(spike-t_min_l):(spike+t_max_l)]
                else:
                    result.append(self.signal[(spike-t_min_l):(spike+t_max_l)])
                Nspikes += 1
        if average:
            result = result/Nspikes
        else:
            result = numpy.array(result)
            
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

    def slice_by_events(self,events,t_min=100,t_max=100):
        """
        Returns a dict containing new AnalogSignals cutout around events.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list 
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)
        
        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =100)
            >> print len(res)
                3
        """
        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert numpy.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        assert len(events) > 0, "events should not be empty and should contained at least one element"
        
        result = {}
        for index, spike in enumerate(events):
            if ((spike-t_min) >= self.t_start) and ((spike+t_max) < self.t_stop):
                spike = spike - self.t_start
                t_start_new = (spike-t_min)
                t_stop_new = (spike+t_max)
                result[index] = self.time_slice(t_start_new, t_stop_new)
        return result



    def mask_events(self,events,t_min=100,t_max=100):
        """
        Returns a new Analog signal which has self.signal of numpy.ma.masked_array, where the internals (t_i-t_min, t_i+t_max) for events={t_i}
        have been masked out.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list 
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)
        
        Examples:
            >> res = signal.mask_events([100,200,300], t_min=0, t_max =100)


        Author: Eilif Muller
        """
        from numpy import ma
        
        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert numpy.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        assert len(events) > 0, "events should not be empty and should contained at least one element"
        
        result = AnalogSignal(self.signal, self.dt, self.t_start, self.t_stop)
        result.signal = ma.masked_array(result.signal, None)

        for index, spike in enumerate(events):
            t_start_new = numpy.max([spike-t_min, self.t_start])
            t_stop_new = numpy.min([spike+t_max, self.t_stop])
            
            i_start = int(round(t_start_new/self.dt))
            i_stop = int(round(t_stop_new/self.dt))
            result.signal.mask[i_start:i_stop]=True
                      
        return result



    def slice_exclude_events(self,events,t_min=100,t_max=100):
        """
        yields new AnalogSignals with events cutout (useful for removing spikes).

        Events should be sorted in chronological order

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list 
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)
        
        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =10)
            >> print len(res)
                4

        Author: Eilif Muller
        """
        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert numpy.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"

        # if no events to remove, return self
        if len(events)==0:
            yield self
            return
        
        t_last = self.t_start
        for spike in events:
            # skip spikes which aren't close to the signal interval
            if spike+t_min<self.t_start or spike-t_min>self.t_stop:
                continue
            
            t_min_local = numpy.max([t_last, spike-t_min])
            t_max_local = numpy.min([self.t_stop, spike+t_max])

            if t_last<t_min_local:
                yield self.time_slice(t_last, t_min_local)

            t_last = t_max_local

        if t_last<self.t_stop:
            yield self.time_slice(t_last, self.t_stop)

    def cov(self,signal):
        """

        Returns the covariance of two signals (self, signal),

        i.e. mean(self.signal*signal)-mean(self.signal)*(mean(signal)


        Inputs:
            signal  - Another AnalogSignal object.  It should have the same temporal dimension
                      and dt.
        
        Examples:
            >> a1 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> a2 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> print a1.cov(a2)
            -0.043763817072107143
            >> print a1.cov(a1)
            1.0063757246782141

        See also:
            NeuroTools.analysis.ccf
            http://en.wikipedia.org/wiki/Covariance

        Author: Eilif Muller

        """

        from numpy import mean

        assert signal.dt == self.dt
        assert signal.signal.shape==self.signal.shape

        return mean(self.signal*signal.signal)-mean(self.signal)*(mean(signal.signal))
    


class AnalogSignalList(object):
    """
    AnalogSignalList(signals, id_list, dt=None, t_start=None, t_stop=None, dims=None)
    
    Return a AnalogSignalList object which will be a list of AnalogSignal objects.

    Inputs:
        signals - a list of tuples (id, value) with all the values sorted in time of the analog signals
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        dt      - if dt is specified, time values should be floats
        t_start - begining of the SpikeList, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be infered from the data
        dims    - dimensions of the recorded population, if not 1D population
    
    dt, t_start and t_stop are shared for all SpikeTrains object within the SpikeList
    
    See also
        load_currentlist load_vmlist, load_conductancelist
    """
    def __init__(self, signals, id_list, dt, t_start=0, t_stop=None, dims=None):
        #logging.debug("Creating an AnalogSignalList. len(signals)=%d, min(id_list)=%d, max(id_list)=%d, dt=%g, t_start=%g, t_stop=%s, dims=%s" % \
        #                 (len(signals), min(id_list), max(id_list), dt, t_start, t_stop, dims))
        
        if t_start is None:
            t_start = 0
        self.t_start        = float(t_start)
        self.t_stop         = t_stop
        self.dt             = float(dt)
        self.dimensions     = dims
        self.analog_signals = {}
        
        signals = numpy.array(signals)
        for id in id_list:
            signal = numpy.transpose(signals[signals[:,0] == id, 1:])[0]
            if len(signal) > 0:
                self.analog_signals[id] = AnalogSignal(signal, self.dt, self.t_start, self.t_stop)
        
        if id_list:
            signals = self.analog_signals.values()
            self.signal_length = len(signals[0])
            for signal in signals[1:]:
                if len(signal) != self.signal_length:
                    raise Exception("Signals must all be the same length %d != %d" % (self.signal_length, len(signal)))
        else:
            logging.warning("id_list is empty")
            self.signal_length = 0
        
        if t_stop is None:
            self.t_stop = self.t_start + self.signal_length*self.dt

    def id_list(self):
        """ 
        Return the list of all the cells ids contained in the
        SpikeList object
        """
        return numpy.array(self.analog_signals.keys())

    def copy(self):
        """
        Return a copy of the AnalogSignalList object
        """
        # Maybe not optimal, should be optimized
        aslist = AnalogSignalList([], [], self.dt, self.t_start, self.t_stop, self.dimensions)
        for id in self.id_list():
            aslist.append(id, self.analog_signals[id])
        return aslist
    
    def __getitem__(self, id):
        if id in self.id_list():
            return self.analog_signals[id]
        else:
            raise Exception("id %d is not present in the AnalogSignal. See id_list()" %id)

    def __setitem__(self, i, val):
        assert isinstance(val, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
        if len(self) > 0:
            errmsgs = []
            for attr in "dt", "t_start", "t_stop":
                if getattr(self, attr) == 0:
                    if getattr(val, attr) != 0:
                        errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr), getattr(val, attr)-getattr(self, attr)))
                elif (getattr(val, attr) - getattr(self, attr))/getattr(self, attr) > 1e-12:
                    errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr), getattr(val, attr)-getattr(self, attr)))
            if len(val) != self.signal_length:
                errmsgs.append("signal length: %g != %g" % (len(val), self.signal_length))
            if errmsgs:
                raise Exception("AnalogSignal being added does not match the existing signals: "+", ".join(errmsgs))
        else:
            self.signal_length = len(val)
            self.t_start = val.t_start
            self.t_stop = val.t_stop
        self.analog_signals[i] = val

    def __len__(self):
        return len(self.analog_signals)
    
    def __iter__(self):
        return self.analog_signals.itervalues()

    def __sub_id_list(self, sub_list=None):
        if sub_list == None:
            return self.id_list()
        if type(sub_list) == int:
            return numpy.random.permutation(self.id_list())[0:sub_list]
        if type(sub_list) == list:
            return sub_list

    def append(self, id, signal):
        """
        Add an AnalogSignal object to the AnalogSignalList
        
        Inputs:
            id     - the id of the new cell
            signal - the AnalogSignal object representing the new cell
        
        The AnalogSignal object is sliced according to the t_start and t_stop times
        of the AnalogSignallist object
        
        See also
            __setitem__
        """
        assert isinstance(signal, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
        if id in self.id_list():
            raise Exception("Id already present in AnalogSignalList. Use setitem instead()")
        else:
            self[id] = signal

    def time_axis(self):
        """
        Return the time axis of the AnalogSignalList object
        """
        return numpy.arange(self.t_start, self.t_stop, self.dt)

    def id_offset(self, offset):
        """
        Add an offset to the whole AnalogSignalList object. All the id are shifted
        with a offset value.
         
        Inputs:
            offset - the id offset
        
        Examples:
            >> as.id_list()
                [0,1,2,3,4]
            >> as.id_offset(10)
            >> as.id_list()
                [10,11,12,13,14]
        """
        id_list = numpy.sort(self.id_list())
        N       = len(id_list)
        for idx in xrange(1,len(id_list)+1):
            id  = id_list[N-idx]
            spk = self.analog_signals.pop(id)
            self.analog_signals[id + offset] = spk

    def id_slice(self, id_list):
        """
        Return a new AnalogSignalList obtained by selecting particular ids
        
        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids
        
        The new AnalogSignalList inherits the time parameters (t_start, t_stop, dt)
        
        See also
            time_slice
        """
        new_AnalogSignalList = AnalogSignalList([], [], self.dt, self.t_start, self.t_stop, self.dimensions)
        id_list = self.__sub_id_list(id_list)
        for id in id_list:
            try:
                new_AnalogSignalList.append(id, self.analog_signals[id])
            except Exception:
                print "id %d is not in the source AnalogSignalList" %id
        return new_AnalogSignalList

    def time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.
        
        See also
            id_slice
        """
        new_AnalogSignalList = AnalogSignalList([], [], self.dt, t_start, t_stop, self.dimensions)
        for id in self.id_list():
            new_AnalogSignalList.append(id, self.analog_signals[id].time_slice(t_start, t_stop))
        return new_AnalogSignalList

    def select_ids(self, criteria=None):
        """
        Return the list of all the cells in the AnalogSignalList that will match the criteria
        expressed with the following syntax. 
        
        Inputs : 
            criteria - a string that can be evaluated on a AnalogSignal object, where the 
                       AnalogSignal should be named ``cell''.
        
        Exemples:
            >> aslist.select_ids("mean(cell.signal) > 20")
            >> aslist.select_ids("cell.std() < 0.2")
        """
        selected_ids = []
        for id in self.id_list():
            cell = self.analog_signals[id]
            if eval(criteria):
                selected_ids.append(id)
        return selected_ids

    def convert(self, format="[values, ids]"):
        """
        Return a new representation of the AnalogSignalList object, in a user designed format.
            format is an expression containing either the keywords values and ids, 
            time and id.
       
        Inputs:
            format    - A template to generate the corresponding data representation, with the keywords
                        values and ids

        Examples: 
            >> aslist.convert("[values, ids]") will return a list of two elements, the 
                first one being the array of all the values, the second the array of all the
                corresponding ids, sorted by time
            >> aslist.convert("[(value,id)]") will return a list of tuples (value, id)
        """
        is_values = re.compile("values")
        is_ids   = re.compile("ids")
        values = numpy.concatenate([st.signal for st in self.analog_signals.itervalues()])
        ids    = numpy.concatenate([id*numpy.ones(len(st.signal), int) for id,st in self.analog_signals.iteritems()])
        if is_values.search(format):
            if is_ids.search(format):
                return eval(format)
            else:
                raise Exception("You must have a format with [values, ids] or [value, id]")
        is_values = re.compile("value")
        is_ids   = re.compile("id")
        if is_values.search(format):
            if is_ids.search(format):
                result = []
                for id, time in zip(ids, values):
                    result.append(eval(format))
            else:
                raise Exception("You must have a format with [values, ids] or [value, id]")
            return result


    def raw_data(self):
        """
        Function to return a N by 2 array of all values and ids.
        
        Examples:
            >> spklist.raw_data()
            >> array([[  1.00000000e+00,   1.00000000e+00],
                      [  1.00000000e+00,   1.00000000e+00],
                      [  2.00000000e+00,   2.00000000e+00],
                         ...,
                      [  2.71530000e+03,   2.76210000e+03]])
        
        See also:
            convert()
        """
        data = numpy.array(self.convert("[values, ids]"))
        data = numpy.transpose(data)
        return data

    def save(self, user_file):
        """
        Save the AnalogSignal in a text or binary file
        
            user_file - The user file that will have its own read/write method
                        By default, if s tring is provided, a StandardTextFile object
                        will be created. Nevertheless, you can also
                        provide a StandardPickleFile
        Examples:
            >> a.save("data.txt")
            >> a.save(StandardTextFile("data.txt"))
            >> a.save(StandardPickleFile("data.pck"))
        """
        as_loader = DataHandler(user_file, self)
        as_loader.save()
    
    def mean(self):
        """
        Return the mean AnalogSignal after having performed the average of all the signals
        present in the AnalogSignalList
        
        Examples:
            >> a.mean()
        
        See also:
            std
        """
        result = numpy.zeros(int((self.t_stop - self.t_start)/self.dt),float)
        for id in self.id_list():
             result += self.analog_signals[id].signal
        return result/len(self)
    
    def std(self):
        """
        Return the standard deviation along time between all the AnalogSignals contained in
        the AnalogSignalList
        
        Examples:
            >> a.std()
               numpy.array([0.01, 0.2404, ...., 0.234, 0.234]
               
        See also:
            mean
        """
        result = numpy.zeros((len(self), int(round((self.t_stop - self.t_start)/self.dt))), float)
        for count, id in enumerate(self.id_list()):
            try:
                result[count,:] = self.analog_signals[id].signal
            except ValueError:
                print result[count,:].shape, self.analog_signals[id].signal.shape
                raise
        return numpy.std(result, axis=0)

    def event_triggered_average(self, eventdict, events_ids = None, analogsignal_ids = None, average = True, t_min = 0, t_max = 100, ylim = None, display = False, mode = 'same', kwargs={}):
        """
        Returns the event triggered averages of the analog signals inside the list.
        The events can be a SpikeList object or a dict containing times.
        The average is performed on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.
        
        Inputs:
            events  - Can be a SpikeList object (and events will be the spikes) or just a dict 
                      of times
            average - If True, return a single vector of the averaged waveform. If False, 
                      return an array of all the waveforms.
            mode    - 'same': the average is only done on same ids --> return {'eventids':average};
                      'all': for all ids in the eventdict the average from all ananlog signals is returned --> return {'eventids':{'analogsignal_ids':average}}
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)
            events_ids - when given only perform average over these ids
            analogsignal_ids = when given only perform average on these ids
            display - if True, a new figure is created for each average. Could also be a subplot.
            ylim    - ylim of the plot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
            
        Examples
            >> vmlist.event_triggered_average(spikelist, average=False, t_min = 50, t_max = 150, mode = 'same')
            >> vmlist.event_triggered_average(spikelist, average=True, mode = 'all')
            >> vmlist.event_triggered_average({'1':[200,300,'3':[234,788]]}, average=False, display=True)
        """
        if isinstance(eventdict, SpikeList):
            eventdict = eventdict.spiketrains
        figure   = get_display(display)
        subplotcount = 1
        
        if events_ids is None:
            events_ids = eventdict.keys()
        if analogsignal_ids is None:
            analogsignal_ids = self.analog_signals.keys()

        x = numpy.ceil(numpy.sqrt(len(analogsignal_ids)))
        y = x
        results = {}

        first_done = False
        
        for id in events_ids:
            events = eventdict[id]
            if len(events) <= 0:
                continue
            if mode is 'same':
                
                if self.analog_signals.has_key(id) and id in analogsignal_ids:
                    sp = pylab.subplot(x,y,subplotcount)
                    results[id] = self.analog_signals[id].event_triggered_average(events,average=average,t_min=t_min,t_max=t_max,display=sp,kwargs=kwargs)
                    pylab.ylim(ylim)
                    pylab.title('Event: %g; Signal: %g'%(id,id))
                    subplotcount += 1
            elif mode is 'all':
                if first_done:
                    figure   = get_display(display)
                first_done = True
                subplotcount_all = 1
                results[id] = {}
                for id_analog in analogsignal_ids:
                    analog_signal = self.analog_signals[id_analog]
                    sp = pylab.subplot(x,y,subplotcount_all)
                    results[id][id_analog] = analog_signal.event_triggered_average(events,average=average,t_min=t_min,t_max=t_max,display=sp,kwargs=kwargs)
                    pylab.ylim(ylim)
                    pylab.title('Event: %g; Signal: %g'%(id,id_analog))
                    subplotcount_all += 1

        if not figure or not HAVE_PYLAB:
            return results


class VmList(AnalogSignalList):

    def plot(self, id_list=None, v_thresh=None, display=True, kwargs={}):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            v_thresh- For graphical purpose, plot a spike when Vm > V_thresh. If None, 
                      just plot the raw Vm
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, v_thresh = -50, display=z, kwargs={'color':'r'})
        """
        subplot   = get_display(display)
        id_list   = self._AnalogSignalList__sub_id_list(id_list)
        time_axis = self.time_axis()  
        if not subplot or not HAVE_MATPLOTLIB:
            print MATPLOTLIB_ERROR
        else:
            xlabel = "Time (ms)"
            ylabel = "Membrane Potential (mV)"
            set_labels(subplot, xlabel, ylabel)
            for id in id_list:
                to_be_plot = self.analog_signals[id].signal
                if v_thresh is not None:
                    to_be_plot = pylab.where(to_be_plot>=v_thresh-0.02, v_thresh+0.5, to_be_plot)
                if len(time_axis) > len(to_be_plot):
                    time_axis = time_axis[:-1]
                if len(to_be_plot) > len(time_axis):
                    to_be_plot = to_be_plot[:-1]
                subplot.plot(time_axis, to_be_plot, **kwargs)
                subplot.hold(1)
            #pylab.draw()


class CurrentList(AnalogSignalList):

    def plot(self, id_list=None, v_thresh=None, display=True, kwargs={}):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, display=z, kwargs={'color':'r'})
        """
        subplot   = get_display(display)
        id_list   = self._AnalogSignalList__sub_id_list(id_list)
        time_axis = self.time_axis()  
        if not subplot or not HAVE_PYLAB:
            print PYLAB_ERROR
        else:
            xlabel = "Time (ms)"
            ylabel = "Current (nA)"
            set_labels(subplot, xlabel, ylabel)
            for id in id_list:
                subplot.plot(time_axis, self.analog_signals[id].signal, **kwargs)
                subplot.hold(1)
            pylab.draw()

class ConductanceList(AnalogSignalList):

    def plot(self, id_list=None, v_thresh=None, display=True, kwargs={}):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, display=z, kwargs={'color':'r'})
        """
        subplot   = get_display(display)
        id_list   = self._AnalogSignalList__sub_id_list(id_list)
        time_axis = self.time_axis()  
        if not subplot or not HAVE_PYLAB:
            print PYLAB_ERROR
        else:
            xlabel = "Time (ms)"
            ylabel = "Conductance (nS)"
            set_labels(subplot, xlabel, ylabel)
            for id in id_list:
                subplot.plot(time_axis, self.analog_signals[id].signal, **kwargs)
                subplot.hold(1)
            pylab.draw()




def load_conductancelist(user_file, id_list=None, dt=None, t_start=None, t_stop=None, dims=None):
    """
    Returns TWO ConductanceList objects from a file. One for the excitatory and the other for
    the inhibitory conductance.
    If the file has been generated by PyNN, 
    a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, if a string
                    is provided, a StandardTextFile object is created
        id_list  - the list of the recorded ids. Can be an int (meaning cells in 
                   the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from either 
    the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    
    Examples:
        >> gexc, ginh = load_conductancelist("mydata.dat")
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="conductance", id_list=id_list, dt=dt, t_start=t_start, t_stop=t_stop, dims=dims)


def load_vmlist(user_file, id_list=None, dt=None, t_start=0, t_stop=None, dims=None):
    """
    Returns a VmList object from a file. If the file has been generated by PyNN, 
    a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, if a string
                    is provided, a StandardTextFile object is created
        id_list  - the list of the recorded ids. Can be an int (meaning cells in 
                   the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from either 
    the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="vm", id_list=id_list, dt=dt, t_start=t_start, t_stop=t_stop, dims=dims)


def load_currentlist(user_file, id_list=None, dt=None, t_start=None, t_stop=None, dims=None):
    """
    Returns a CurrentList object from a file. If the file has been generated by PyNN, 
    a header should be found with following parameters:
     ---> dims, dt, id of the first cell, id of the last cell. 
    They must be specified otherwise.  Then the classical PyNN format for text file is:
     ---> one line per event:  data value, GID
    
    Inputs:
        user_file - the user_file object with read/write methods. By defaults, if a string
                    is provided, a StandardTextFile object is created
        id_list  - the list of the recorded ids. Can be an int (meaning cells in 
                   the range (0,..,N)), or a list. 
        dims     - if the cells were aranged on a 2/3D grid, a tuple with the dimensions
        dt       - the discretization step, in ms
        t_start  - begining of the simulation, in ms.
        t_stop   - end of the simulation, in ms

    If dims, dt, t_start, t_stop or id_list are None, they will be infered from either 
    the data or from the header. All times are in milliseconds. 
    The format of the file (text, pickle or hdf5) will be inferred automatically
    """
    analog_loader = DataHandler(user_file)
    return analog_loader.load_analogs(type="current", id_list=id_list, dt=dt, t_start=t_start, t_stop=t_stop, dims=dims)


