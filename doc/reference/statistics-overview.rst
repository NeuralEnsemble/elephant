Overview
--------

.. toctree::
    :hidden:
    
    tutorials/statistical.ipynb
    
    
Rate estimation
~~~~~~~~~~~~~~~

One of the most prominent features of one or several spike trains is the frequency at which spikes occur. This is what is commonly referred to as the rate of the spike train. The function `mean_firing_rate()` is the most simple function to estimate the firing rate of a spike train in a given interval based on the spike count.  

Despite the simplicity of this measure, estimating the rate of a spike train is not straight-forward due to its nature as a point process in particular in the case where the rate is potentially non-stationary (i.e., it changes over time) and we which to estimate the instantaenous firing rate at a given time point :math:`t`.

Often, neuroscientists will use multiple trials to make an estimate of the time-resolved rate. Here, the time axis is typically binned and spike counts are collected per bin, across trials. This type of analysis can be performed using the `time_histogram` function.

.. plot

   import matplotlib.pyplot as plt
   import quantities as pq
   spike_trains = [elephant.spike_train_generation.homogenous_poisson_process(rate=5 * pq.s) for _ in range(100) ]
   histogram = elephant.statistics.time_histogram(spike_trains)
   plt.plot(histogram.time,histogram)


.. TODO Spike interval statistics
.. TODO Statistics across spike trains


Tutorial
--------

:doc:`View tutorial <../tutorials/statistics>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/INM-6/elephant/enh/module_doc?filepath=doc/tutorials/statistics.ipynb

