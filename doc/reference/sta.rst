=======================
Spike-triggered average
=======================

.. testsetup::

   import numpy as np
   import neo
   from quantities import ms
   from elephant.sta import spike_triggered_average

   signal1 = np.arange(1000.0)
   signal2 = np.arange(1, 1001.0)
   spiketrain1 = neo.SpikeTrain([10.12, 20.23, 30.45], units=ms, t_stop=50*ms)
   spiketrain2 = neo.SpikeTrain([10.34, 20.56, 30.67], units=ms, t_stop=50*ms)

.. automodule:: elephant.sta
   :members:
