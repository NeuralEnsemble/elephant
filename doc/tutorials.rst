*********
Tutorials
*********

These tutorials provide narrative explanations, sample code, and expected
output for some of the neurophysiological analyses in Elephant. You can browse
the tutorials or launch them in mybinder to change and interact with the code.


Introductory
------------

* Statistics of spike trains.

  Covers ``statistics`` module with an introduction to
  `Neo <https://neo.readthedocs.io/en/stable/>`_ input-output data types like
  `SpikeTrain`, `AnalogSignal`, etc.

  :doc:`View the notebook <../tutorials/statistics>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/statistics.ipynb


Advanced
--------

* Unitary Event Analysis.

  The analysis detects coordinated spiking activity that occurs significantly
  more often than predicted by the firing rates of neurons alone. It's superior
  to simple statistics.

  :doc:`View the notebook <../tutorials/unitary_event_analysis>` or run
  interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/unitary_event_analysis.ipynb

* Gaussian Process Factor Analysis (GPFA).

  GPFA is a dimensionality reduction method for neural trajectory visualization
  of parallel spike trains.

  :doc:`View the notebook <../tutorials/gpfa>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/gpfa.ipynb

* Analysis of Sequences of Synchronous EvenTs (ASSET)

  :doc:`View the notebook <../tutorials/asset>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/asset.ipynb

* Granger causality

  :doc:`View the notebook <../tutorials/granger_causality>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/granger_causality.ipynb


Additional
----------

* Parallel

  ``elephant.parallel`` module provides a simple interface to parallelize
  multiple calls to any user-specified function.

  :doc:`View the notebook <../tutorials/parallel>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/parallel.ipynb

..
    Index the notebooks in a hidden toctree to avoid sphinx warnings.

.. toctree::
    :hidden:

    tutorials/asset.ipynb
    tutorials/gpfa.ipynb
    tutorials/parallel.ipynb
    tutorials/statistics.ipynb
    tutorials/unitary_event_analysis.ipynb
    tutorials/granger_causality.ipynb
