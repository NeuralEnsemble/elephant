*********
Tutorials
*********

These tutorials provide narrative explanations, sample code, and expected
output for some of the neurophysiological analyses in Elephant. You can browse
the tutorials, launch them in mybinder or try them on ebrains to change and
interact with the code.

Launching a notebook on EBRAINS and Binder both provide a convenient way to
run and interact with code. The main difference between the two is that
changes made to a notebook launched on Binder are not saved, while
changes made to a notebook launched on EBRAINS are persistent and bound to the
user's EBRAINS account. This makes EBRAINS a great choice if you want to save
your work and come back to it later.

Introductory
------------

* Statistics of spike trains.

  Covers ``statistics`` module with an introduction to
  `Neo <https://neo.readthedocs.io/en/stable/>`_ input-output data types like
  `SpikeTrain`, `AnalogSignal`, etc.

  :doc:`View the notebook <../tutorials/statistics>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/statistics.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fstatistics.ipynb&branch=master

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
  ..
      .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2F+unitary_event_analysis.ipynb&branch=master

* Gaussian Process Factor Analysis (GPFA).

  GPFA is a dimensionality reduction method for neural trajectory visualization
  of parallel spike trains.

  :doc:`View the notebook <../tutorials/gpfa>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/gpfa.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fgpfa.ipynb&branch=master

* Spike Pattern Detection and Evaluation (SPADE)

  :doc:`View the notebook <../tutorials/spade>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/spade.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fspade.ipynb&branch=master

* Analysis of Sequences of Synchronous EvenTs (ASSET)

  :doc:`View the notebook <../tutorials/asset>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/asset.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fasset.ipynb&branch=master

* Granger causality

  :doc:`View the notebook <../tutorials/granger_causality>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/granger_causality.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fgranger_causality.ipynb+&branch=master


Additional
----------

* Parallel

  ``elephant.parallel`` module provides a simple interface to parallelize
  multiple calls to any user-specified function.

  :doc:`View the notebook <../tutorials/parallel>` or run interactively:

  .. image:: https://mybinder.org/badge.svg
     :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/parallel.ipynb
  .. image:: https://img.shields.io/badge/launch-ebrains-brightgreen
     :target: https://lab.ch.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNeuralEnsemble%2Felephant&urlpath=lab%2Ftree%2Felephant%2Fdoc%2Ftutorials%2Fparallel.ipynb+&branch=master

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
    tutorials/spade.ipynb
