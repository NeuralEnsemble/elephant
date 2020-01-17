Synopsis
~~~~~~~~

Unitary Event (UE) analysis is a statistical method that
analyzes excess spike correlation between simultaneously recorded neurons in a
time resolved manner by comparing the empirical spike coincidences to the
expected number based on the firing rates of the neurons (see :cite:`unitary-event-analysis-Gruen99_67`).

Background
~~~~~~~~~~

It has been proposed that cortical neurons organize dynamically into functional
groups (“cell assemblies”) by the temporal structure of their joint spiking
activity. The Unitary Events analysis method detects conspicuous patterns of
synchronous spike activity among simultaneously recorded single neurons. The
statistical significance of a pattern is evaluated by comparing the empirical
number of occurrences to the number expected given the firing rates of the
neurons. Key elements of the method are the proper formulation of the null
hypothesis and the derivation of the corresponding count distribution of
synchronous spike events used in the significance test. The analysis is
performed in a sliding window manner and yields a time-resolved measure of
significant spike synchrony. For further reading, see :cite:`unitary-event-analysis-Riehle97_1950,unitary-event-analysis-Gruen02_43,unitary-event-analysis-Gruen02_81,unitary-event-analysis-Gruen03,unitary-event-analysis-Gruen09_1126`


References
~~~~~~~~~~

.. bibliography:: ../bib/elephant.bib
   :labelprefix: UEA
   :keyprefix: unitary-event-analysis
   :style: unsrt


Tutorial
--------

:doc:`View tutorial <../tutorials/unitary_event_analysis>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/INM-6/elephant/enh/module_doc?filepath=doc/tutorials/unitary_event_analysis.ipynb


Author Contributions
--------------------

- Vahid Rostami (VH)
- Sonja Gruen (SG)
- Markus Diesmann (MD)
VH implemented the method, SG and MD provided guidance.
