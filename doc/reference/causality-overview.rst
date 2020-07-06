Overview
--------
This module provides function to estimate causal influences of signals on each other.

Granger causality
~~~~~~~~~~~~~~~~~
Granger causality is a method to determine causal influence of one signal on another based on autoregressive modelling. It was developed by Nobel prize laureate Clive Granger and has been adopted in various numerical fields ever since :cite:`granger-granger1969investigating`. In its simplest form, the method tests whether the past values of one signal help to reduce the prediction error of another signal, compared to the past values of the latter signal alone. If it does reduce the prediction error, the first signal is said to Granger cause the other signal.

Limitations
"""""""""""
The user must be mindful of the method's limitations, which are assumptions of covariance stationary data, linearity imposed by the underlying autoregressive modelling as well as the fact that the variables not included in the model will not be accounted for :cite:`granger-seth2007granger`.

Implementation
""""""""""""""
The mathematical implementation of Granger causality methods in this module closely follows :cite:`granger-ding2006granger`.

References
""""""""""

.. bibliography:: ../bib/elephant.bib
   :labelprefix: gr
   :keyprefix: granger-
   :style: unsrt




