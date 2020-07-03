Overview
--------
This module provides function to estimate causal influences of signals on each other.

Granger causality
~~~~~~~~~~~~~~~
Granger causality is a method to determine causal influence of one signal on another based on autoregressive modelling. It was developed by Nobel prize laureate Clive Granger and has been adopted in various numerical fields ever since [1]. In its simplest form, the method tests whether the past values of one signal help to reduce the prediction error of another signal, compared to the past values of the latter signal alone. If it does reduce the prediction error, the first signal is said to Granger cause the other signal.


Limitations
"""""""""""
The user must be mindful of the method's limitations, which are assumptions of covariance stationary data, linearity imposed by the underlying autoregressive modelling as well as the fact that the variables not included in the model will not be accounted for [2].

Implementation
""""""""""""""
The mathematical implementation of Granger causality methods in this module closely follows [3].

References
""""""""""
[1] Granger CWJ (1969) Investigating causal relations by econometric models and cross-spectral methods. Econometrica 37:424-438.

[2] Seth A (2007) Granger causality. Scholarpedia, 2(7):1667., revision #127333

[3] Ding M, Chen Y, Bressler SL (2006) Granger causality: Basic theory and application to neuroscience. In Schelter S, Winterhalder N, Timmer J. Handbook of Time Series Analysis. Wiley, Wienheim.




