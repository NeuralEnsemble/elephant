*************
Release Notes
*************

Elephant 0.4.3 release notes
===========================
March 2nd 2018

Other changes
=============
* Bug fixes in `spade` module:
  * Fixed an incompatibility with the latest version of an external library

Elephant 0.4.2 release notes
===========================
March 1st 2018

New functions
=============
* `spike_train_generation` module:
  * **inhomogeneous_poisson()** function
* Modules for Spatio Temporal Pattern Detection (SPADE) `spade_src`:
  * Module SPADE: `spade.py`
* Module `statistics.py`:
  * Added CV2 (coefficient of variation for non-stationary time series)
* Module `spike_train_correlation.py`:
  * Added normalization in **cross-correlation histogram()** (CCH)

Other changes
=============
* Adapted the `setup.py` to automatically install the spade modules including the compiled `C` files `fim.so`
* Included testing environment for MPI in `travis.yml`
* Changed function arguments  in `current_source_density.py` to `neo.AnalogSignal` instead list of `neo.AnalogSignal` objects
* Fixes to travis and setup configuration files
* Fixed bug in ISI function `isi()`, `statistics.py` module
* Fixed bug in `dither_spikes()`, `spike_train_surrogates.py`
* Minor bug fixes
 
Elephant 0.4.1 release notes
============================
March 23rd 2017

Other changes
=============
* Fix in `setup.py` to correctly import the current source density module

Elephant 0.4.0 release notes
============================
March 22nd 2017

New functions
=============
* `spike_train_generation` module:
    * peak detection: **peak_detection()**
* Modules for Current Source Density: `current_source_density_src`
    * Module Current Source Density: `KCSD.py`
    * Module for Inverse Current Source Density: `icsd.py`

API changes
===========
* Interoperability between Neo 0.5.0 and Elephant
    * Elephant has adapted its functions to the changes in Neo 0.5.0,
      most of the functionality behaves as before
    * See Neo documentation for recent changes: http://neo.readthedocs.io/en/latest/whatisnew.html

Other changes
=============
* Fixes to travis and setup configuration files.
* Minor bug fixes.
* Added module `six` for Python 2.7 backwards compatibility


Elephant 0.3.0 release notes
============================
April 12st 2016

New functions
=============
* `spike_train_correlation` module:
    * cross correlation histogram: **cross_correlation_histogram()**
* `spike_train_generation` module:
    * single interaction process (SIP): **single_interaction_process()**
    * compound Poisson process (CPP): **compound_poisson_process()**
* `signal_processing` module:
    * analytic signal: **hilbert()**
* `sta` module:
    * spike field coherence: **spike_field_coherence()**
* Module to represent kernels: `kernels` module
* Spike train metrics / dissimilarity / synchrony measures: `spike_train_dissimilarity` module
* Unitary Event (UE) analysis: `unitary_event_analysis` module
* Analysis of Sequences of Synchronous EvenTs (ASSET): `asset` module

API changes
===========
* Function **instantaneous_rate()** now uses kernels as objects defined in the `kernels` module. The previous implementation of the function using the `make_kernel()` function is deprecated, but still temporarily available as `oldfct_instantaneous_rate()`.

Other changes
=============
* Fixes to travis and readthedocs configuration files.


Elephant 0.2.1 release notes
============================
February 18th 2016

Minor bug fixes.


Elephant 0.2.0 release notes
============================
September 22nd 2015

New functions
=============

* Added covariance function **covariance()** in the `spike_train_correlation` module
* Added complexity pdf **complexity_pdf()** in the `statistics` module
* Added spike train extraction from analog signals via threshold detection the in `spike_train_generation` module
* Added **coherence()** function for analog signals in the `spectral` module
* Added **Cumulant Based Inference for higher-order of Correlation (CuBIC)** in the `cubic` module for correlation analysis of parallel recorded spike trains

API changes
===========
* **Optimized kernel bandwidth** in `rate_estimation` function: Calculates the optimized kernel width when the paramter kernel width is specified as `auto`

Other changes
=============
* **Optimized creation of sparse matrices**: The creation speed of the sparse matrix inside the `BinnedSpikeTrain` class is optimized
* Added **Izhikevich neuron simulator** in the `make_spike_extraction_test_data` module
* Minor improvements to the test and continous integration infrastructure
