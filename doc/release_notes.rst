*************
Release Notes
*************

Elephant 0.2.0 release notes
==========================
September 22nd 2015

Welcome to the 0.2.0 release of Elephant!

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
