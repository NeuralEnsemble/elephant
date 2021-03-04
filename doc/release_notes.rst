=============
Release Notes
=============

Elephant 0.10.0 release notes
=============================

Documentation
-------------
The documentation is revised and restructured by categories (https://github.com/NeuralEnsemble/elephant/pull/386) to simplify navigation on readthedocs and improve user experience. All citations used in Elephant are stored in a single [BibTex file](https://github.com/NeuralEnsemble/elephant/blob/master/doc/bib/elephant.bib).

Optimizations
-------------

CUDA and OpenCL support
***********************
[Analysis of Sequences of Synchronous EvenTs](https://elephant.readthedocs.io/en/latest/reference/asset.html) has become the first module in Elephant that supports CUDA and OpenCL (https://github.com/NeuralEnsemble/elephant/pull/351, https://github.com/NeuralEnsemble/elephant/pull/404, https://github.com/NeuralEnsemble/elephant/pull/399). Whether you have an Nvidia GPU or just run the analysis on a laptop with a built-in Intel graphics card, the speed-up is **X100** and **X1000** compared to a single CPU core. The computations are optimized to a degree that you can analyse and look for spike patterns in real data in several minutes of compute time on a laptop. The installation instructions are described in the [install](https://elephant.readthedocs.io/en/latest/install.html) section.

Other optimizations
*******************
* Surrogates: sped up bin shuffling (https://github.com/NeuralEnsemble/elephant/pull/400) and reimplemented the continuous time version (https://github.com/NeuralEnsemble/elephant/pull/397)
* Improved memory efficiency of creating a BinnedSpikeTrain (https://github.com/NeuralEnsemble/elephant/pull/395)

New functionality and features
------------------------------
* Synchrofact detection (https://github.com/NeuralEnsemble/elephant/pull/322) is a method to detect highly synchronous spikes (at the level of sampling rate precision with an option to extend this to jittered synchrony) and annotate or optionally remove them.
* Added `phase_locking_value`, `mean_phase_vector`, and `phase_difference` functions (https://github.com/NeuralEnsemble/elephant/pull/385/files)
* BinnedSpikeTrain:
  - added `to_spike_trains` and `time_slice` functions (https://github.com/NeuralEnsemble/elephant/pull/390). Now you can slice a binned spike train as `bst[:, i:j]` or `bst.time_slice(t_start, t_stop)`. Also, with `to_spike_trains` function, you can generate a realization of spike trains that maps to the same BinnedSpikeTrain object when binned.
  - optional CSC format (https://github.com/NeuralEnsemble/elephant/pull/402)
  - the `copy` parameter (False by default) in the `binarize` function makes a *shallow* copy, if set to True, of the output BinnedSpikeTrain object (https://github.com/NeuralEnsemble/elephant/pull/402)
* Granger causality tutorial notebook (https://github.com/NeuralEnsemble/elephant/pull/393)
* Unitary Event Analysis support multiple pattern hashes (https://github.com/NeuralEnsemble/elephant/pull/387)

Bug fixes
---------
* Account for unidirectional spiketrain->segment links in synchrofact deletion (https://github.com/NeuralEnsemble/elephant/pull/398)
* Joint-ISI dithering: fixed a bug regarding first ISI bin (https://github.com/NeuralEnsemble/elephant/pull/396)
* Fix LvR values from being off when units are in seconds (https://github.com/NeuralEnsemble/elephant/pull/389)


Elephant 0.9.0 release notes
****************************

This release is titled to accompany the [2nd Elephant User Workshop](https://www.humanbrainproject.eu/en/education/participatecollaborate/infrastructure-events-trainings/2nd-elephant-user-workshop/)

Viziphant
---------
Meet Viziphant, the visualization of Elephant analysis methods, at https://viziphant.readthedocs.io/en/latest/. This package provides support to easily plot and visualize the output of Elephant functions in a few lines of code.

Provenance tracking
-------------------
Provenance is becoming a separate direction in Elephant. Many things are still to come, and we started with annotating `time_histogram`, `instantaneous_rate` and `cross_correlation_histogram` outputs to carry the information about the parameters these functions used. This allowed Viziphant, the visualization of Elephant analyses, to look for the `.annotations` dictionary of the output of these function to "understand" how the object has been generated and label the plot axes accordingly.

New functionality and features
------------------------------
* Time-domain pairwise and conditional pairwise Granger causality measures (https://github.com/NeuralEnsemble/elephant/pull/332, https://github.com/NeuralEnsemble/elephant/pull/359)
* Spike contrast function that measures the synchrony of spike trains (https://github.com/NeuralEnsemble/elephant/pull/354; thanks to @Broxy7 for bringing this in Elephant).
* Revised local variability LvR (https://github.com/NeuralEnsemble/elephant/pull/346) as an alternative to the LV measure.
* Three surrogate methods: Trial-shifting, Bin Shuffling, ISI dithering (https://github.com/NeuralEnsemble/elephant/pull/343).
* Added a new function to generate spike trains: `inhomogeneous_gamma_process` (https://github.com/NeuralEnsemble/elephant/pull/339).
* The output of `instantaneous_rate` function is now a 2D matrix of shape `(time, len(spiketrains))` (https://github.com/NeuralEnsemble/elephant/issues/363). Not only can the users assess the averaged instantaneous rate (`rates.mean(axis=1)`) but also explore how much the instantaneous rate deviates from trial to trial (`rates.std(axis=1)`) (originally asked in https://github.com/NeuralEnsemble/elephant/issues/363).

Python 3 only
-------------
* Python 2.7 and 3.5 support is dropped. You can still however enjoy the features of Elephant v0.9.0 with Python 2.7 or 3.5 by installing Elephant from [this](https://github.com/NeuralEnsemble/elephant/tree/295c6bd7fea196cf9665a78649fafedab5840cfa) commit `pip install git+https://github.com/NeuralEnsemble/elephant@295c6bd7fea196cf9665a78649fafedab5840cfa#egg=elephant[extras]`
* Added Python 3.9 support.

Optimization
------------
* You have been asking for direct numpy support for years. Added `_t_start`, `_t_stop`, and `_bin_size` attributes of BinnedSpikeTrain are guaranteed to be of the same units and hence are unitless (https://github.com/NeuralEnsemble/elephant/pull/378). It doesn't mean though that you need to care about units on your own: `t_start`, `t_stop`, and `bin_size` properties are still quantities with units. The `.rescale()` method of a BinnedSpikeTrain rescales the internal units to new ones in-place. The following Elephant functions are optimized with unitless BinnedSpikeTrain:
  - cross_correlation_histogram
  - bin_shuffling (one of the surrogate methods)
  - spike_train_timescale
* X4 faster binning and overall BinnedSpikeTrain object creation (https://github.com/NeuralEnsemble/elephant/pull/368).
* `instantaneous_rate` function is vectorized to work with a list of spike train trials rather than computing them in a loop (previously, `for spiketrain in spiketrains; do compute instantaneous_rate(spiketrain); done`), which brought X25 speedup (https://github.com/NeuralEnsemble/elephant/pull/362; thanks to @gyyang for the idea and original implementation).
* Memory-efficient `zscore` function (https://github.com/NeuralEnsemble/elephant/pull/372).
* Don't sort the input array in ISI function (https://github.com/NeuralEnsemble/elephant/pull/371), which reduces function algorithmic time complexity from `O(N logN)` to linear `O(N)`. Now, when the input time array is not sorted, a warning is shown.
* Vectorized Current Source Density `generate_lfp` function (https://github.com/NeuralEnsemble/elephant/pull/358).

Breaking changes
----------------
* mpi4py package is removed from the extra requirements to allow `pip install elephant[extras]` on machines without MPI installed system-wide. Refer to [MPI support](https://elephant.readthedocs.io/en/latest/install.html#mpi-support) installation page in elephant.
* BinnedSpikeTrain (https://github.com/NeuralEnsemble/elephant/pull/368, https://github.com/NeuralEnsemble/elephant/pull/377):
  - previously, when t_start/stop, if set manually, was outside of the shared time interval, only the shared [t_start_shared=max(t_start), t_stop_shared=min(t_stop)] interval was implicitly considered without any warnings. Now an error is thrown with a description on how to fix it.
  - removed `lst_input`, `input_spiketrains`, `matrix_columns`, `matrix_rows` (in favor of the new attribute - `shape`), `tolerance`, `is_spiketrain`, `is_binned` attributes from BinnedSpikeTrain class. Part of them are confusing (e.g., `is_binned` was just the opposite of `is_spiketrain`, but one can erroneously think that it's data is clipped to 0 and 1), and part of them - `lst_input`, `input_spiketrains` input data - should not have been saved as attributes of an object in the first place because the input spike trains are not used after the sparse matrix is created.
  - now the users can directly access `.sparse_matrix` attribute of BinnedSpikeTrain to do efficient (yet unsafe in general) operations. For this reason, `to_sparse_array()` function, which does not make a copy, as one could think of, is deprecated.
* `instantaneous_rate` function (https://github.com/NeuralEnsemble/elephant/pull/362):
  - in case of multiple input spike trains, the output of the instantaneous rate function is (always) a 2D matrix of shape `(time, len(spiketrains))` instead of a pseudo 1D array (previous behavior) of shape `(time, 1)` that contained the instantaneous rate summed across input spike trains;
  - in case of multiple input spike trains, the user needs to manually provide the input kernel instead of `auto`, which is set by default, for the reason that it's currently not clear how to estimate the common kernel for a set of spike trains. If you have an idea how to do this, we`d appreciate if you let us know by [getting in touch with us](https://elephant.readthedocs.io/en/latest/get_in_touch.html).

Other changes
-------------
* `waveform_snr` function now directly takes a 2D or 3D waveforms matrix rather than a spike train (deprecated behavior).
* Added a warning in fanofactor function when the input spiketrains vary in their durations (https://github.com/NeuralEnsemble/elephant/pull/341).
* SPADE: New way to count patterns for multiple testing (https://github.com/NeuralEnsemble/elephant/pull/347)
* GPFA renamed 'xsm' -> 'latent_variable' and 'xorth' -> 'latent_variable_orth'

Bug fixes
---------
* Instantaneous rate arrays were not centered at the origin for spike trains that are symmetric at t=0 with `center_kernel=True` option (https://github.com/NeuralEnsemble/elephant/pull/362).
* The number of discarded spikes that fall into the last bin of a BinnedSpikeTrain object was incorrectly calculated (https://github.com/NeuralEnsemble/elephant/pull/368).
* Fixed index selection in `spike_triggered_phase` (https://github.com/NeuralEnsemble/elephant/pull/382)
* Fixed surrogates bugs:
  - `joint-ISI` and `shuffle ISI` output spike trains were not sorted in time (https://github.com/NeuralEnsemble/elephant/pull/364);
  - surrogates get arbitrary sampling_rate (https://github.com/NeuralEnsemble/elephant/pull/353), which relates to the provenance tracking issue;



Elephant 0.8.0 release notes
****************************

New features
------------
* The `parallel` module is a new experimental module (https://github.com/NeuralEnsemble/elephant/pull/307) to run python functions concurrently. Supports native (pythonic) ProcessPollExecutor and MPI. Not limited to Elephant functional.
* Added an optional `refractory_period` argument, set to None by default, to `dither_spikes` function (https://github.com/NeuralEnsemble/elephant/pull/297).
* Added `cdf` and `icdf` functions in Kernel class to correctly estimate the median index, needed for `instantaneous_rate` function in statistics.py (https://github.com/NeuralEnsemble/elephant/pull/313).
* Added an optional `center_kernel` argument, set to True by default (to behave as in Elephant <0.8.0 versions) to `instantaneous_rate` function in statistics.py (https://github.com/NeuralEnsemble/elephant/pull/313).

New tutorials
-------------
* Analysis of Sequences of Synchronous EvenTs (ASSET) tutorial: https://elephant.readthedocs.io/en/latest/tutorials/asset.html
* Parallel module tutorial: https://elephant.readthedocs.io/en/latest/tutorials/parallel.html

Optimization
------------
* Optimized ASSET runtime by a factor of 10 and more (https://github.com/NeuralEnsemble/elephant/pull/259, https://github.com/NeuralEnsemble/elephant/pull/333).

Python 2.7 and 3.5 deprecation
------------------------------
Python 2.7 and 3.5 are deprecated and will not be maintained by the end of 2020. Switch to Python 3.6+.

Breaking changes
----------------
* Naming convention changes (`binsize` -> `bin_size`, etc.) in almost all Elephant functions (https://github.com/NeuralEnsemble/elephant/pull/316).

Elephant 0.7.0 release notes
****************************

Breaking changes
----------------
* [gpfa] GPFA dimensionality reduction method is rewritten in easy-to-use scikit-learn class style format (https://github.com/NeuralEnsemble/elephant/pull/287):

.. code-block:: python

    gpfa = GPFA(bin_size=20*pq.ms, x_dim=8)
    results = gpfa.fit_transform(spiketrains, returned_data=['xorth', 'xsm'])

New tutorials
-------------
* GPFA dimensionality reduction method: https://elephant.readthedocs.io/en/latest/tutorials/gpfa.html
* Unitary Event Analysis of coordinated spiking activity: https://elephant.readthedocs.io/en/latest/tutorials/unitary_event_analysis.html
* (Introductory) statistics module: https://elephant.readthedocs.io/en/latest/tutorials/statistics.html

Deprecations
------------
* **Python 2.7 support will be dropped on Dec 31, 2020.** Please switch to Python 3.6, 3.7, or 3.8.
* [spike train generation] `homogeneous_poisson_process_with_refr_period()`, introduced in v0.6.4, is deprecated and will be deleted in v0.8.0. Use `homogeneous_poisson_process(refractory_period=...)` instead.
* [pandas bridge] pandas\_bridge module is deprecated and will be deleted in v0.8.0.

New features
------------
* New documentation style, guidelines, tutorials, and more (https://github.com/NeuralEnsemble/elephant/pull/294).
* Python 3.8 support (https://github.com/NeuralEnsemble/elephant/pull/282).
* [spike train generation] Added `refractory_period` flag in `homogeneous_poisson_process()` (https://github.com/NeuralEnsemble/elephant/pull/292) and `inhomogeneous_poisson_process()` (https://github.com/NeuralEnsemble/elephant/pull/295) functions. The default is `refractory_period=None`, meaning no refractoriness.
* [spike train correlation] `cross_correlation_histogram()` supports different t_start and t_stop of input spiketrains.
* [waveform features] `waveform_width()` function extracts the width (trough-to-peak TTP) of a waveform (https://github.com/NeuralEnsemble/elephant/pull/279).
* [signal processing] Added `scaleopt` flag in `pairwise_cross_correlation()` to mimic the behavior of Matlab's `xcorr()` function (https://github.com/NeuralEnsemble/elephant/pull/277). The default is `scaleopt=unbiased` to be consistent with the previous versions of Elephant.
* [spike train surrogates] Joint-ISI dithering method via `JointISI` class (https://github.com/NeuralEnsemble/elephant/pull/275).

Bug fixes
---------
* [spike train correlation] Fix CCH Border Correction (https://github.com/NeuralEnsemble/elephant/pull/298). Now, the border correction in `cross_correlation_histogram()` correctly reflects the number of bins used for the calculation at each lag. The correction factor is now unity at full overlap.
* [phase analysis] `spike_triggered_phase()` incorrect behavior when the spike train and the analog signal had different time units (https://github.com/NeuralEnsemble/elephant/pull/270).

Performance
-----------
* [spade] SPADE x7 speedup (https://github.com/NeuralEnsemble/elephant/pull/280, https://github.com/NeuralEnsemble/elephant/pull/285, https://github.com/NeuralEnsemble/elephant/pull/286). Moreover, SPADE is now able to handle all surrogate types that are available in Elephant, as well as more types of statistical corrections.
* [conversion] Fast & memory-efficient `covariance()` and Pearson `corrcoef()` (https://github.com/NeuralEnsemble/elephant/pull/274). Added flag `fast=True` by default in both functions.
* [conversion] Use fast fftconvolve instead of np.correlate in `cross_correlation_histogram()` (https://github.com/NeuralEnsemble/elephant/pull/273).


Elephant 0.6.4 release notes
****************************

This release has been made for the "1st Elephant User Workshop" (https://www.humanbrainproject.eu/en/education/participatecollaborate/infrastructure-events-trainings/1st-elephant-user-workshop-accelerate-structured-and-reproducibl).


Main features
-------------
* neo v0.8.0 compatible


New modules
-----------
* GPFA - Gaussian-process factor analysis - dimensionality reduction method for neural trajectory visualization (https://github.com/NeuralEnsemble/elephant/pull/233). _Note: the API could change in the future._


Bug fixes
---------
* [signal processing] Keep `array_annotations` in the output of signal processing functions (https://github.com/NeuralEnsemble/elephant/pull/258).
* [SPADE] Fixed the calculation of the duration of a pattern in the output (https://github.com/NeuralEnsemble/elephant/pull/254).
* [statistics] Fixed automatic kernel selection yields incorrect values (https://github.com/NeuralEnsemble/elephant/pull/246).


Improvements
------------
* Vectorized `spike_time_tiling_coefficient()` function - got rid of a double for-loop (https://github.com/NeuralEnsemble/elephant/pull/244)
* Reduced the number of warnings during the tests (https://github.com/NeuralEnsemble/elephant/pull/238).
* Removed unused debug code in `spade/fast_fca.py` (https://github.com/NeuralEnsemble/elephant/pull/249).
* Improved doc string of `covariance()` and `corrcoef()` (https://github.com/NeuralEnsemble/elephant/pull/260).



Elephant 0.6.3 release notes
****************************
July 22nd 2019

The release v0.6.3 is mostly about improving maintenance.

New functions
-------------
* `waveform_features` module
    * Waveform signal-to-noise ratio (https://github.com/NeuralEnsemble/elephant/pull/219).
* Added support for Butterworth `sosfiltfilt` - numerically stable (in particular, higher order) filtering (https://github.com/NeuralEnsemble/elephant/pull/234).

Bug fixes
---------
* Fixed neo version typo in requirements file (https://github.com/NeuralEnsemble/elephant/pull/218)
* Fixed broken docs (https://github.com/NeuralEnsemble/elephant/pull/230, https://github.com/NeuralEnsemble/elephant/pull/232)
* Fixed issue with 32-bit arch (https://github.com/NeuralEnsemble/elephant/pull/229)

Other changes
-------------
* Added issue templates (https://github.com/NeuralEnsemble/elephant/pull/226)
* Single VERSION file (https://github.com/NeuralEnsemble/elephant/pull/231)

Elephant 0.6.2 release notes
****************************
April 23rd 2019

New functions
-------------
* `signal_processing` module
    * New functions to calculate the area under a time series and the derivative of a time series.

Other changes
-------------
* Added support to initialize binned spike train representations with a matrix
* Multiple bug fixes


Elephant 0.6.1 release notes
****************************
April 1st 2019

New functions
-------------
* `signal_processing` module
    * New function to calculate the cross-correlation function for analog signals.
* `spade` module
    * Spatio-temporal spike pattern detection now includes the option to assess significance also based on time-lags of patterns, in addition to patterns size and frequency (referred to as 3D pattern spectrum).

Other changes
-------------
* This release fixes a number of compatibility issues in relation to API breaking changes in the Neo library.
* Fixed error in STTC calculation (spike time tiling coefficient)
* Minor bug fixes


Elephant 0.6.0 release notes
****************************
October 12th 2018

New functions
-------------
* `cell_assembly_detection` module
    * New function to detect higher-order correlation structures such as patterns in parallel spike trains based on Russo et al, 2017.
*  **wavelet_transform()** function in `signal_prosessing.py` module
    * Function for computing wavelet transform of a given time series based on Le van Quyen et al. (2001)

Other changes
-------------
* Switched to multiple `requirements.txt` files which are directly read into the `setup.py`
* `instantaneous_rate()` accepts now list of spiketrains
* Minor bug fixes


Elephant 0.5.0 release notes
****************************
April 4nd 2018

New functions
-------------
* `change_point_detection` module:
    * New function to detect changes in the firing rate
* `spike_train_correlation` module:
    * New function to calculate the spike time tiling coefficient
* `phase_analysis` module:
    * New function to extract spike-triggered phases of an AnalogSignal
* `unitary_event_analysis` module:
    * Added new unit test to the UE function to verify the method based on data of a recent [Re]Science publication

Other changes
-------------
* Minor bug fixes


Elephant 0.4.3 release notes
****************************
March 2nd 2018

Other changes
-------------
* Bug fixes in `spade` module:
    * Fixed an incompatibility with the latest version of an external library


Elephant 0.4.2 release notes
****************************
March 1st 2018

New functions
-------------
* `spike_train_generation` module:
    * **inhomogeneous_poisson()** function
* Modules for Spatio Temporal Pattern Detection (SPADE) `spade_src`:
    * Module SPADE: `spade.py`
* Module `statistics.py`:
    * Added CV2 (coefficient of variation for non-stationary time series)
* Module `spike_train_correlation.py`:
    * Added normalization in **cross-correlation histogram()** (CCH)

Other changes
-------------
* Adapted the `setup.py` to automatically install the spade modules including the compiled `C` files `fim.so`
* Included testing environment for MPI in `travis.yml`
* Changed function arguments  in `current_source_density.py` to `neo.AnalogSignal` instead list of `neo.AnalogSignal` objects
* Fixes to travis and setup configuration files
* Fixed bug in ISI function `isi()`, `statistics.py` module
* Fixed bug in `dither_spikes()`, `spike_train_surrogates.py`
* Minor bug fixes


Elephant 0.4.1 release notes
****************************
March 23rd 2017

Other changes
-------------
* Fix in `setup.py` to correctly import the current source density module


Elephant 0.4.0 release notes
****************************
March 22nd 2017

New functions
-------------
* `spike_train_generation` module:
    * peak detection: **peak_detection()**
* Modules for Current Source Density: `current_source_density_src`
    * Module Current Source Density: `KCSD.py`
    * Module for Inverse Current Source Density: `icsd.py`

API changes
-----------
* Interoperability between Neo 0.5.0 and Elephant
    * Elephant has adapted its functions to the changes in Neo 0.5.0,
      most of the functionality behaves as before
    * See Neo documentation for recent changes: http://neo.readthedocs.io/en/latest/whatisnew.html

Other changes
-------------
* Fixes to travis and setup configuration files.
* Minor bug fixes.
* Added module `six` for Python 2.7 backwards compatibility


Elephant 0.3.0 release notes
****************************
April 12st 2016

New functions
-------------
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
-----------
* Function **instantaneous_rate()** now uses kernels as objects defined in the `kernels` module. The previous implementation of the function using the `make_kernel()` function is deprecated, but still temporarily available as `oldfct_instantaneous_rate()`.

Other changes
-------------
* Fixes to travis and readthedocs configuration files.


Elephant 0.2.1 release notes
****************************
February 18th 2016

Other changes
-------------
Minor bug fixes.


Elephant 0.2.0 release notes
****************************
September 22nd 2015

New functions
-------------
* Added covariance function **covariance()** in the `spike_train_correlation` module
* Added complexity pdf **complexity_pdf()** in the `statistics` module
* Added spike train extraction from analog signals via threshold detection the in `spike_train_generation` module
* Added **coherence()** function for analog signals in the `spectral` module
* Added **Cumulant Based Inference for higher-order of Correlation (CuBIC)** in the `cubic` module for correlation analysis of parallel recorded spike trains

API changes
-----------
* **Optimized kernel bandwidth** in `rate_estimation` function: Calculates the optimized kernel width when the paramter kernel width is specified as `auto`

Other changes
-------------
* **Optimized creation of sparse matrices**: The creation speed of the sparse matrix inside the `BinnedSpikeTrain` class is optimized
* Added **Izhikevich neuron simulator** in the `make_spike_extraction_test_data` module
* Minor improvements to the test and continous integration infrastructure
