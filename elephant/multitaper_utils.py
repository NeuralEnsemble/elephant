"""
Copyright (c) 2006-2011, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

	* Redistributions of source code must retain the above copyright
	   notice, this list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above
	   copyright notice, this list of conditions and the following
	   disclaimer in the documentation and/or other materials provided
	   with the distribution.

	* Neither the name of the NIPY Developers nor the names of any
	   contributors may be used to endorse or promote products derived
	   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Nitime 0.6 code slightly adapted by D. Mingers, Aug 2016
d.mingers@web.de

CONTENT OF THIS FILE:

Miscellaneous utilities for time series analysis.

"""
from __future__ import print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal.signaltools as signaltools


# import scipy.linalg as linalg
# import scipy.signal as sig

def jackknifed_sdf_variance(yk, eigvals, sides='onesided', adaptive=True):
	r"""
	Returns the variance of the log-sdf estimated through jack-knifing
	a group of independent sdf estimates.

	Parameters
	----------

	yk : ndarray (K, L)
	   The K DFTs of the tapered sequences
	eigvals : ndarray (K,)
	   The eigenvalues corresponding to the K DPSS tapers
	sides : str, optional
	   Compute the jackknife pseudovalues over as one-sided or
	   two-sided spectra
	adpative : bool, optional
	   Compute the adaptive weighting for each jackknife pseudovalue

	Returns
	-------

	var : The estimate for log-sdf variance

	Notes
	-----

	The jackknifed mean estimate is distributed about the true mean as
	a Student's t-distribution with (K-1) degrees of freedom, and
	standard error equal to sqrt(var). However, Thompson and Chave [1]
	point out that this variance better describes the sample mean.


	[1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
	Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
	"""
	K = yk.shape[ 0 ]

	from multitaper_spectral import mtm_cross_spectrum

	# the samples {S_k} are defined, with or without weights, as
	# S_k = | x_k |**2
	# | x_k |**2 = | y_k * d_k |**2          (with adaptive weights)
	# | x_k |**2 = | y_k * sqrt(eig_k) |**2  (without adaptive weights)

	all_orders = set(range(K))
	jk_sdf = [ ]
	# get the leave-one-out estimates -- ideally, weights are recomputed
	# for each leave-one-out. This is now the case.
	for i in range(K):
		items = list(all_orders.difference([ i ]))
		spectra_i = np.take(yk, items, axis=0)
		eigs_i = np.take(eigvals, items)
		if adaptive:
			# compute the weights
			weights, _ = _adaptive_weights(spectra_i, eigs_i, sides=sides)
		else:
			weights = eigs_i[ :, None ]
		# this is the leave-one-out estimate of the sdf
		jk_sdf.append(
				mtm_cross_spectrum(
						spectra_i, spectra_i, weights, sides=sides
				)
		)
	# log-transform the leave-one-out estimates and the mean of estimates
	jk_sdf = np.log(jk_sdf)
	# jk_avg should be the mean of the log(jk_sdf(i))
	jk_avg = jk_sdf.mean(axis=0)

	K = float(K)

	jk_var = (jk_sdf - jk_avg)
	np.power(jk_var, 2, jk_var)
	jk_var = jk_var.sum(axis=0)

	# Thompson's recommended factor, eq 18
	# Jackknifing Multitaper Spectrum Estimates
	# IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
	f = (K - 1)**2 / K / (K - 0.5)
	jk_var *= f
	return jk_var


# -----------------------------------------------------------------------------
# Multitaper utils
# -----------------------------------------------------------------------------
def _adaptive_weights(yk, eigvals, sides='onesided', max_iter=150):
	r"""
	Perform an iterative procedure to find the optimal weights for K
	direct spectral estimators of DPSS tapered signals.

	Parameters
	----------

	yk : ndarray (K, N)
	   The K DFTs of the tapered sequences
	eigvals : ndarray, length-K
	   The eigenvalues of the DPSS tapers
	sides : str
	   Whether to compute weights on a one-sided or two-sided spectrum
	max_iter : int
	   Maximum number of iterations for weight computation

	Returns
	-------

	weights, nu

	   The weights (array like sdfs), and the
	   "equivalent degrees of freedom" (array length-L)

	Notes
	-----

	The weights to use for making the multitaper estimate, such that
	:math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

	If there are less than 3 tapers, then the adaptive weights are not
	found. The square root of the eigenvalues are returned as weights,
	and the degrees of freedom are 2*K

	"""
	from multitaper_spectral import mtm_cross_spectrum
	K = len(eigvals)
	if sides not in [ 'one_sided', 'two_sided' ]:
		warnings.warn('Warning: strange input: sides', UserWarning)
	if max_iter <= 0:
		warnings.warn('Warning: strange input: iterations', UserWarning)
	if K < 3:
		warnings.warn('Warning--not adaptively combining the spectral '
					  'estimators due to a low number of tapers.', UserWarning)
		# we'll hope this is a correct length for L
		N = yk.shape[ -1 ]
		L = N / 2 + 1 if sides == 'onesided' else N
		return (np.multiply.outer(np.sqrt(eigvals), np.ones(L)), 2 * K)
	rt_eig = np.sqrt(eigvals)

	# combine the SDFs in the traditional way in order to estimate
	# the variance of the timeseries
	N = yk.shape[ 1 ]
	sdf = mtm_cross_spectrum(yk, yk, eigvals[ :, None ], sides=sides)
	L = sdf.shape[ -1 ]
	var_est = np.sum(sdf, axis=-1) / N
	bband_sup = (1 - eigvals) * var_est

	# The process is to iteratively switch solving for the following
	# two expressions:
	# (1) Adaptive Multitaper SDF:
	# S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
	#
	# (2) Weights
	# d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
	#
	# Where lam_k are the eigenvalues corresponding to the DPSS tapers,
	# and the expected value of the broadband bias function
	# E{B_k(f)} is replaced by its full-band integration
	# (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

	# start with an estimate from incomplete data--the first 2 tapers
	sdf_iter = mtm_cross_spectrum(yk[ :2 ], yk[ :2 ], eigvals[ :2, None ],
								  sides=sides)
	err = np.zeros((K, L))
	# for numerical considerations, don't bother doing adaptive
	# weighting after 150 dB down
	min_pwr = sdf_iter.max() * 10**(-150 / 20.)
	default_weights = np.where(sdf_iter < min_pwr)[ 0 ]
	adaptiv_weights = np.where(sdf_iter >= min_pwr)[ 0 ]

	w_def = rt_eig[ :, None ] * sdf_iter[ default_weights ]
	w_def /= eigvals[ :, None ] * sdf_iter[ default_weights ] + bband_sup[ :,
																None ]

	d_sdfs = np.abs(yk[ :, adaptiv_weights ])**2
	if L < N:
		d_sdfs *= 2
	sdf_iter = sdf_iter[ adaptiv_weights ]
	yk = yk[ :, adaptiv_weights ]
	for n in range(max_iter):
		d_k = rt_eig[ :, None ] * sdf_iter[ None, : ]
		d_k /= eigvals[ :, None ] * sdf_iter[ None, : ] + bband_sup[ :, None ]
		# Test for convergence -- this is overly conservative, since
		# iteration only stops when all frequencies have converged.
		# A better approach is to iterate separately for each freq, but
		# that is a nonvectorized algorithm.
		# sdf_iter = mtm_cross_spectrum(yk, yk, d_k, sides=sides)
		sdf_iter = np.sum(d_k**2 * d_sdfs, axis=0)
		sdf_iter /= np.sum(d_k**2, axis=0)
		# Compute the cost function from eq 5.4 in Thomson 1982
		cfn = eigvals[ :, None ] * (sdf_iter[ None, : ] - d_sdfs)
		cfn /= (eigvals[ :, None ] * sdf_iter[ None, : ] + bband_sup[ :,
														   None ])**2
		cfn = np.sum(cfn, axis=0)
		# there seem to be some pathological freqs sometimes ..
		# this should be a good heuristic
		if np.percentile(cfn**2, 95) < 1e-12:
			break
	else:  # If you have reached maximum number of iterations
		# Issue a warning and return non-converged weights:
		e_s = 'Breaking due to iterative meltdown in '
		e_s += 'multitaper_utils._adaptive_weights.'
		warnings.warn(e_s, RuntimeWarning)
	weights = np.zeros((K, L))
	weights[ :, adaptiv_weights ] = d_k
	weights[ :, default_weights ] = w_def
	nu = 2 * (weights**2).sum(axis=-2)
	return weights, nu


# -----------------------------------------------------------------------------
# Eigensystem utils
# -----------------------------------------------------------------------------

# If we can get it, we want the cythonized version
try:
	from _utils import _tridisolve
	print('cython version of tridisolve imported')

# If that doesn't work, we define it here:
except ImportError:
	def _tridisolve(d, e, b, overwrite_b=True):
		"""
		Symmetric tridiagonal system solver,
		from Golub and Van Loan, Matrix Computations pg 157

		Parameters
		----------

		d : ndarray
		  main diagonal stored in d[:]
		e : ndarray
		  superdiagonal stored in e[:-1]
		b : ndarray
		  RHS vector

		Returns
		-------

		x : ndarray
		  Solution to Ax = b (if overwrite_b is False). Otherwise solution is
		  stored in previous RHS vector b

		"""
		N = len(b)
		# work vectors
		dw = d.copy()
		ew = e.copy()
		if overwrite_b:
			x = b
		else:
			x = b.copy()
		for k in range(1, N):
			# e^(k-1) = e(k-1) / d(k-1)
			# d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
			t = ew[ k - 1 ]
			ew[ k - 1 ] = t / dw[ k - 1 ]
			dw[ k ] = dw[ k ] - t * ew[ k - 1 ]
		for k in range(1, N):
			x[ k ] = x[ k ] - ew[ k - 1 ] * x[ k - 1 ]
		x[ N - 1 ] = x[ N - 1 ] / dw[ N - 1 ]
		for k in range(N - 2, -1, -1):
			x[ k ] = x[ k ] / dw[ k ] - ew[ k ] * x[ k + 1 ]

		if not overwrite_b:
			return x


def _tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-6):
	"""Perform an inverse iteration to find the eigenvector corresponding
	to the given eigenvalue in a symmetric tridiagonal system.

	Parameters
	----------

	d : ndarray
	  main diagonal of the tridiagonal system
	e : ndarray
	  offdiagonal stored in e[:-1]
	w : float
	  eigenvalue of the eigenvector
	x0 : ndarray
	  initial point to start the iteration
	rtol : float
	  tolerance for the norm of the difference of iterates

	Returns
	-------

	e : ndarray
	  The converged eigenvector

	"""
	eig_diag = d - w
	if x0 is None:
		x0 = np.random.randn(len(d))
	x_prev = np.zeros_like(x0)
	norm_x = np.linalg.norm(x0)
	# the eigenvector is unique up to sign change, so iterate
	# until || |x^(n)| - |x^(n-1)| ||^2 < rtol
	x0 /= norm_x
	while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
		x_prev = x0.copy()
		_tridisolve(eig_diag, e, x0)
		norm_x = np.linalg.norm(x0)
		x0 /= norm_x
	return x0


# # ---------------------------------------------------------------------------
# # Correlation/Covariance utils
# # ---------------------------------------------------------------------------


def _circle_to_hz(omega, Fsamp):
	"""For a frequency grid spaced on the unit circle of an imaginary plane,
	return the corresponding freqency grid in Hz.
	"""
	return Fsamp * omega / (2 * np.pi)


def _remove_bias(x, axis):
	"Subtracts an estimate of the mean from signal x at axis"
	padded_slice = [ slice(d) for d in x.shape ]
	padded_slice[ axis ] = np.newaxis
	mn = np.mean(x, axis=axis)
	return x - mn[ tuple(padded_slice) ]


def crosscov(x, y, axis=-1, all_lags=False, debias=True, normalize=True,
			 corr=False):
	"""Returns the crosscovariance sequence between two ndarrays.
	This is performed by calling fftconvolve on x, y[::-1]

	Parameters
	----------

	x : ndarray
	y : ndarray
	axis : time axis
	all_lags : {True/False}
	   whether to return all nonzero lags, or to clip the length of s_xy
	   to be the length of x and y. If False, then the zero lag covariance
	   is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
	debias : {True/False}
	   Always removes an estimate of the mean along the axis, unless
	   told not to (eg X and Y are known zero-mean)
	corr : {True/False}
		Compute the crosscovariance via the correlation of the input ndarrays.
		Only possible if removal of bias allowed.

	Returns
	-------

	cxy : ndarray
	   The crosscovariance function

	Notes
	-----

	cross covariance of processes x and y is defined as

	.. math::

	C_{xy}[k]=E\{(X(n+k)-E\{X\})(Y(n)-E\{Y\})^{*}\}

	where X and Y are discrete, stationary (or ergodic) random processes

	Also note that this routine is the workhorse for all auto/cross/cov/corr
	functions.

	"""
	if x.shape[ axis ] != y.shape[ axis ]:
		raise ValueError(
				'crosscov() only works on same-length sequences for now'
		)
	if debias:
		x = _remove_bias(x, axis)
		y = _remove_bias(y, axis)
	slicing = [ slice(d) for d in x.shape ]
	slicing[ axis ] = slice(None, None, -1)
	cxy = fftconvolve(x, y[ tuple(slicing) ].conj(), axis=axis, mode='full')
	N = x.shape[ axis ]
	if normalize:
		cxy /= N
	if all_lags:
		return cxy
	slicing[ axis ] = slice(N - 1, 2 * N - 1)
	return cxy[ tuple(slicing) ]


def autocov(x, **kwargs):
	"""Returns the autocovariance of signal s at all lags.

	Parameters
	----------

	x : ndarray
	axis : time axis
	all_lags : {True/False}
	   whether to return all nonzero lags, or to clip the length of r_xy
	   to be the length of x and y. If False, then the zero lag correlation
	   is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

	Returns
	-------

	cxx : ndarray
	   The autocovariance function

	Notes
	-----

	Adheres to the definition

	.. math::

	C_{xx}[k]=E\{(X[n+k]-E\{X\})(X[n]-E\{X\})^{*}\}

	where X is a discrete, stationary (ergodic) random process
	"""
	# only remove the mean once, if needed
	debias = kwargs.pop('debias', True)
	axis = kwargs.get('axis', -1)
	if debias:
		x = _remove_bias(x, axis)
	kwargs[ 'debias' ] = False
	return crosscov(x, x, **kwargs)


def autocorr(x, **kwargs):
	"""Returns the autocorrelation of signal s at all lags.

	Parameters
	----------

	x : ndarray
	axis : time axis
	all_lags : {True/False}
	   whether to return all nonzero lags, or to clip the length of r_xy
	   to be the length of x and y. If False, then the zero lag correlation
	   is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

	Notes
	-----

	Adheres to the definition

	.. math::

	R_{xx}[k]=E\{X[n+k]X^{*}[n]\}

	where X is a discrete, stationary (ergodic) random process

	"""
	# do same computation as autocovariance,
	# but without subtracting the mean
	kwargs[ 'debias' ] = False
	return autocov(x, **kwargs)


def fftconvolve(in1, in2, mode="full", axis=None):
	""" Convolve two N-dimensional arrays using FFT. See convolve.

	This is a fix of scipy.signal.fftconvolve, adding an axis argument and
	importing locally the stuff only needed for this function

	"""
	s1 = np.array(in1.shape)
	s2 = np.array(in2.shape)
	complex_result = (np.issubdtype(in1.dtype, np.complex) or
					  np.issubdtype(in2.dtype, np.complex))

	if axis is None:
		size = s1 + s2 - 1
		fslice = tuple([ slice(0, int(sz)) for sz in size ])
	else:
		equal_shapes = s1 == s2
		# allow equal_shapes[axis] to be False
		equal_shapes[ axis ] = True
		assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
		size = s1[ axis ] + s2[ axis ] - 1
		fslice = [ slice(l) for l in s1 ]
		fslice[ axis ] = slice(0, int(size))
		fslice = tuple(fslice)

	# Always use 2**n-sized FFT
	fsize = 2**int(np.ceil(np.log2(size)))
	if axis is None:
		IN1 = fftpack.fftn(in1, fsize)
		IN1 *= fftpack.fftn(in2, fsize)
		ret = fftpack.ifftn(IN1)[ fslice ].copy()
	else:
		IN1 = fftpack.fft(in1, fsize, axis=axis)
		IN1 *= fftpack.fft(in2, fsize, axis=axis)
		ret = fftpack.ifft(IN1, axis=axis)[ fslice ].copy()
	del IN1
	if not complex_result:
		ret = ret.real
	if mode == "full":
		return ret
	elif mode == "same":
		if np.product(s1, axis=0) > np.product(s2, axis=0):
			osize = s1
		else:
			osize = s2
		return signaltools._centered(ret, osize)
	elif mode == "valid":
		return signaltools._centered(ret, abs(s2 - s1) + 1)
