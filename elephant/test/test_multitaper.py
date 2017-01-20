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


Nitime 0.6 code adapted by D. Mingers, Aug 2016
d.mingers@web.de

CONTENT OF THIS FILE:

Tests multitaperes spectral time series analysis.

"""
import os
import unittest
import warnings
import numpy as np
from numpy.testing import assert_allclose
import numpy.testing.decorators as dec
import elephant
from elephant import multitaper_spectral as mts
# from elephant import multitaper_utils as mtu

# Define globally
test_dir_path = os.path.join(elephant.__path__[ 0 ], 'test')


# include the following for testing multi_taper_psc/csd:
def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)


def periodogram(s, Fs=2 * np.pi, Sk=None, N=None,
                sides='default', normalize=True):
    """Takes an N-point periodogram estimate of the PSD function. The
    number of points N, or a precomputed FFT Sk may be provided. By default,
    the PSD function returned is normalized so that the integral of the PSD
    is equal to the mean squared amplitude (mean energy) of s (see Notes).
    Parameters
    ----------
    s : ndarray
        Signal(s) for which to estimate the PSD, time dimension in the last
        axis
    Fs : float (optional)
       The sampling rate. Defaults to 2*pi
    Sk : ndarray (optional)
        Precomputed FFT of s
    N : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]
    sides : str (optional) [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return.
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided
    PSD normalize : boolean (optional, default=True) Normalizes the PSD
    Returns
    -------
    (f, psd) : tuple
       f: The central frequencies for the frequency bands
       PSD estimate for each row of s
    """
    import scipy.fftpack as fftpack

    if Sk is not None:
        N = Sk.shape[ -1 ]
    else:
        N = s.shape[ -1 ] if not N else N
        Sk = fftpack.fft(s, n=N)
    pshape = list(Sk.shape)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    if sides == 'onesided':
        # putative Nyquist freq
        Fn = N // 2 + 1
        # last duplicate freq
        Fl = (N + 1) // 2
        pshape[ -1 ] = Fn
        P = np.zeros(pshape, 'd')
        freqs = np.linspace(0, Fs // 2, Fn)
        P[ ..., 0 ] = (Sk[ ..., 0 ] * Sk[ ..., 0 ].conj()).real
        P[ ..., 1:Fl ] = 2 * (Sk[ ..., 1:Fl ] * Sk[ ..., 1:Fl ].conj()).real
        if Fn > Fl:
            P[ ..., Fn - 1 ] = (
            Sk[ ..., Fn - 1 ] * Sk[ ..., Fn - 1 ].conj()).real
    else:
        P = (Sk * Sk.conj()).real
        freqs = np.linspace(0, Fs, N, endpoint=False)
    if normalize:
        P /= (Fs * s.shape[ -1 ])
    return freqs, P


def ar_generator(N=512, sigma=1., coefs=None, drop_transients=0, v=None):
    """
    This generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    where v(n) is a stationary stochastic process with zero mean
    and variance = sigma. XXX: confusing variance notation
    Parameters
    ----------
    N : int
      sequence length
    sigma : float
      power of the white noise driving process
    coefs : sequence
      AR coefficients for k = 1, 2, ..., P
    drop_transients : int
      number of initial IIR filter transient terms to drop
    v : ndarray
      custom noise process
    Parameters
    ----------
    N : float
       The number of points in the AR process generated. Default: 512
    sigma : float
       The variance of the noise in the AR process. Default: 1
    coefs : list or array of floats
       The AR model coefficients. Default: [2.7607, -3.8106, 2.6535, -0.9238],
       which is a sequence shown to be well-estimated by an order 8 AR system.
    drop_transients : float
       How many samples to drop from the beginning of the sequence (the
       transient phases of the process), so that the process can be considered
       stationary.
    v : float array
       Optionally, input a specific sequence of noise samples (this over-rides
       the sigma parameter). Default: None
    Returns
    -------
    u : ndarray
       the AR sequence
    v : ndarray
       the unit-variance innovations sequence
    coefs : ndarray
       feedback coefficients from k=1,len(coefs)
    The form of the feedback coefficients is a little different than
    the normal linear constant-coefficient difference equation. Therefore
    the transfer function implemented in this method is
    H(z) = sigma**0.5 / ( 1 - sum_k coefs(k)z**(-k) )    1 <= k <= P
    Examples
    --------
    >>> import nitime.algorithms as alg
    >>> ar_seq, nz, alpha = ar_generator()
    >>> fgrid, hz = alg.freq_response(1.0, a=np.r_[1, -alpha])
    >>> sdf_ar = (hz * hz.conj()).real
    """
    from scipy.signal import lfilter
    if coefs is None:
        # this sequence is shown to be estimated well by an order 8 AR system
        coefs = np.array([ 2.7607, -3.8106, 2.6535, -0.9238 ])
    else:
        coefs = np.asarray(coefs)

    # The number of terms we generate must include the dropped transients, and
    # then at the end we cut those out of the returned array.
    N += drop_transients

    # Typically uses just pass sigma in, but optionally they can provide their
    # own noise vector, case in which we use it
    if v is None:
        v = np.random.normal(size=N)
        v -= v[ drop_transients: ].mean()

    b = [ sigma**0.5 ]
    a = np.r_[ 1, -coefs ]
    u = sig.lfilter(b, a, v)

    # Only return the data after the drop_transients terms
    return u[ drop_transients: ], v[ drop_transients: ], coefs


def freq_response(b, a=1., n_freqs=1024, sides='onesided'):
    """
    Returns the frequency response of the IIR or FIR filter described
    by beta and alpha coefficients.
    Parameters
    ----------
    b : beta sequence (moving average component)
    a : alpha sequence (autoregressive component)
    n_freqs : size of frequency grid
    sides : {'onesided', 'twosided'}
       compute frequencies between [-PI,PI), or from [0, PI]
    Returns
    -------
    fgrid, H(e^jw)
    Notes
    -----
    For a description of the linear constant-coefficient difference equation,
    see
    http://en.wikipedia.org/wiki/Z-transform
    """
    # transitioning to scipy freqz
    from scipy.signal import freqz
    real_n = n_freqs // 2 + 1 if sides == 'onesided' else n_freqs
    return freqz(b, a=a, worN=real_n, whole=sides != 'onesided')


class MultitaperSpectralTests(unittest.TestCase):
    def test_dpss_windows_short(self):
        """Are eigenvalues representing spectral concentration near unity?"""
        # these values from Percival and Walden 1993
        _, l = mts.dpss_windows(31, 6, 4)
        unos = np.ones(4)
        assert_allclose(l, unos)
        _, l = mts.dpss_windows(31, 7, 4)
        assert_allclose(l, unos)
        _, l = mts.dpss_windows(31, 8, 4)
        assert_allclose(l, unos)
        _, l = mts.dpss_windows(31, 8, 4.2)
        assert_allclose(l, unos)

    def test_dpss_windows_with_matlab(self):
        """Do the dpss windows resemble the equivalent matlab result
        The variable b is read in from a text file generated by issuing:
        dpss(100,2)
        in matlab
        """
        a, _ = mts.dpss_windows(100, 2, 4)
        b = np.loadtxt(os.path.join(test_dir_path, 'dpss_testdata1.txt'))
        assert_allclose(a, b.T)

    def test_tapered_spectra(self):
        """ Testing, if wrong input to tapered_spectra triggers warnings. """
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            mts.tapered_spectra(s=np.ones((2, 4)),
                                tapers=(1, 2),
                                NFFT=3,
                                low_bias=False)
            self.assertTrue(len(w) == 1)

    def test_get_spectra(self):
        """Testing get_spectra"""
        # adapt this to include the welchs_psd from elephant?
        t = np.linspace(0, 16 * np.pi, 2 ** 10)
        x = (np.sin(t) + np.sin(2 * t) + np.sin(3 * t) +
             0.1 * np.random.rand(t.shape[ -1 ]))

        N = x.shape[ -1 ]
        f_multi_taper = mts.get_spectra(x, method={
            'this_method': 'multi_taper_csd'})

        self.assertEqual(f_multi_taper[ 0 ].shape, (N // 2 + 1,))

        # Test for multi-channel data
        x = np.reshape(x, (2, x.shape[ -1 ] // 2))
        N = x.shape[ -1 ]

        f_multi_taper = mts.get_spectra(x, method={
            'this_method': 'multi_taper_csd'})

        self.assertEqual(f_multi_taper[ 0 ].shape[ 0 ], N / 2 + 1)

    def test_mtm_cross_spectrum(self):
        """ Testing mtm_cross_spectrum with incompatible inputs. """
        tx = np.ones(400)
        ty = np.ones(401)
        weights = tx
        with self.assertRaises(Exception) as context:
            mts.mtm_cross_spectrum(tx, ty, weights, sides='twosided')
            self.assertTrue('shape mismatch' in context.exception)

    def test_multi_taper_psd(self):
        """ Power spectral estimates for a sequence with known spectre. """
        # generate the multi-tapered estimate of the spectrum:
        N = 2048.
        BW = 4.0
        array = np.linspace(0., 1., N)
        ar_seq = np.sin(35. * np.pi * array) + np.ones_like(array)
        f, psd_mt, nu = mts.multi_taper_psd(
                ar_seq, Fs=N, adaptive=True, jackknife=False, sides='onesided',
                BW=BW)
        self.assertTrue(np.sum(psd_mt >= 1e-3) <= BW)

    @dec.slow
    def test_dpss_windows_long(self):
        """ Test that very long dpss windows can be generated (using
        interpolation)"""

        # This one is generated using interpolation:
        a1, e = mts.dpss_windows(166800, 4, 8, interp_from=4096)

        # This one is calculated:
        a2, e = mts.dpss_windows(166800, 4, 8)

        # They should be very similar:
        assert_allclose(a1, a2, atol=1e-5)

        # They should both be very similar to the same one calculated in matlab
        # (using 'a = dpss(166800, 4, 8)').
        test_dir_path = os.path.join(elephant.__path__[ 0 ], 'test')
        matlab_long_dpss = np.load(
            os.path.join(test_dir_path, 'dpss_testdata2.npy'))
        # We only have the first window to compare against:
        # Both for the interpolated case:
        assert_allclose(a1[ 0 ], matlab_long_dpss, atol=1e-5)
        # As well as the calculated case:
        assert_allclose(a1[ 0 ], matlab_long_dpss, atol=1e-5)


def suite():
    suite = unittest.makeSuite(MultitaperSpectralTests, 'test')
    return suite

# def shortprint():
#     import matplotlib.pyplot as plt
#     import scipy.signal as sig
#     N = 512.
#     array = np.linspace(0., 1., N)
#     ar_seq = np.sin(20 * np.pi * array) + np.ones_like(array)
#     f, psd_mt, nu = mts.multi_taper_psd(
#             ar_seq, Fs=N, adaptive=True, jackknife=False, sides='onesided', NW=2., BW=4.)
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(211)
#     ax1.plot(np.linspace(0, 1., ar_seq.shape[ 0 ]), ar_seq)
#     ax2 = fig.add_subplot(212)
#     ax2.plot(f, psd_mt[ 0:f.shape[ 0 ] ], marker='.', c='r')
#     freqs, array2 = sig.welch(ar_seq, fs=N)
#     ax2.plot(freqs, array2[ 0:freqs.shape[ 0 ] ], marker='.', c='g')
#     plt.show()


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
