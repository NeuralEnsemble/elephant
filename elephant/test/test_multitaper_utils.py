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

Tests utilities for multitapered spectral time series analysis.

"""

import unittest
import numpy as np
import numpy.random as npr
import warnings
import elephant.multitaper_utils as mtu
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
import numpy.linalg as la

def expected_jk_variance(K):
    """Compute the expected value of the jackknife variance estimate
    over K windows below. This expected value formula is based on the
    asymptotic expansion of the trigamma function derived in
    [Thompson_1994]

    Paramters
    ---------

    K : int
      Number of tapers used in the multitaper method

    Returns
    -------

    evar : float
      Expected value of the jackknife variance estimator

    """

    kf = float(K)
    return ((1 / kf) * (kf - 1) / (kf - 0.5) *
            ((kf - 1) / (kf - 2)) ** 2 * (kf - 3) / (kf - 2))


class MultitaperUtilityTests(unittest.TestCase):

    def test_adaptive_weights(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            mtu._adaptive_weights(np.ones(1), np.ones(1), 'three_sided', -1)
            self.assertTrue(len(w) == 3)

    def test_tridi_inverse_iteration(self):
        import scipy.linalg as la
        import time as time
        from scipy.sparse import spdiags
        # set up a spectral concentration eigenvalue problem for testing
        N = 2000
        NW = 4
        K = 8
        W = float(NW) / N
        start = time.time()
        nidx = np.arange(N, dtype='d')
        ab = np.zeros((2, N), 'd')
        # store this separately for tridisolve later
        sup_diag = np.zeros((N,), 'd')
        sup_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
        ab[0, 1:] = sup_diag[:-1]
        ab[1] = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
        # only calculate the highest Kmax-1 eigenvalues
        w = la.eigvals_banded(ab, select='i', select_range=(N - K, N - 1))
        w = w[::-1]
        E = np.zeros((K, N), 'd')
        t = np.linspace(0, np.pi, N)
        # make sparse tridiagonal matrix for eigenvector check
        sp_data = np.zeros((3, N), 'd')
        sp_data[0, :-1] = sup_diag[:-1]
        sp_data[1] = ab[1]
        sp_data[2, 1:] = sup_diag[:-1]
        A = spdiags(sp_data, [-1, 0, 1], N, N)
        E = np.zeros((K, N), 'd')
        for j in range(K):
            e = mtu._tridi_inverse_iteration(
                ab[1], sup_diag, w[j], x0=np.sin((j + 1) * t)
            )
            b = A * e
            self.assertTrue(
                np.linalg.norm(np.abs(b) - np.abs(w[j] * e)) < 1e-8,
                'Inverse iteration eigenvector solution is inconsistent with '
                'given eigenvalue'
            )
            E[j] = e

        # also test orthonormality of the eigenvectors
        ident = np.dot(E, E.T)
        stop = time.time()
        assert_allclose(ident, np.eye(K), rtol=0.0, atol=1e-12,
                        err_msg='Result incorrect for unity.')
        self.assertTrue(stop - start <= 10.0, 'Running time too high.')

    def test_remove_bias(self):
        x = np.arange(64).reshape(4, 4, 4)
        x0 = mtu._remove_bias(x, axis=1)
        self.assertEqual((x0.mean(axis=1) == 0).all(), True)

    def test_crosscov(self):
        N = 128
        ar_seq1 = npr.randn(N) + np.sin(np.linspace(0., 2. * np.pi, 128))
        ar_seq2 = npr.randn(N) + np.cos(np.linspace(0., 2. * np.pi, 128))

        def corr(x, y, all_lags=True):
            # Computes sxy[k] = E{x[n]*y[n+k]}
            x = mtu._remove_bias(x, 0)
            y = mtu._remove_bias(y, 0)
            lx, ly = len(x), len(y)
            pad_len = lx + ly - 1
            sxy = np.correlate(x, y, mode='full') / lx
            if all_lags:
                return sxy
            c_idx = pad_len // 2
            return sxy[c_idx:]

        for all_lags in (True, False):
            sxy = mtu.crosscov(ar_seq1, ar_seq2, all_lags=all_lags)
            # catch the given warning about wrong arguments
            sxy_ref = corr(ar_seq1, ar_seq2, all_lags=all_lags)
            err = sxy_ref - sxy
            mse = np.dot(err, err) / N
            self.assertTrue(mse < 1e-12,
                            'Large mean square error w.r.t. reference cross covariance')

    def test_autocorr(self):
        N = 128
        ar_seq = npr.randn(N) + np.sin(np.linspace(0., 2. * np.pi, 128))
        rxx = mtu.autocorr(ar_seq)
        self.assertTrue(rxx[0] == rxx.max(),
                        'Zero lag autocorrelation is not maximum autocorrelation')
        rxx = mtu.autocorr(ar_seq, all_lags=True)
        self.assertTrue(rxx[127] == rxx.max(),
                        'Zero lag autocorrelation is not maximum autocorrelation')

    def test_tridisolve(self):
        """ Test the banded matrix algorithm on the identity matrix """
        N = 4
        line1 = np.ones(N)
        line2 = np.zeros(N)
        b = np.ones(N)
        mtu._tridisolve(line1, line2, b, overwrite_b=True)
        assert_allclose(b, line1)


def suite():
    suite = unittest.makeSuite(MultitaperUtilityTests, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
