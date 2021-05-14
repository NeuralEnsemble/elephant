# -*- coding: utf-8 -*-
"""
Unit tests for the kernels module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import math
import unittest
import warnings

import numpy as np
import quantities as pq
import scipy.integrate as spint
from numpy.testing import assert_array_almost_equal

import elephant.kernels as kernels


class kernel_TestCase(unittest.TestCase):
    def setUp(self):
        self.kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)

    def test_error_kernels(self):
        """
        Test of various error cases in the kernels module.
        """
        self.assertRaises(
            TypeError, kernels.RectangularKernel, sigma=2.0)
        self.assertRaises(
            ValueError, kernels.RectangularKernel, sigma=-0.03 * pq.s)
        self.assertRaises(
            ValueError, kernels.AlphaKernel, sigma=2.0 * pq.ms,
            invert=2)
        rec_kernel = kernels.RectangularKernel(sigma=0.3 * pq.ms)
        self.assertRaises(
            TypeError, rec_kernel, [1, 2, 3])
        self.assertRaises(
            TypeError, rec_kernel, [1, 2, 3] * pq.V)
        kernel = kernels.Kernel(sigma=0.3 * pq.ms)
        self.assertRaises(
            NotImplementedError, kernel._evaluate, [1, 2, 3] * pq.V)
        self.assertRaises(
            NotImplementedError, kernel.boundary_enclosing_area_fraction,
            fraction=0.9)
        self.assertRaises(TypeError,
                          rec_kernel.boundary_enclosing_area_fraction, [1, 2])
        self.assertRaises(ValueError,
                          rec_kernel.boundary_enclosing_area_fraction, -10)
        self.assertEqual(kernel.is_symmetric(), False)
        self.assertEqual(rec_kernel.is_symmetric(), True)

    def test_alpha_kernel_extreme(self):
        alp_kernel = kernels.AlphaKernel(sigma=0.3 * pq.ms)
        quantile = alp_kernel.boundary_enclosing_area_fraction(0.9999999)
        self.assertAlmostEqual(quantile.magnitude, 4.055922083048838)

    def test_kernels_normalization(self):
        """
        Test that each kernel normalizes to area one.
        """
        sigma = 0.1 * pq.mV
        fraction = 0.9999
        kernel_resolution = sigma / 100.0
        kernel_list = [kernel_type(sigma, invert=False) for
                       kernel_type in self.kernel_types]
        for kernel in kernel_list:
            b = kernel.boundary_enclosing_area_fraction(fraction).magnitude
            n_points = int(2 * b / kernel_resolution.magnitude)
            restric_defdomain = np.linspace(
                -b, b, num=n_points) * sigma.units
            kern = kernel(restric_defdomain)
            norm = spint.cumtrapz(y=kern.magnitude,
                                  x=restric_defdomain.magnitude)[-1]
            self.assertAlmostEqual(norm, 1, delta=0.003)

    def test_kernels_stddev(self):
        """
        Test that the standard deviation calculated from the kernel (almost)
        equals the parameter sigma with which the kernel was constructed.
        """
        sigma = 0.5 * pq.s
        fraction = 0.9999
        kernel_resolution = sigma / 50.0
        for invert in (False, True):
            kernel_list = [kernel_type(sigma, invert) for
                           kernel_type in self.kernel_types]
            for kernel in kernel_list:
                b = kernel.boundary_enclosing_area_fraction(
                    fraction).magnitude
                n_points = int(2 * b / kernel_resolution.magnitude)
                restric_defdomain = np.linspace(
                    -b, b, num=n_points) * sigma.units
                kern = kernel(restric_defdomain)
                av_integr = kern * restric_defdomain
                average = spint.cumtrapz(
                    y=av_integr.magnitude,
                    x=restric_defdomain.magnitude)[-1] * sigma.units
                var_integr = (restric_defdomain - average) ** 2 * kern
                variance = spint.cumtrapz(
                    y=var_integr.magnitude,
                    x=restric_defdomain.magnitude)[-1] * sigma.units ** 2
                stddev = np.sqrt(variance)
                self.assertAlmostEqual(stddev, sigma, delta=0.01 * sigma)

    def test_kernel_boundary_enclosing(self):
        """
        Test whether the integral of the kernel with boundary taken from
        the return value of the method boundary_enclosing_area_fraction
        is (almost) equal to the input variable `fraction` of
        boundary_enclosing_area_fraction.
        """
        sigma = 0.5 * pq.s
        kernel_resolution = sigma / 500.0
        kernel_list = [kernel_type(sigma, invert=False) for
                       kernel_type in self.kernel_types]
        for fraction in np.arange(0.15, 1.0, 0.4):
            for kernel in kernel_list:
                b = kernel.boundary_enclosing_area_fraction(fraction).magnitude
                n_points = int(2 * b / kernel_resolution.magnitude)
                restric_defdomain = np.linspace(
                    -b, b, num=n_points) * sigma.units
                kern = kernel(restric_defdomain)
                frac = spint.cumtrapz(y=kern.magnitude,
                                      x=restric_defdomain.magnitude)[-1]
                self.assertAlmostEqual(frac, fraction, delta=0.002)

    def test_kernel_output_same_size(self):
        time_array = np.linspace(0, 10, num=20) * pq.s
        for kernel_type in self.kernel_types:
            kernel = kernel_type(sigma=1 * pq.s)
            kernel_points = kernel(time_array)
            self.assertEqual(len(kernel_points), len(time_array))

    def test_element_wise_only(self):
        # Test that kernel operation is applied element-wise without any
        # recurrent magic (e.g., convolution)
        np.random.seed(19)
        t_array = np.linspace(-10, 10, num=100) * pq.s
        t_shuffled = t_array.copy()
        np.random.shuffle(t_shuffled)
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=1 * pq.s, invert=invert)
                kernel_shuffled = np.sort(kernel(t_shuffled))
                kernel_expected = np.sort(kernel(t_array))
                assert_array_almost_equal(kernel_shuffled, kernel_expected)

    def test_kernel_pdf_range(self):
        t_array = np.linspace(-10, 10, num=1000) * pq.s
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=1 * pq.s, invert=invert)
                kernel_array = kernel(t_array)
                in_range = (kernel_array <= 1) & (kernel_array >= 0)
                self.assertTrue(in_range.all())

    def test_boundary_enclosing_area_fraction(self):
        # test that test_boundary_enclosing_area_fraction does not depend
        # on the invert
        sigma = 1 * pq.s
        fractions_test = np.linspace(0, 1, num=10, endpoint=False)
        for kern_cls in self.kernel_types:
            kernel = kern_cls(sigma=sigma, invert=False)
            kernel_inverted = kern_cls(sigma=sigma, invert=True)
            for fraction in fractions_test:
                self.assertAlmostEqual(
                    kernel.boundary_enclosing_area_fraction(fraction),
                    kernel_inverted.boundary_enclosing_area_fraction(fraction)
                )

    def test_icdf(self):
        sigma = 1 * pq.s
        fractions_test = np.linspace(0, 1, num=10, endpoint=False)
        for kern_cls in self.kernel_types:
            kernel = kern_cls(sigma=sigma, invert=False)
            kernel_inverted = kern_cls(sigma=sigma, invert=True)
            for fraction in fractions_test:
                # ICDF(0) for several kernels produces -inf
                # of fsolve complains about stuck at local optima
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    icdf = kernel.icdf(fraction)
                    icdf_inverted = kernel_inverted.icdf(fraction)
                if kernel.is_symmetric():
                    self.assertAlmostEqual(icdf, icdf_inverted)
                else:
                    # AlphaKernel, ExponentialKernel
                    self.assertGreaterEqual(icdf, 0 * pq.s)
                    self.assertLessEqual(icdf_inverted, 0 * pq.s)

    def test_cdf_icdf(self):
        sigma = 1 * pq.s
        fractions_test = np.linspace(0, 1, num=10, endpoint=False)
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=sigma, invert=invert)
                for fraction in fractions_test:
                    # ICDF(0) for several kernels produces -inf
                    # of fsolve complains about stuck at local optima
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        self.assertAlmostEqual(
                            kernel.cdf(kernel.icdf(fraction)), fraction)

    def test_icdf_cdf(self):
        sigma = 1 * pq.s
        times = np.linspace(-10, 10) * sigma.units
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=sigma, invert=invert)
                for t in times:
                    cdf = kernel.cdf(t)
                    self.assertGreaterEqual(cdf, 0.)
                    self.assertLessEqual(cdf, 1.)
                    if 0 < cdf < 1:
                        self.assertAlmostEqual(
                            kernel.icdf(cdf), t, places=2)

    def test_icdf_at_1(self):
        sigma = 1 * pq.s
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=sigma, invert=invert)
                if isinstance(kernel, (kernels.RectangularKernel,
                                       kernels.TriangularKernel)):
                    icdf = kernel.icdf(1.0)
                    # check finite
                    self.assertLess(np.abs(icdf.magnitude), np.inf)
                else:
                    self.assertRaises(ValueError, kernel.icdf, 1.0)

    def test_cdf_symmetric(self):
        sigma = 1 * pq.s
        cutoff = 1e2 * sigma  # a large value
        times = np.linspace(-cutoff, cutoff, num=10)
        kern_symmetric = filter(lambda kern_type: issubclass(
            kern_type, kernels.SymmetricKernel), self.kernel_types)
        for kern_cls in kern_symmetric:
            kernel = kern_cls(sigma=sigma, invert=False)
            kernel_inverted = kern_cls(sigma=sigma, invert=True)
            for t in times:
                self.assertAlmostEqual(kernel.cdf(t), kernel_inverted.cdf(t))


class KernelOldImplementation(unittest.TestCase):
    def setUp(self):
        self.kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)
        self.sigma = 1 * pq.s
        self.time_input = np.linspace(-10, 10, num=100) * self.sigma.units

    def test_triangular(self):
        def evaluate_old(t):
            t_units = t.units
            t_abs = np.abs(t.magnitude)
            tau = math.sqrt(6) * kernel.sigma.rescale(t_units).magnitude
            kernel_pdf = (t_abs < tau) * 1 / tau * (1 - t_abs / tau)
            kernel_pdf = pq.Quantity(kernel_pdf, units=1 / t_units)
            return kernel_pdf

        for invert in (False, True):
            kernel = kernels.TriangularKernel(self.sigma, invert=invert)
            assert_array_almost_equal(kernel(self.time_input),
                                      evaluate_old(self.time_input))

    def test_gaussian(self):
        def evaluate_old(t):
            t_units = t.units
            t = t.magnitude
            sigma = kernel.sigma.rescale(t_units).magnitude
            kernel_pdf = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(
                -0.5 * (t / sigma) ** 2)
            kernel_pdf = pq.Quantity(kernel_pdf, units=1 / t_units)
            return kernel_pdf

        for invert in (False, True):
            kernel = kernels.GaussianKernel(self.sigma, invert=invert)
            assert_array_almost_equal(kernel(self.time_input),
                                      evaluate_old(self.time_input))

    def test_laplacian(self):
        def evaluate_old(t):
            t_units = t.units
            t = t.magnitude
            tau = kernel.sigma.rescale(t_units).magnitude / math.sqrt(2)
            kernel_pdf = 1 / (2 * tau) * np.exp(-np.abs(t / tau))
            kernel_pdf = pq.Quantity(kernel_pdf, units=1 / t_units)
            return kernel_pdf

        for invert in (False, True):
            kernel = kernels.LaplacianKernel(self.sigma, invert=invert)
            assert_array_almost_equal(kernel(self.time_input),
                                      evaluate_old(self.time_input))

    def test_exponential(self):
        def evaluate_old(t):
            t_units = t.units
            t = t.magnitude
            tau = kernel.sigma.rescale(t_units).magnitude
            if not kernel.invert:
                kernel_pdf = (t >= 0) * 1 / tau * np.exp(-t / tau)
            else:
                kernel_pdf = (t <= 0) * 1 / tau * np.exp(t / tau)
            kernel_pdf = pq.Quantity(kernel_pdf, units=1 / t_units)
            return kernel_pdf

        for invert in (False, True):
            kernel = kernels.ExponentialKernel(self.sigma, invert=invert)
            assert_array_almost_equal(kernel(self.time_input),
                                      evaluate_old(self.time_input))


class KernelMedianIndex(unittest.TestCase):
    def setUp(self):
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)
        self.sigma = 1 * pq.s
        self.time_input = np.linspace(-10, 10, num=100) * self.sigma.units
        self.kernels = []
        for kern_cls in kernel_types:
            for invert in (False, True):
                self.kernels.append(kern_cls(self.sigma, invert=invert))

    def test_small_size(self):
        time_empty = [] * pq.s
        time_size_2 = [0, 1] * pq.s
        for kernel in self.kernels:
            self.assertRaises(ValueError, kernel.median_index, time_empty)
            median_id = kernel.median_index(time_size_2)
            self.assertEqual(median_id, 0)

    def test_not_sorted(self):
        np.random.seed(9)
        np.random.shuffle(self.time_input)
        for kernel in self.kernels:
            self.assertRaises(ValueError, kernel.median_index, self.time_input)

    def test_non_support(self):
        time_negative = np.linspace(-100, -20) * pq.s
        for kernel in self.kernels:
            if isinstance(kernel, (kernels.GaussianKernel,
                                   kernels.LaplacianKernel)):
                continue
            kernel.invert = False
            median_id = kernel.median_index(time_negative)
            self.assertEqual(median_id, len(time_negative) // 2)
            self.assertAlmostEqual(kernel.cdf(time_negative[median_id]), 0.)

    def test_old_implementation(self):
        def median_index(t):
            cumsum = kernel(t).cumsum()
            dt = (t[-1] - t[0]) / (len(t) - 1)
            quantiles = cumsum * dt
            return np.nonzero(quantiles >= 0.5)[0].min()

        for kernel in self.kernels:
            median_id = kernel.median_index(self.time_input)
            median_id_old = median_index(self.time_input)
            # the old implementation was off by 1 index, because the cumsum
            # did not start with 0 (the zero element should have been added
            # in the cumsum in old implementation).
            self.assertLessEqual(abs(median_id - median_id_old), 1)


if __name__ == '__main__':
    unittest.main()
