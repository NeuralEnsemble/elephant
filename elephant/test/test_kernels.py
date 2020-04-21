# -*- coding: utf-8 -*-
"""
Unit tests for the kernels module.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

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
        self.fraction = 0.9999

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
        kernel_resolution = sigma / 100.0
        kernel_list = [kernel_type(sigma, invert=False) for
                       kernel_type in self.kernel_types]
        for kernel in kernel_list:
            b = kernel.boundary_enclosing_area_fraction(
                self.fraction).magnitude
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
        kernel_resolution = sigma / 50.0
        for invert in (False, True):
            kernel_list = [kernel_type(sigma, invert) for
                           kernel_type in self.kernel_types]
            for kernel in kernel_list:
                b = kernel.boundary_enclosing_area_fraction(
                    self.fraction).magnitude
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

    def test_median_index(self):
        resolution = 3
        t_array = np.linspace(0, 1, num=10 ** resolution) * pq.s
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=1 * pq.s, invert=invert)
                kernel_array = kernel(t_array)
                median_index = kernel.median_index(t_array)
                median = np.median(kernel_array)
                self.assertAlmostEqual(kernel_array[median_index], median,
                                       places=resolution)

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
                kernel_shuffled = kernel(t_shuffled)
                kernel_shuffled.sort()
                kernel_expected = kernel(t_array)
                kernel_expected.sort()
                assert_array_almost_equal(kernel_shuffled, kernel_expected)

    def test_kernel_pdf_range(self):
        t_array = np.linspace(-10, 10, num=1000) * pq.s
        for kern_cls in self.kernel_types:
            for invert in (False, True):
                kernel = kern_cls(sigma=1 * pq.s, invert=invert)
                kernel_array = kernel(t_array)
                in_range = (kernel_array <= 1) & (kernel_array >= 0)
                self.assertTrue(in_range.all())


if __name__ == '__main__':
    unittest.main()
