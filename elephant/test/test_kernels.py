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
import elephant.kernels as kernels


class kernel_TestCase(unittest.TestCase):
    def setUp(self):
        self.kernel_types = [obj for obj in kernels.__dict__.values()
                             if isinstance(obj, type) and
                             issubclass(obj, kernels.Kernel) and
                             hasattr(obj, "_evaluate") and
                             obj is not kernels.Kernel and
                             obj is not kernels.SymmetricKernel]
        self.fraction = 0.9999

    def test_error_kernels(self):
        """
        Test of various error cases in the kernels module.
        """
        self.assertRaises(
            TypeError, kernels.RectangularKernel, sigma=2.0)
        self.assertRaises(
            ValueError, kernels.RectangularKernel, sigma=-0.03*pq.s)
        self.assertRaises(
            ValueError, kernels.RectangularKernel, sigma=2.0*pq.ms,
            invert=2)
        rec_kernel = kernels.RectangularKernel(sigma=0.3*pq.ms)
        self.assertRaises(
            TypeError, rec_kernel, [1, 2, 3])
        self.assertRaises(
            TypeError, rec_kernel, [1, 2, 3]*pq.V)
        kernel = kernels.Kernel(sigma=0.3*pq.ms)
        self.assertRaises(
            NotImplementedError, kernel._evaluate, [1, 2, 3]*pq.V)
        self.assertRaises(
            NotImplementedError, kernel.boundary_enclosing_area_fraction,
            fraction=0.9)
        self.assertRaises(TypeError,
                          rec_kernel.boundary_enclosing_area_fraction, [1, 2])
        self.assertRaises(ValueError,
                          rec_kernel.boundary_enclosing_area_fraction, -10)
        self.assertEqual(kernel.is_symmetric(), False)
        self.assertEqual(rec_kernel.is_symmetric(), True)

    @unittest.skip('very time-consuming test')
    def test_error_alpha_kernel(self):
        alp_kernel = kernels.AlphaKernel(sigma=0.3*pq.ms)
        self.assertRaises(ValueError,
            alp_kernel.boundary_enclosing_area_fraction, 0.9999999)

    def test_kernels_normalization(self):
        """
        Test that each kernel normalizes to area one.
        """
        sigma = 0.1 * pq.mV
        kernel_resolution = sigma / 100.0
        kernel_list = [kernel_type(sigma, invert=False) for
                       kernel_type in self.kernel_types]
        for kernel in kernel_list:
            b = kernel.boundary_enclosing_area_fraction(self.fraction).magnitude
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
                b = kernel.boundary_enclosing_area_fraction(self.fraction).magnitude
                n_points = int(2 * b / kernel_resolution.magnitude)
                restric_defdomain = np.linspace(
                    -b, b, num=n_points) * sigma.units
                kern = kernel(restric_defdomain)
                av_integr = kern * restric_defdomain
                average = spint.cumtrapz(y=av_integr.magnitude,
                                         x=restric_defdomain.magnitude)[-1] * \
                          sigma.units
                var_integr = (restric_defdomain-average)**2 * kern
                variance = spint.cumtrapz(y=var_integr.magnitude,
                                          x=restric_defdomain.magnitude)[-1] * \
                           sigma.units**2
                stddev = np.sqrt(variance)
                self.assertAlmostEqual(stddev, sigma, delta=0.01*sigma)

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


if __name__ == '__main__':
    unittest.main()
