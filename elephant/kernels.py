# -*- coding: utf-8 -*-
"""
Definition of classes of various kernel functions to be used in convolution,
e.g., for data smoothing (low pass filtering) or firing rate estimation.

Currently implemented forms are rectangular, triangular, epanechnikovlike,
gaussian, laplacian, exponential, and alpha function, of which exponential
and alpha kernels have asymmetric form.

Exponential and alpha kernels may also be used to represent postynaptic
currents / potentials in a linear (current-based) model.

Examples of usage:
kernel1 = kernels.GaussianKernel(sigma=100*ms)
kernel2 = kernels.ExponentialKernel(sigma=8*mm, direction=-1)

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import quantities as pq
import numpy as np
import scipy.special

default_kernel_area_fraction = 0.99999


class Kernel(object):
    """
    Base class for kernels.

    :param sigma: Standard deviation of the kernel.
        This parameter determines the time resolution of the kernel estimate
        and makes different kernels comparable (Meier R, Egert U, Aertsen A,
        Nawrot MP, "FIND - a unified framework for neural data analysis";
        Neural Netw. 2008 Oct; 21(8):1085-93.) for symmetric kernels.
    :type sigma: Quantity scalar
    :param direction (optional): Orientation of asymmetric kernels,
            e.g., exponential or alpha kernels.
    :type direction: integer of value 1 or -1, default: 1
    """

    def __init__(self, sigma, direction=1):

        if not (isinstance(sigma, pq.Quantity)):
            raise TypeError("sigma must be a quantity!")

        if sigma.magnitude < 0:
            raise ValueError("sigma cannot be negative!")

        if direction not in (1, -1):
            raise ValueError("direction must be either 1 or -1")

        self.sigma = sigma
        self.direction = direction

    def __call__(self, t):
        """ Evaluates the kernel at all points in the array `t`.

        :param t: Interval on which the kernel is evaluated, not necessarily
            a time interval.
        :type t: Quantity 1D
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        if not (isinstance(t, pq.Quantity)):
            raise TypeError("The argument of the kernel callable must be "
                            "of type quantity!")

        if t.dimensionality.simplified != self.sigma.dimensionality.simplified:
            raise TypeError("The dimensionality of sigma and the input array "
                            "to the callable kernel object must be the same. "
                            "Otherwise a normalization to 1 of the kernel "
                            "cannot be performed.")

        self._sigma_scaled = self.sigma.rescale(t.units)
        # A hidden variable _sigma_scaled is introduced here in order to avoid
        # accumulation of floating point errors of sigma upon multiple
        # usages of the __call__ - function for the same Kernel instance.

        return self._evaluate(t)

    def _evaluate(self, t):
        """ Evaluates the kernel.

        :param t: Interval on which the kernel is evaluated, not necessarily
            a time interval.
        :type t: Quantity 1D
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        raise NotImplementedError("The Kernel class should not be used directly, "
                                  "instead the subclasses for the single kernels.")

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        """ Calculates the boundary :math:`b` so that the integral from
        :math:`-b` to :math:`b` encloses at least a certain fraction of the
        integral over the complete kernel. By definition the returned value
        of the method boundary_enclosing_area_fraction is hence non-negative, even
        if the whole probability mass of the kernel is concentrated over
        negative support for direction-inverted kernels.

        :param float fraction: Fraction of the whole area which at least has
            to be enclosed.
        :returns: boundary
        :rtype: Quantity scalar
        """
        raise NotImplementedError("The Kernel class should not be used directly, "
                                  "instead the subclasses for the single kernels.")

    def _check_fraction(self, fraction):
        """
        Checks the input variable of the method boundary_enclosing_area_fraction
        for validity of type and value.
        :param fraction: Fraction of the area under the kernel function.
        """
        if not isinstance(fraction, (float, int)):
            raise TypeError("`fraction` must be float or integer!")
        if not 0 <= fraction <= 1:
            raise ValueError("`fraction` must be in the interval [0, 1]!")

    def m_idx(self, t):
        """
        Estimates the index of the Median of the kernel.
        This parameter is not mandatory for symmetrical kernels but it is
        required when asymmetrical kernels have to be aligned at their median.

        :param t: Interval on which the kernel is evaluated,
        :type t: Quantity 1D
        :returns: Estimated value of the kernel median
        :rtype: int

        Remarks:
        The formula in this method using retrieval of the sampling interval
        from t only works for t with equidistant intervals!
        The formula calculates the Median slightly wrong by the potentially
        ignored probability in the distribution corresponding to lower values
        than the minimum in the array t.
        """
        return np.nonzero(self(t).cumsum() *
                          (t[len(t)-1] - t[0]) / (len(t) - 1) >= 0.5)[0].min()

    def is_symmetric(self):
        """ Should return `True` if the kernel is symmetric. """
        return False


class SymmetricKernel(Kernel):
    """ Base class for symmetric kernels. """

    # def __init__(self, sigma, direction):
    #     Kernel.__init__(self, sigma, direction)

    def is_symmetric(self):
        return True


class RectangularKernel(SymmetricKernel):
    """
    Class for rectangular kernels

    :math:`K(t) = \left\{\begin{array}{ll} \frac{1}{2 \tau}, & |t| < \tau \\
    0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{3} \sigma` corresponding to the half width
    of the kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `direction` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.
    """
    min_cutoff = np.sqrt(3.0)

    def _evaluate(self, t):
        return (0.5 / (np.sqrt(3.0) * self._sigma_scaled)) * \
               (np.absolute(t) < np.sqrt(3.0) * self._sigma_scaled)

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        # @doc_inherit
        self._check_fraction(fraction)
        return np.sqrt(3.0) * self.sigma


class TriangularKernel(SymmetricKernel):
    """
    Class for triangular kernels

    :math:`K(t) = \left\{ \begin{array}{ll} \frac{1}{\tau} (1
    - \frac{|t|}{\tau}), & |t| < \tau \\ 0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{6} \sigma` corresponding to the half width of the
    kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `direction` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.
    """
    min_cutoff = np.sqrt(6.0)

    def _evaluate(self, t):
        return (1.0 / (np.sqrt(6.0) * self._sigma_scaled)) * np.maximum(
            0.0,
            (1.0 - (np.absolute(t) /
                    (np.sqrt(6.0) * self._sigma_scaled)).magnitude))

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        self._check_fraction(fraction)
        return np.sqrt(6.0) * self.sigma


class EpanechnikovLikeKernel(SymmetricKernel):
    """
    Class for epanechnikov-like kernels

    :math:`K(t) = \left\{\begin{array}{ll} (3 /(4 d)) (1 - (t / d)^2),
    & |t| < d \\
    0, & |t| \geq d \end{array} \right`
    with :math:`d = \sqrt{5} \sigma` being the half width of the kernel.

    The Epanechnikov kernel under full consideration of its axioms has a half
    width of :math:`sqrt{5}`. Ignoring one axiom also the respective kernel
    with half width = 1 can be called Epanechnikov kernel.
    ( https://de.wikipedia.org/wiki/Epanechnikov-Kern )
    However, arbitrary width of this type of kernel is here preferred to be
    called 'Epanechnikov-like' kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `direction` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.
    """
    min_cutoff = np.sqrt(5.0)

    def _evaluate(self, t):
        return (3.0 / (4.0 * np.sqrt(5.0) * self._sigma_scaled)) * np.maximum(
            0.0,
            1 - (t / (np.sqrt(5.0) * self._sigma_scaled)).magnitude ** 2)

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        self._check_fraction(fraction)
        return np.sqrt(5.0) * self.sigma


class GaussianKernel(SymmetricKernel):
    """
    Class for gaussian kernels

    :math:`K(t) = (\frac{1}{\sigma \sqrt{2 \pi}})
    \exp(-\frac{t^2}{2 \sigma^2})`
    with :math:`\sigma` being the standard deviation.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `direction` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.
    """
    min_cutoff = 3.0

    def _evaluate(self, t):
        return (1.0 / (np.sqrt(2.0 * np.pi) * self._sigma_scaled)) * np.exp(
            -0.5 * (t / self._sigma_scaled).magnitude ** 2)

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        self._check_fraction(fraction)
        return self.sigma * np.sqrt(2.0) * scipy.special.erfinv(fraction)


class LaplacianKernel(SymmetricKernel):
    """
    Class for laplacian kernels

    :math:`K(t) = \frac{1}{2 \tau} \exp(-|\frac{t}{\tau}|)`
    with :math:`\tau = \sigma / \sqrt{2}`.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `direction` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.
    """
    min_cutoff = 3.0

    def _evaluate(self, t):
        return (1 / (np.sqrt(2.0) * self._sigma_scaled)) * np.exp(
            -(np.absolute(t) * np.sqrt(2.0) / self._sigma_scaled).magnitude)

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction) / np.sqrt(2.0)


# Potential further symmetric kernels from Wiki Kernels (statistics):
# Quartic (biweight), Triweight, Tricube, Cosine, Logistics, Silverman


# class ExponentialKernel(AsymmetricKernel):
class ExponentialKernel(Kernel):
    """
    Class for exponential kernels

    :math:`K(t) = \left\{\begin{array}{ll} (1 / \tau) \exp{-t / \tau},
    & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma`.
    """
    min_cutoff = 3.0

    def _evaluate(self, t):
        if self.direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: (1.0 / self._sigma_scaled.magnitude) * np.exp(
                        (-t / self._sigma_scaled).magnitude)]) / t.units
        elif self.direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: (1.0 / self._sigma_scaled.magnitude) * np.exp(
                        (t / self._sigma_scaled).magnitude),
                    lambda t: 0]) / t.units
        return kernel

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction)


# class AlphaKernel(AsymmetricKernel):
class AlphaKernel(Kernel):
    """
    Class for alpha kernels

    :math:`K(t) = \left\{\begin{array}{ll} (1 / (\tau)^2) t \exp{-t / \tau},
    & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma / \sqrt{2}`.
    """
    min_cutoff = 3.0

    def _evaluate(self, t):
        if self.direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: 2.0 * (t / self._sigma_scaled**2).magnitude *
                              np.exp((-t * np.sqrt(2.0) /
                                      self._sigma_scaled).magnitude)]) / t.units
        elif self.direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: -2.0 * (t / self._sigma_scaled**2).magnitude *
                              np.exp((t * np.sqrt(2.0) /
                                      self._sigma_scaled).magnitude),
                    lambda t: 0 ]) / t.units
        return kernel

    def boundary_enclosing_area_fraction(self,
                                    fraction=default_kernel_area_fraction):
        """
        An analytical expression for the boundary of the integral as a function
        of the area under the alpha kernel function cannot be given.
        Hence in this case the value of the boundary is determined by kernel-
        approximating numerical integration.
        """
        self._check_fraction(fraction)
        sigma_division = 500            # arbitrary choice
        self._sigma_scaled = self.sigma
        interval = self.sigma / sigma_division
        area = 0
        counter = 0
        while area < fraction:
            area += (self._evaluate((counter + 1) * self.direction * interval) +
                     self._evaluate(counter * self.direction * interval)) * interval / 2
            counter += 1
        return counter * interval

