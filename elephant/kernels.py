# -*- coding: utf-8 -*-
"""
Definition of classes of various kernel functions to be used
in convolution, e.g., in firing rate estimation.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import quantities as pq
import numpy as np
import scipy.special

default_kernel_area_fraction = 0.99999


class Kernel(object):
    """ Base class for kernels.  """

    def __init__(self, sigma, direction):
        """
        :param sigma: Standard deviation of the kernel.
        :type sigma: Quantity scalar
        :param direction: direction of asymmetric kernels
        :type direction: integer of value 1 or -1
        """
        self.sigma = sigma
        self.direction = direction

    def __call__(self, t):
        """ Evaluates the kernel at all time points in the array `t`.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        if  t.dimensionality.simplified != self.sigma.dimensionality.simplified:
            raise TypeError("The dimensionality of sigma and the input array to the callable kernel object "
                        "must be the same. Otherwise a normalization to 1 of the kernel cannot be performed.")

        self.sigma = self.sigma.rescale(t.units)

        return self.evaluate(t)

    def evaluate(self, t):
        """ Evaluates the kernel.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        raise NotImplementedError()

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        """ Calculates the boundary :math:`b` so that the integral from
        :math:`-b` to :math:`b` encloses at least a certain fraction of the
        integral over the complete kernel.

        :param float fraction: Fraction of the whole area which at least has to
            be enclosed.
        :returns: boundary
        :rtype: Quantity scalar
        """
        raise NotImplementedError()

    def m_idx(self, t):
        """
        Estimates the index of the Median of the kernel.
        This parameter is not mandatory for symmetrical kernels but it is required
        when asymmetrical kernels have to be aligned at their median.

        :param t: Time points at which the kernel is evaluated
        :type t: Quantity 1D
        :returns: Estimated value of the kernel median
        :rtype: int

        Remarks:
        The formula in this method using retrieval of the sampling period from t
        only works for t with equidistant time intervals!
        The formula calculates the Median slightly wrong by the potentially ignored probability in the
        distribution corresponding to lower values than the minimum in the array t.
        """
        return np.nonzero(self(t).cumsum() * (t[len(t)-1] -t[0]) / (len(t)-1) >= 0.5)[0].min()

    def is_symmetric(self):
        """ Should return `True` if the kernel is symmetric. """
        return False


class SymmetricKernel(Kernel):
    """ Base class for symmetric kernels. """

    def __init__(self, sigma, direction):
        """
        :param sigma: Standard deviation of the kernel.
        :type sigma: Quantity scalar
        """
        Kernel.__init__(self, sigma, direction)

    def is_symmetric(self):
        return True


class RectangularKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} \frac{1}{2 \tau}, & |t| < \tau \\
    0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{3} \sigma` corresponding to the half width of the kernel.
    """
    min_stddevmultfactor = np.sqrt(3.0)

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        return (0.5 / (np.sqrt(3.0) * self.sigma)) * \
               (np.absolute(t) < np.sqrt(3.0) * self.sigma)

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return np.sqrt(3.0) * self.sigma


class TriangularKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{ \begin{array}{ll} \frac{1}{\tau} (1
    - \frac{|t|}{\tau}), & |t| < \tau \\ 0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{6} \sigma` corresponding to the half width of the kernel.
    """
    min_stddevmultfactor = np.sqrt(6.0)

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        return (1.0 / (np.sqrt(6.0) * self.sigma)) * np.maximum(
            0.0,
            (1.0 - (np.absolute(t) / (np.sqrt(6.0) * self.sigma)).magnitude))

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return np.sqrt(6.0) * self.sigma


class EpanechnikovLikeKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (3 /(4 d)) (1 - (t / d)^2), & |t| < d \\
    0, & |t| \geq d \end{array} \right`
    with :math:`d = \sqrt{5} \sigma` being the half width of the kernel.

    The Epanechnikov kernel under full consideration of its axioms has a half
    width of :math:`sqrt{5}`. Ignoring one axiom also the respective kernel
    with half width = 1 can be called Epanechnikov kernel.
    ( https://de.wikipedia.org/wiki/Epanechnikov-Kern )
    However, arbitrary width of this type of kernel is here preferred to be
    called 'Epanechnikov-like' kernel.
    """
    min_stddevmultfactor = np.sqrt(5.0)

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        return (3.0 / (4.0 * np.sqrt(5.0) * self.sigma)) * np.maximum(
            0.0,
            1 - (t / (np.sqrt(5.0) * self.sigma)).magnitude ** 2)

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return np.sqrt(5.0) * self.sigma


class GaussianKernel(SymmetricKernel):
    """ :math:`K(t) = (\frac{1}{\sigma \sqrt{2 \pi}}) \exp(-\frac{t^2}{2 \sigma^2})`
    with :math:`\sigma` being the standard deviation.
    """
    min_stddevmultfactor = 3.0

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        return (1.0 / (np.sqrt(2.0 * np.pi) * self.sigma)) * np.exp(
            -0.5 * (t / self.sigma).magnitude ** 2)

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return self.sigma * np.sqrt(2.0) * scipy.special.erfinv(fraction)


class LaplacianKernel(SymmetricKernel):
    """ :math:`K(t) = \frac{1}{2 \tau} \exp(-|\frac{t}{\tau}|)`
    with :math:`\tau = \sigma / \sqrt{2}`.
    """
    min_stddevmultfactor = 3.0

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        return (1 / (np.sqrt(2.0) * self.sigma)) * np.exp(
            -(np.absolute(t) * np.sqrt(2.0) / self.sigma).magnitude)

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return -self.sigma * np.log(1.0 - fraction) / np.sqrt(2.0)


## Potential further symmetric kernels from Wiki Kernels (statistics):
## Quartic (biweight), Triweight, Tricube, Cosine, Logistics, Silverman


class ExponentialKernel(Kernel):
## class ExponentialKernel(AsymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (1 / \tau) \exp{-t / \tau}, & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma`.
    """
    min_stddevmultfactor = 3.0

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        if self.direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: (1.0 / self.sigma.magnitude) * np.exp(
                        (-t / self.sigma).magnitude)]) / t.units
        elif self.direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: (1.0 / self.sigma.magnitude) * np.exp(
                        (t / self.sigma).magnitude),
                    lambda t: 0]) / t.units
        return kernel

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return -self.sigma * np.log(1.0 - fraction)


class AlphaKernel(Kernel):
## class AlphaKernel(AsymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (1 / (\tau)^2) t \exp{-t / \tau}, & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma / \sqrt{2}`.
    """
    min_stddevmultfactor = 3.0

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    def evaluate(self, t):
        if self.direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: 2.0 * (t / (self.sigma)**2).magnitude *
                              np.exp((-t * np.sqrt(2.0) / self.sigma).magnitude)]) / t.units
        elif self.direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: -2.0 * (t / (self.sigma)**2).magnitude *
                              np.exp((t * np.sqrt(2.0) / self.sigma).magnitude),
                    lambda t: 0 ]) / t.units
        return kernel

    def boundary_enclosing_at_least(self, fraction = default_kernel_area_fraction):
        return - self.sigma * (1 + scipy.special.lambertw((fraction - 1.0) / np.exp(1.0))) / np.sqrt(2.0)
