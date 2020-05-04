# -*- coding: utf-8 -*-
"""
Definition of a hierarchy of classes for kernel functions to be used
in convolution, e.g., for data smoothing (low pass filtering) or
firing rate estimation.


Base kernel classes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: toctree/kernels/

    Kernel
    SymmetricKernel

Symmetric kernels
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: toctree/kernels/

    RectangularKernel
    TriangularKernel
    EpanechnikovLikeKernel
    GaussianKernel
    LaplacianKernel

Asymmetric kernels
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: toctree/kernels/

    ExponentialKernel
    AlphaKernel


Examples
--------
>>> import quantities as pq
>>> kernel1 = GaussianKernel(sigma=100*pq.ms)
>>> kernel2 = ExponentialKernel(sigma=8*pq.ms, invert=True)

:copyright: Copyright 2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import math

import numpy as np
import quantities as pq
import scipy.optimize
import scipy.special

__all__ = [
    'RectangularKernel', 'TriangularKernel', 'EpanechnikovLikeKernel',
    'GaussianKernel', 'LaplacianKernel', 'ExponentialKernel', 'AlphaKernel'
]


class Kernel(object):
    r"""
    This is the base class for commonly used kernels.

    **General definition of a kernel:**

    A function :math:`K(x, y)` is called a kernel function if
    :math:`\int{K(x, y) g(x) g(y) \textrm{d}x \textrm{d}y} \ \geq 0 \quad
    \forall g \in L_2`


    **Currently implemented kernels are:**

        * rectangular
        * triangular
        * epanechnikovlike
        * gaussian
        * laplacian
        * exponential (asymmetric)
        * alpha function (asymmetric)

    In neuroscience, a popular application of kernels is in performing
    smoothing operations via convolution. In this case, the kernel has the
    properties of a probability density, i.e., it is positive and normalized
    to one. Popular choices are the rectangular or Gaussian kernels.

    Exponential and alpha kernels may also be used to represent the
    postsynaptic current/potentials in a linear (current-based) model.

    Parameters
    ----------
    sigma : pq.Quantity
        Standard deviation of the kernel.
    invert : bool, optional
        If True, asymmetric kernels (e.g., exponential or alpha kernels) are
        inverted along the time axis.
        Default: False.

    Raises
    ------
    TypeError
        If `sigma` is not `pq.Quantity`.

        If `sigma` is negative.

        If `invert` is not `bool`.

    """

    def __init__(self, sigma, invert=False):
        if not isinstance(sigma, pq.Quantity):
            raise TypeError("'sigma' must be a quantity")

        if sigma.magnitude < 0:
            raise ValueError("'sigma' cannot be negative")

        if not isinstance(invert, bool):
            raise ValueError("'invert' must be bool")

        self.sigma = sigma
        self.invert = invert

    def __repr__(self):
        return "{cls}(sigma={sigma}, invert={invert})".format(
            cls=self.__class__.__name__, sigma=self.sigma, invert=self.invert)

    def __call__(self, t):
        """
        Evaluates the kernel at all points in the array `t`.

        Parameters
        ----------
        t : pq.Quantity
            Vector with the interval on which the kernel is evaluated,
            not necessarily a time interval.

        Returns
        -------
        pq.Quantity
            Vector with the result of the kernel evaluations.

        Raises
        ------
        TypeError
            If `t` is not `pq.Quantity`.

            If the dimensionality of `t` and :attr:`sigma` are different.

        """
        if not isinstance(t, pq.Quantity):
            raise TypeError("The argument 't' of the kernel callable must be "
                            "of type Quantity")

        if t.dimensionality.simplified != self.sigma.dimensionality.simplified:
            raise TypeError("The dimensionality of sigma and the input array "
                            "to the callable kernel object must be the same. "
                            "Otherwise a normalization to 1 of the kernel "
                            "cannot be performed.")

        return self._evaluate(t)

    def _evaluate(self, t):
        """
        Evaluates the kernel.

        Parameters
        ----------
        t : pq.Quantity
            Vector with the interval on which the kernel is evaluated, not
            necessarily a time interval.

        Returns
        -------
        pq.Quantity
            Vector with the result of the kernel evaluation.

        """
        raise NotImplementedError(
            "The Kernel class should not be used directly, "
            "instead the subclasses for the single kernels.")

    def boundary_enclosing_area_fraction(self, fraction):
        """
        Calculates the boundary :math:`b` so that the integral from
        :math:`-b` to :math:`b` encloses a certain fraction of the
        integral over the complete kernel.

        By definition the returned value is hence non-negative, even if the
        whole probability mass of the kernel is concentrated over negative
        support for inverted kernels.

        Parameters
        ----------
        fraction : float
            Fraction of the whole area which has to be enclosed.

        Returns
        -------
        pq.Quantity
            Boundary of the kernel containing area `fraction` under the
            kernel density.

        Raises
        ------
        ValueError
            If `fraction` was chosen too close to one, such that in
            combination with integral approximation errors the calculation of
            a boundary was not possible.

        """
        raise NotImplementedError(
            "The Kernel class should not be used directly, "
            "instead the subclasses for the single kernels.")

    @staticmethod
    def _check_fraction(fraction):
        """
        Checks the input variable of the method
        :attr:`boundary_enclosing_area_fraction` for validity of type and
        value.

        Parameters
        ----------
        fraction : float or int
            Fraction of the area under the kernel function.

        Raises
        ------
        TypeError
            If `fraction` is neither a float nor an int.

            If `fraction` is not in the interval [0, 1).

        """
        if not isinstance(fraction, (float, int)):
            raise TypeError("`fraction` must be float or integer")
        if not 0 <= fraction < 1:
            raise ValueError("`fraction` must be in the interval [0, 1)")

    def median_index(self, t):
        """
        Estimates the index of the Median of the kernel.
        This parameter is not mandatory for symmetrical kernels but it is
        required when asymmetrical kernels have to be aligned at their median.

        Parameters
        ----------
        t : pq.Quantity
            Vector with the interval on which the kernel is evaluated.

        Returns
        -------
        int
            Index of the estimated value of the kernel median.

        Notes
        -----
        The estimation is correct when the intervals in `t` are equidistant.
        """
        # FIXME: the current implementation is wrong in general
        return np.nonzero(self(t).cumsum() *
                          (t[len(t) - 1] - t[0]) / (len(t) - 1) >= 0.5)[
            0].min()
        kernel = self(t).magnitude
        return np.argsort(kernel)[len(kernel) // 2]

    def is_symmetric(self):
        """
        In the case of symmetric kernels, this method is overwritten in the
        class `SymmetricKernel`, where it returns True, hence leaving the
        here returned value False for the asymmetric kernels.

        Returns
        -------
        bool
            True in classes `SymmetricKernel`, `RectangularKernel`,
            `TriangularKernel`, `EpanechnikovLikeKernel`, `GaussianKernel`,
            and `LaplacianKernel`.
            False in classes `Kernel`, `ExponentialKernel`, and `AlphaKernel`.
        """
        return isinstance(self, SymmetricKernel)

    @property
    def min_cutoff(self):
        """
        Half width of the kernel.

        Returns
        -------
        float
            The returned value varies according to the kernel type.
        """
        raise NotImplementedError


class SymmetricKernel(Kernel):
    """
    Base class for symmetric kernels.
    """


class RectangularKernel(SymmetricKernel):
    r"""
    Class for rectangular kernels.

    .. math::
        K(t) = \left\{\begin{array}{ll} \frac{1}{2 \tau}, & |t| < \tau \\
        0, & |t| \geq \tau \end{array} \right.

    with :math:`\tau = \sqrt{3} \sigma` corresponding to the half width
    of the kernel.

    The parameter `invert` has no effect on symmetric kernels.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-3, 3, num=100) * pq.s
       kernel = kernels.RectangularKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("RectangularKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(3.0)
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        t_abs = np.abs(t.magnitude)
        tau = math.sqrt(3) * self.sigma.rescale(t_units).magnitude
        kernel = (t_abs < tau) * 1 / (2 * tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return np.sqrt(3.0) * self.sigma * fraction


class TriangularKernel(SymmetricKernel):
    r"""
    Class for triangular kernels.

    .. math::
        K(t) = \left\{ \begin{array}{ll} \frac{1}{\tau} (1
        - \frac{|t|}{\tau}), & |t| < \tau \\
         0, & |t| \geq \tau \end{array} \right.

    with :math:`\tau = \sqrt{6} \sigma` corresponding to the half width of
    the kernel.

    The parameter `invert` has no effect on symmetric kernels.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-3, 3, num=1000) * pq.s
       kernel = kernels.TriangularKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("TriangularKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(6.0)
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        t_abs = np.abs(t.magnitude)
        tau = math.sqrt(6) * self.sigma.rescale(t_units).magnitude
        kernel = (t_abs < tau) * 1 / tau * (1 - t_abs / tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return np.sqrt(6.0) * self.sigma * (1 - np.sqrt(1 - fraction))


class EpanechnikovLikeKernel(SymmetricKernel):
    r"""
    Class for Epanechnikov-like kernels.

    .. math::
        K(t) = \left\{\begin{array}{ll} (3 /(4 d)) (1 - (t / d)^2),
        & |t| < d \\
        0, & |t| \geq d \end{array} \right.

    with :math:`d = \sqrt{5} \sigma` being the half width of the kernel.

    The Epanechnikov kernel under full consideration of its axioms has a half
    width of :math:`\sqrt{5}`. Ignoring one axiom also the respective kernel
    with half width = 1 can be called Epanechnikov kernel [1]_.
    However, arbitrary width of this type of kernel is here preferred to be
    called 'Epanechnikov-like' kernel.

    The parameter `invert` has no effect on symmetric kernels.

    References
    ----------
    .. [1] https://de.wikipedia.org/wiki/Epanechnikov-Kern

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-3, 3, num=100) * pq.s
       kernel = kernels.EpanechnikovLikeKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("EpanechnikovLikeKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(5.0)
        return min_cutoff

    def _evaluate(self, t):
        sigma = self.sigma.rescale(t.units)
        return (3.0 / (4.0 * np.sqrt(5.0) * sigma)) * np.maximum(
            0.0,
            1 - (t / (np.sqrt(5.0) * sigma)).magnitude ** 2)

    def boundary_enclosing_area_fraction(self, fraction):
        r"""
        Refer to :func:`Kernel.boundary_enclosing_area_fraction` for the
        documentation.

        Notes
        -----
        For Epanechnikov-like kernels, integration of its density within
        the boundaries 0 and :math:`b`, and then solving for :math:`b` leads
        to the problem of finding the roots of a polynomial of third order.
        The implemented formulas are based on the solution of this problem
        given in [1]_, where the following 3 solutions are given:

        * :math:`u_1 = 1`, solution on negative side;
        * :math:`u_2 = \frac{-1 + i\sqrt{3}}{2}`, solution for larger
          values than zero crossing of the density;
        * :math:`u_3 = \frac{-1 - i\sqrt{3}}{2}`, solution for smaller
          values than zero crossing of the density.

        The solution :math:`u_3` is the relevant one for the problem at hand,
        since it involves only positive area contributions.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Cubic_function

        """
        self._check_fraction(fraction)
        # Python's complex-operator cannot handle quantities, hence the
        # following construction on quantities is necessary:
        Delta_0 = complex(1.0 / (5.0 * self.sigma.magnitude ** 2), 0) / \
            self.sigma.units ** 2
        Delta_1 = complex(2.0 * np.sqrt(5.0) * fraction /
                          (25.0 * self.sigma.magnitude ** 3), 0) / \
            self.sigma.units ** 3
        C = ((Delta_1 + (Delta_1 ** 2.0 - 4.0 * Delta_0 ** 3.0) ** (
            1.0 / 2.0)) /
            2.0) ** (1.0 / 3.0)
        u_3 = complex(-1.0 / 2.0, -np.sqrt(3.0) / 2.0)
        b = -5.0 * self.sigma ** 2 * (u_3 * C + Delta_0 / (u_3 * C))
        return b.real


class GaussianKernel(SymmetricKernel):
    r"""
    Class for gaussian kernels.

    .. math::
        K(t) = (\frac{1}{\sigma \sqrt{2 \pi}}) \exp(-\frac{t^2}{2 \sigma^2})

    with :math:`\sigma` being the standard deviation.

    The parameter `invert` has no effect on symmetric kernels.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-3, 3, num=100) * pq.s
       kernel = kernels.GaussianKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("GaussianKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        t = t.magnitude
        sigma = self.sigma.rescale(t_units).magnitude
        kernel = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(
            -0.5 * (t / sigma) ** 2)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return self.sigma * np.sqrt(2.0) * scipy.special.erfinv(fraction)


class LaplacianKernel(SymmetricKernel):
    r"""
    Class for laplacian kernels.

    .. math::
        K(t) = \frac{1}{2 \tau} \exp\left(-\left|\frac{t}{\tau}\right|\right)

    with :math:`\tau = \sigma / \sqrt{2}`.

    The parameter `invert` has no effect on symmetric kernels.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-3, 3, num=1000) * pq.s
       kernel = kernels.LaplacianKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("LaplacianKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        t = t.magnitude
        tau = self.sigma.rescale(t_units).magnitude / math.sqrt(2)
        kernel = 1 / (2 * tau) * np.exp(-np.abs(t / tau))
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction) / np.sqrt(2.0)


# Potential further symmetric kernels from Wiki Kernels (statistics):
# Quartic (biweight), Triweight, Tricube, Cosine, Logistics, Silverman


class ExponentialKernel(Kernel):
    r"""
    Class for exponential kernels.

    .. math::
        K(t) = \left\{\begin{array}{ll} (1 / \tau) \exp{(-t / \tau)},
        & t > 0 \\
        0, & t \leq 0 \end{array} \right.

    with :math:`\tau = \sigma`.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-1, 4, num=100) * pq.s
       kernel = kernels.ExponentialKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("ExponentialKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()


    """

    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        t = t.magnitude
        tau = self.sigma.rescale(t_units).magnitude
        if not self.invert:
            kernel = (t >= 0) * 1 / tau * np.exp(-t / tau)
        else:
            kernel = (t <= 0) * 1 / tau * np.exp(t / tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction)


class AlphaKernel(Kernel):
    r"""
    Class for alpha kernels.

    .. math::
        K(t) = \left\{\begin{array}{ll} (1 / \tau^2)
        \ t\ \exp{(-t / \tau)}, & t > 0 \\
        0, & t \leq 0 \end{array} \right.

    with :math:`\tau = \sigma / \sqrt{2}`.

    Examples
    --------

    .. plot::
       :include-source:

       from elephant import kernels
       import quantities as pq
       import numpy as np
       import matplotlib.pyplot as plt

       time_array = np.linspace(-1, 4, num=100) * pq.s
       kernel = kernels.AlphaKernel(sigma=1*pq.s)
       kernel_time = kernel(time_array)
       plt.plot(time_array, kernel_time)
       plt.title("AlphaKernel with sigma=1s")
       plt.xlabel("time, s")
       plt.ylabel("kernel, 1/s")
       plt.show()

    """

    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        t_units = t.units
        tau = self.sigma.rescale(t_units).magnitude / math.sqrt(2)
        t = t.magnitude
        if not self.invert:
            kernel = (t >= 0) * 1 / tau ** 2 * t * np.exp(-t / tau)
        else:
            kernel = (t <= 0) * 1 / tau ** 2 * (-t) * np.exp(t / tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        tau = self.sigma.magnitude / math.sqrt(2)

        def cdf(x):
            # CDF of the AlphaKernel, subtracted 'fraction'
            # evaluates the error of the root of cdf(x) = fraction
            return 1 - fraction - np.exp(-x / tau) * (x + tau) / tau

        # fraction is a good starting point for CDF approximation
        x_quantile = scipy.optimize.fsolve(cdf, x0=fraction)[0]
        x_quantile = pq.Quantity(x_quantile, units=1 / self.sigma.units)
        return x_quantile
