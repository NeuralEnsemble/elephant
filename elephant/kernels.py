# -*- coding: utf-8 -*-
"""
Definition of a hierarchy of classes for kernel functions to be used
in convolution, e.g., for data smoothing (low pass filtering) or
firing rate estimation.


Symmetric kernels
*****************

.. autosummary::
    :toctree: _toctree/kernels/

    RectangularKernel
    TriangularKernel
    EpanechnikovLikeKernel
    GaussianKernel
    LaplacianKernel

Asymmetric kernels
******************

.. autosummary::
    :toctree: _toctree/kernels/

    ExponentialKernel
    AlphaKernel


Examples
********

Example 1. Gaussian kernel

>>> import neo
>>> import quantities as pq
>>> from elephant import kernels
>>> kernel = kernels.GaussianKernel(sigma=300 * pq.ms)
>>> kernel
GaussianKernel(sigma=300.0 ms, invert=False)
>>> spiketrain = neo.SpikeTrain([-1, 0, 1], t_start=-1, t_stop=1, units='s')
>>> kernel_pdf = kernel(spiketrain)
>>> kernel_pdf
array([0.00514093, 1.3298076 , 0.00514093]) * 1/s

Cumulative Distribution Function

>>> kernel.cdf(0 * pq.s)
0.5
>>> kernel.cdf(1 * pq.s)
0.9995709396668032

Inverse Cumulative Distribution Function

>>> kernel.icdf(0.5)
array(0.) * ms
>>> kernel.icdf(0.9)
array(384.46546966) * ms

Example 2. Alpha kernel

>>> kernel = kernels.AlphaKernel(sigma=1 * pq.s)
>>> kernel(spiketrain)
array([-0.        ,  0.        ,  0.48623347]) * 1/s
>>> kernel.cdf(0 * pq.s)
0.0
>>> kernel.icdf(0.5)
array(1.18677054) * s

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import math

import numpy as np
import quantities as pq
import scipy.optimize
import scipy.special
import scipy.stats

from elephant.utils import deprecated_alias

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
        Default: False

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

    @deprecated_alias(t='times')
    def __call__(self, times):
        """
        Evaluates the kernel at all points in the array `times`.

        Parameters
        ----------
        times : pq.Quantity
            A vector with time intervals on which the kernel is evaluated.

        Returns
        -------
        pq.Quantity
            Vector with the result of the kernel evaluations.

        Raises
        ------
        TypeError
            If `times` is not `pq.Quantity`.

            If the dimensionality of `times` and :attr:`sigma` are different.

        """
        self._check_time_input(times)
        return self._evaluate(times)

    def _evaluate(self, times):
        """
        Evaluates the kernel Probability Density Function, PDF.

        Parameters
        ----------
        times : pq.Quantity
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

    def _check_fraction(self, fraction):
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
        if isinstance(self, (TriangularKernel, RectangularKernel)):
            valid = 0 <= fraction <= 1
            bracket = ']'
        else:
            valid = 0 <= fraction < 1
            bracket = ')'
        if not valid:
            raise ValueError("`fraction` must be in the interval "
                             "[0, 1{}".format(bracket))

    def _check_time_input(self, t):
        if not isinstance(t, pq.Quantity):
            raise TypeError("The argument 't' of the kernel callable must be "
                            "of type Quantity")

        if t.dimensionality.simplified != self.sigma.dimensionality.simplified:
            raise TypeError("The dimensionality of sigma and the input array "
                            "to the callable kernel object must be the same. "
                            "Otherwise a normalization to 1 of the kernel "
                            "cannot be performed.")

    @deprecated_alias(t='time')
    def cdf(self, time):
        r"""
        Cumulative Distribution Function, CDF.

        Parameters
        ----------
        time : pq.Quantity
            The input time scalar.

        Returns
        -------
        float
            CDF at `time`.

        """
        raise NotImplementedError

    def icdf(self, fraction):
        r"""
        Inverse Cumulative Distribution Function, ICDF, also known as a
        quantile.

        Parameters
        ----------
        fraction : float
            The fraction of CDF to compute the quantile from.

        Returns
        -------
        pq.Quantity
            The time scalar `times` such that `CDF(t) = fraction`.

        """
        raise NotImplementedError

    @deprecated_alias(t='times')
    def median_index(self, times):
        r"""
        Estimates the index of the Median of the kernel.

        We define the Median index :math:`i` of a kernel as:

        .. math::
            t_i = \text{ICDF}\left( \frac{\text{CDF}(t_0) +
            \text{CDF}(t_{N-1})}{2} \right)

        where :math:`t_0` and :math:`t_{N-1}` are the first and last entries of
        the input array, CDF and ICDF stand for Cumulative Distribution
        Function and its Inverse, respectively.

        This function is not mandatory for symmetrical kernels but it is
        required when asymmetrical kernels have to be aligned at their median.

        Parameters
        ----------
        times : pq.Quantity
            Vector with the interval on which the kernel is evaluated.

        Returns
        -------
        int
            Index of the estimated value of the kernel median.

        Raises
        ------
        TypeError
            If the input array is not a time pq.Quantity array.

        ValueError
            If the input array is empty.
            If the input array is not sorted.

        See Also
        --------
        Kernel.cdf : cumulative distribution function
        Kernel.icdf : inverse cumulative distribution function

        """
        self._check_time_input(times)
        if len(times) == 0:
            raise ValueError("The input time array is empty.")
        if len(times) <= 2:
            # either left or right; choose left
            return 0
        is_sorted = (np.diff(times.magnitude) >= 0).all()
        if not is_sorted:
            raise ValueError("The input time array must be sorted (in "
                             "ascending order).")
        cdf_mean = 0.5 * (self.cdf(times[0]) + self.cdf(times[-1]))
        if cdf_mean == 0.:
            # any index of the kernel non-support is valid; choose median
            return len(times) // 2
        icdf = self.icdf(fraction=cdf_mean)
        icdf = icdf.rescale(times.units).magnitude
        # icdf is guaranteed to be in (t_start, t_end) interval
        median_index = np.nonzero(times.magnitude >= icdf)[0][0]
        return median_index

    def is_symmetric(self):
        r"""
        True for symmetric kernels and False otherwise (asymmetric kernels).

        A kernel is symmetric if its PDF is symmetric w.r.t. time:

        .. math::
            \text{pdf}(-t) = \text{pdf}(t)

        Returns
        -------
        bool
            Whether the kernels is symmetric or not.
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

    def _evaluate(self, times):
        t_units = times.units
        t_abs = np.abs(times.magnitude)
        tau = math.sqrt(3) * self.sigma.rescale(t_units).magnitude
        kernel = (t_abs < tau) * 1 / (2 * tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = math.sqrt(3) * self.sigma.rescale(time.units).magnitude
        time = np.clip(time.magnitude, a_min=-tau, a_max=tau)
        cdf = (time + tau) / (2 * tau)
        return cdf

    def icdf(self, fraction):
        self._check_fraction(fraction)
        tau = math.sqrt(3) * self.sigma
        icdf = tau * (2 * fraction - 1)
        return icdf

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

    def _evaluate(self, times):
        tau = math.sqrt(6) * self.sigma.rescale(times.units).magnitude
        kernel = scipy.stats.triang.pdf(times.magnitude, c=0.5, loc=-tau,
                                        scale=2 * tau)
        kernel = pq.Quantity(kernel, units=1 / times.units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = math.sqrt(6) * self.sigma.rescale(time.units).magnitude
        cdf = scipy.stats.triang.cdf(time.magnitude, c=0.5, loc=-tau,
                                     scale=2 * tau)
        return cdf

    def icdf(self, fraction):
        self._check_fraction(fraction)
        tau = math.sqrt(6) * self.sigma.magnitude
        icdf = scipy.stats.triang.ppf(fraction, c=0.5, loc=-tau, scale=2 * tau)
        return icdf * self.sigma.units

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
    with half width = 1 can be called `Epanechnikov kernel
    <https://de.wikipedia.org/wiki/Epanechnikov-Kern>`_.
    However, arbitrary width of this type of kernel is here preferred to be
    called 'Epanechnikov-like' kernel.

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

    def _evaluate(self, times):
        tau = math.sqrt(5) * self.sigma.rescale(times.units).magnitude
        t_div_tau = np.clip(times.magnitude / tau, a_min=-1, a_max=1)
        kernel = 3. / (4. * tau) * np.maximum(0., 1 - t_div_tau ** 2)
        kernel = pq.Quantity(kernel, units=1 / times.units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = math.sqrt(5) * self.sigma.rescale(time.units).magnitude
        t_div_tau = np.clip(time.magnitude / tau, a_min=-1, a_max=1)
        cdf = 3. / 4 * (t_div_tau - t_div_tau ** 3 / 3.) + 0.5
        return cdf

    def icdf(self, fraction):
        self._check_fraction(fraction)
        # CDF(t) = -1/4 t^3 + 3/4 t + 1/2
        coefs = [-1. / 4, 0, 3. / 4, 0.5 - fraction]
        roots = np.roots(coefs)
        icdf = next(root for root in roots if -1 <= root <= 1)
        tau = math.sqrt(5) * self.sigma
        return icdf * tau

    def boundary_enclosing_area_fraction(self, fraction):
        r"""
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

        Notes
        -----
        For Epanechnikov-like kernels, integration of its density within
        the boundaries 0 and :math:`b`, and then solving for :math:`b` leads
        to the problem of finding the roots of a polynomial of third order.
        The implemented formulas are based on the solution of a
        `cubic function <https://en.wikipedia.org/wiki/Cubic_function>`_,
        where the following 3 solutions are given:

        * :math:`u_1 = 1`, solution on negative side;
        * :math:`u_2 = \frac{-1 + i\sqrt{3}}{2}`, solution for larger
          values than zero crossing of the density;
        * :math:`u_3 = \frac{-1 - i\sqrt{3}}{2}`, solution for smaller
          values than zero crossing of the density.

        The solution :math:`u_3` is the relevant one for the problem at hand,
        since it involves only positive area contributions.
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

    def _evaluate(self, times):
        sigma = self.sigma.rescale(times.units).magnitude
        kernel = scipy.stats.norm.pdf(times.magnitude, loc=0, scale=sigma)
        kernel = pq.Quantity(kernel, units=1 / times.units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        sigma = self.sigma.rescale(time.units).magnitude
        cdf = scipy.stats.norm.cdf(time, loc=0, scale=sigma)
        return cdf

    def icdf(self, fraction):
        self._check_fraction(fraction)
        icdf = scipy.stats.norm.ppf(fraction, loc=0,
                                    scale=self.sigma.magnitude)
        return icdf * self.sigma.units

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

    def _evaluate(self, times):
        tau = self.sigma.rescale(times.units).magnitude / math.sqrt(2)
        kernel = scipy.stats.laplace.pdf(times.magnitude, loc=0, scale=tau)
        kernel = pq.Quantity(kernel, units=1 / times.units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = self.sigma.rescale(time.units).magnitude / math.sqrt(2)
        cdf = scipy.stats.laplace.cdf(time.magnitude, loc=0, scale=tau)
        return cdf

    def icdf(self, fraction):
        self._check_fraction(fraction)
        tau = self.sigma.magnitude / math.sqrt(2)
        icdf = scipy.stats.laplace.ppf(fraction, loc=0, scale=tau)
        return icdf * self.sigma.units

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
        & t \geq 0 \\
        0, & t < 0 \end{array} \right.

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

    def _evaluate(self, times):
        tau = self.sigma.rescale(times.units).magnitude
        if self.invert:
            times = -times
        kernel = scipy.stats.expon.pdf(times.magnitude, loc=0, scale=tau)
        kernel = pq.Quantity(kernel, units=1 / times.units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = self.sigma.rescale(time.units).magnitude
        time = time.magnitude
        if self.invert:
            time = np.minimum(time, 0)
            return np.exp(time / tau)
        time = np.maximum(time, 0)
        return 1. - np.exp(-time / tau)

    def icdf(self, fraction):
        self._check_fraction(fraction)
        if self.invert:
            return self.sigma * np.log(fraction)
        return -self.sigma * np.log(1.0 - fraction)

    def boundary_enclosing_area_fraction(self, fraction):
        # the boundary b, which encloses a 'fraction' of CDF in [-b, b] range,
        # does not depend on the invert, if the kernel is cut at zero.
        # It's easier to compute 'b' for a kernel that has not been inverted.
        kernel = self.__class__(sigma=self.sigma, invert=False)
        return kernel.icdf(fraction)


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

    def _evaluate(self, times):
        t_units = times.units
        tau = self.sigma.rescale(t_units).magnitude / math.sqrt(2)
        times = times.magnitude
        if self.invert:
            times = -times
        kernel = (times >= 0) * 1 / tau ** 2 * times * np.exp(-times / tau)
        kernel = pq.Quantity(kernel, units=1 / t_units)
        return kernel

    @deprecated_alias(t='time')
    def cdf(self, time):
        self._check_time_input(time)
        tau = self.sigma.rescale(time.units).magnitude / math.sqrt(2)
        cdf = self._cdf_stripped(time.magnitude, tau)
        return cdf

    def _cdf_stripped(self, t, tau):
        # CDF without time units
        if self.invert:
            t = np.minimum(t, 0)
            return np.exp(t / tau) * (tau - t) / tau
        t = np.maximum(t, 0)
        return 1 - np.exp(-t / tau) * (t + tau) / tau

    def icdf(self, fraction):
        self._check_fraction(fraction)
        tau = self.sigma.magnitude / math.sqrt(2)

        def cdf(x):
            # CDF fof the AlphaKernel, subtracted 'fraction'
            # evaluates the error of the root of cdf(x) = fraction
            return self._cdf_stripped(x, tau) - fraction

        # fraction is a good starting point for CDF approximation
        x0 = fraction if not self.invert else fraction - 1
        x_quantile = scipy.optimize.fsolve(cdf, x0=x0, xtol=1e-7)[0]
        x_quantile = pq.Quantity(x_quantile, units=self.sigma.units)
        return x_quantile

    def boundary_enclosing_area_fraction(self, fraction):
        # the boundary b, which encloses a 'fraction' of CDF in [-b, b] range,
        # does not depend on the invert, if the kernel is cut at zero.
        # It's easier to compute 'b' for a kernel that has not been inverted.
        kernel = self.__class__(sigma=self.sigma, invert=False)
        return kernel.icdf(fraction)
