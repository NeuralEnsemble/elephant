# -*- coding: utf-8 -*-
"""
Definition of a hierarchy of classes for kernel functions to be used
in convolution, e.g., for data smoothing (low pass filtering) or
firing rate estimation.


Base kernel classes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: kernels/

    Kernel
    SymmetricKernel

Symmetric kernels
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: kernels/

    RectangularKernel
    TriangularKernel
    EpanechnikovLikeKernel
    GaussianKernel
    LaplacianKernel

Asymmetric kernels
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: kernels/

    ExponentialKernel
    AlphaKernel


Examples
--------
>>> import quantities as pq
>>> kernel1 = GaussianKernel(sigma=100*pq.ms)
>>> kernel2 = ExponentialKernel(sigma=8*pq.mm, invert=True)

:copyright: Copyright 2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import quantities as pq
import scipy.special


class Kernel(object):
    r"""
    This is the base class for commonly used kernels.

    **General definition of kernel:**

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
    postynaptic current/potentials in a linear (current-based) model.

    Parameters
    ----------
    sigma : pq.Quantity
        Standard deviation of the kernel.
    invert: bool, optional
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

        if not (isinstance(sigma, pq.Quantity)):
            raise TypeError("sigma must be a quantity!")

        if sigma.magnitude < 0:
            raise ValueError("sigma cannot be negative!")

        if not isinstance(invert, bool):
            raise ValueError("invert must be bool!")

        self.sigma = sigma
        self.invert = invert

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

        Parameter
        ---------
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
        self._check_fraction(fraction)
        sigma_division = 500            # arbitrary choice
        interval = self.sigma / sigma_division
        self._sigma_scaled = self.sigma
        area = 0
        counter = 0
        while area < fraction:
            area += (self._evaluate((counter + 1) * interval) +
                     self._evaluate(counter * interval)) * interval / 2
            area += (self._evaluate(-1 * (counter + 1) * interval) +
                     self._evaluate(-1 * counter * interval)) * interval / 2
            counter += 1
            if(counter > 250000):
                raise ValueError("fraction was chosen too close to one such "
                                 "that in combination with integral "
                                 "approximation errors the calculation of a "
                                 "boundary was not possible.")
        return counter * interval

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
            raise TypeError("`fraction` must be float or integer!")
        if not 0 <= fraction < 1:
            raise ValueError("`fraction` must be in the interval [0, 1)!")

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
        The formula in this method using retrieval of the sampling interval
        from `t` only works for `t` with equidistant intervals!
        The formula calculates the Median slightly wrong by the potentially
        ignored probability in the distribution corresponding to lower values
        than the minimum in the array `t`.

        """
        return np.nonzero(self(t).cumsum() *
                          (t[len(t) - 1] - t[0]) / (len(t) - 1) >= 0.5)[0].min()

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
        return False

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

    def is_symmetric(self):
        return True


class RectangularKernel(SymmetricKernel):
    r"""
    Class for rectangular kernels.

    .. math::
        K(t) = \left\{\begin{array}{ll} \frac{1}{2 \tau}, & |t| < \tau \\\
        0, & |t| \geq \tau \end{array} \right.

    with :math:`\tau = \sqrt{3} \sigma` corresponding to the half width
    of the kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `invert` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(3.0)
        return min_cutoff

    def _evaluate(self, t):
        return (0.5 / (np.sqrt(3.0) * self._sigma_scaled)) * \
               (np.absolute(t) < np.sqrt(3.0) * self._sigma_scaled)

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return np.sqrt(3.0) * self.sigma * fraction


class TriangularKernel(SymmetricKernel):
    """
    Class for triangular kernels.

    .. math::
        K(t) = \\left\\{ \\begin{array}{ll} \\frac{1}{\\tau} (1
        - \\frac{|t|}{\\tau}), & |t| < \\tau \\\\
         0, & |t| \\geq \\tau \\end{array} \\right.

    with :math:`\\tau = \\sqrt{6} \\sigma` corresponding to the half width of
    the kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `invert` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(6.0)
        return min_cutoff

    def _evaluate(self, t):
        return (1.0 / (np.sqrt(6.0) * self._sigma_scaled)) * np.maximum(
            0.0,
            (1.0 - (np.absolute(t) /
                    (np.sqrt(6.0) * self._sigma_scaled)).magnitude))

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return np.sqrt(6.0) * self.sigma * (1 - np.sqrt(1 - fraction))


class EpanechnikovLikeKernel(SymmetricKernel):
    """
    Class for epanechnikov-like kernels.

    .. math::
        K(t) = \\left\\{\\begin{array}{ll} (3 /(4 d)) (1 - (t / d)^2),
        & |t| < d \\\\
        0, & |t| \\geq d \\end{array} \\right.

    with :math:`d = \\sqrt{5} \\sigma` being the half width of the kernel.

    The Epanechnikov kernel under full consideration of its axioms has a half
    width of :math:`\\sqrt{5}`. Ignoring one axiom also the respective kernel
    with half width = 1 can be called Epanechnikov kernel [1]_.
    However, arbitrary width of this type of kernel is here preferred to be
    called 'Epanechnikov-like' kernel.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `invert` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.

    Attributes
    ----------
    min_cutoff : float

    References
    ----------
    .. [1] https://de.wikipedia.org/wiki/Epanechnikov-Kern

    """
    @property
    def min_cutoff(self):
        min_cutoff = np.sqrt(5.0)
        return min_cutoff

    def _evaluate(self, t):
        return (3.0 / (4.0 * np.sqrt(5.0) * self._sigma_scaled)) * np.maximum(
            0.0,
            1 - (t / (np.sqrt(5.0) * self._sigma_scaled)).magnitude ** 2)

    def boundary_enclosing_area_fraction(self, fraction):
        """
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
        * :math:`u_2 = \\frac{-1 + i\\sqrt{3}}{2}`, solution for larger
          values than zero crossing of the density;
        * :math:`u_3 = \\frac{-1 - i\\sqrt{3}}{2}`, solution for smaller
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
        Delta_0 = complex(1.0 / (5.0 * self.sigma.magnitude**2), 0) / \
            self.sigma.units**2
        Delta_1 = complex(2.0 * np.sqrt(5.0) * fraction /
                          (25.0 * self.sigma.magnitude**3), 0) / \
            self.sigma.units**3
        C = ((Delta_1 + (Delta_1**2.0 - 4.0 * Delta_0**3.0)**(1.0 / 2.0)) /
             2.0)**(1.0 / 3.0)
        u_3 = complex(-1.0 / 2.0, -np.sqrt(3.0) / 2.0)
        b = -5.0 * self.sigma**2 * (u_3 * C + Delta_0 / (u_3 * C))
        return b.real


class GaussianKernel(SymmetricKernel):
    """
    Class for gaussian kernels.

    .. math::
        K(t) = (\\frac{1}{\\sigma \\sqrt{2 \\pi}})
        \\exp(-\\frac{t^2}{2 \\sigma^2})

    with :math:`\\sigma` being the standard deviation.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `invert` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        return (1.0 / (np.sqrt(2.0 * np.pi) * self._sigma_scaled)) * np.exp(
            -0.5 * (t / self._sigma_scaled).magnitude ** 2)

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return self.sigma * np.sqrt(2.0) * scipy.special.erfinv(fraction)


class LaplacianKernel(SymmetricKernel):
    """
    Class for laplacian kernels.

    .. math::
        K(t) = \\frac{1}{2 \\tau} \\exp(-|\\frac{t}{\\tau}|)

    with :math:`\\tau = \\sigma / \\sqrt{2}`.

    Besides the standard deviation `sigma`, for consistency of interfaces the
    parameter `invert` needed for asymmetric kernels also exists without
    having any effect in the case of symmetric kernels.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        return (1 / (np.sqrt(2.0) * self._sigma_scaled)) * np.exp(
            -(np.absolute(t) * np.sqrt(2.0) / self._sigma_scaled).magnitude)

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction) / np.sqrt(2.0)


# Potential further symmetric kernels from Wiki Kernels (statistics):
# Quartic (biweight), Triweight, Tricube, Cosine, Logistics, Silverman


class ExponentialKernel(Kernel):
    """
    Class for exponential kernels.

    .. math::
        K(t) = \\left\\{\\begin{array}{ll} (1 / \\tau) \\exp{(-t / \\tau)},
        & t > 0 \\\\
        0, & t \\leq 0 \\end{array} \\right.

    with :math:`\\tau = \\sigma`.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        if not self.invert:
            kernel = (t >= 0) * (1. / self._sigma_scaled.magnitude) *\
                np.exp((-t / self._sigma_scaled).magnitude) / t.units
        elif self.invert:
            kernel = (t <= 0) * (1. / self._sigma_scaled.magnitude) *\
                np.exp((t / self._sigma_scaled).magnitude) / t.units
        return kernel

    def boundary_enclosing_area_fraction(self, fraction):
        self._check_fraction(fraction)
        return -self.sigma * np.log(1.0 - fraction)


class AlphaKernel(Kernel):
    """
    Class for alpha kernels.

    .. math::
        K(t) = \\left\\{\\begin{array}{ll} (1 / \\tau^2)
        \\ t\\ \\exp{(-t / \\tau)}, & t > 0 \\\\
        0, & t \\leq 0 \\end{array} \\right.

    with :math:`\\tau = \\sigma / \\sqrt{2}`.

    For the alpha kernel an analytical expression for the boundary of the
    integral as a function of the area under the alpha kernel function
    cannot be given. Hence in this case the value of the boundary is
    determined by kernel-approximating numerical integration, inherited
    from the `Kernel` class.

    Attributes
    ----------
    min_cutoff : float

    """
    @property
    def min_cutoff(self):
        min_cutoff = 3.0
        return min_cutoff

    def _evaluate(self, t):
        if not self.invert:
            kernel = (t >= 0) * 2. * (t / self._sigma_scaled**2).magnitude *\
                np.exp((
                    -t * np.sqrt(2.) / self._sigma_scaled).magnitude) / t.units
        elif self.invert:
            kernel = (t <= 0) * -2. * (t / self._sigma_scaled**2).magnitude *\
                np.exp((
                    t * np.sqrt(2.) / self._sigma_scaled).magnitude) / t.units
        return kernel
