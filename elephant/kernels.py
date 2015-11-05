import copy
import quantities as pq
import numpy as np
import scipy.signal
import scipy.special
import tools

import numpy as np

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

    ## def __call__(self, t, sigma=None):
    def __call__(self, t):
        """ Evaluates the kernel at all time points in the array `t`.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        ## :param sigma: If not `None` this overwrites the sigma of the `Kernel` instance.
        ## :type sigma: Quantity scalar
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """

        ## if sigma is None:
        ##     sigma = self.sigma

        if  t.dimensionality.simplified != self.sigma.dimensionality.simplified:
            raise TypeError("The dimensionality of sigma and the input array to the callable kernel object "
                        "must be the same. Otherwise a normalization to 1 of the kernel cannot be performed.")

        return self._evaluate(t)

    def _evaluate(self, t):
        """ Evaluates the kernel.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        raise NotImplementedError()

    def boundary_enclosing_at_least(self, fraction):
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
        Calculates the index of the Median of the kernel.

        Remark: The following formula using retrieval of the sampling period from t only works for t with equidistant time intervals!
        """
        return np.nonzero(self(t).cumsum() * (t[len(t)-1] -t[0])/(len(t)-1) >= 0.5)[0].min()


    def is_symmetric(self):
        """ Should return `True` if the kernel is symmetric. """
        return False

    def summed_dist_matrix(self, vectors, presorted=False):
        """ Calculates the sum of all element pair distances for each
        pair of vectors.

        If :math:`(a_1, \\dots, a_n)` and :math:`(b_1, \\dots, b_m)` are the
        :math:`u`-th and :math:`v`-th vector from `vectors` and :math:`K` the
        kernel, the resulting entry in the 2D array will be :math:`D_{uv}
        = \\sum_{i=1}^{n} \\sum_{j=1}^{m} K(a_i - b_j)`.

        :param sequence vectors: A sequence of Quantity 1D to calculate the
            summed distances for each pair. The required units depend on the
            kernel. Usually it will be the inverse unit of the kernel size.
        :param bool presorted: Some optimized specializations of this function
            may need sorted vectors. Set `presorted` to `True` if you know that
            the passed vectors are already sorted to skip the sorting and thus
            increase performance.
        :rtype: Quantity 2D
        """

        D = np.empty((len(vectors), len(vectors)))
        if len(vectors) > 0:
            might_have_units = self(vectors[0])
            if hasattr(might_have_units, 'units'):
                D = D * might_have_units.units
            else:
                D = D * pq.dimensionless

        for i, j in np.ndindex(len(vectors), len(vectors)):
            D[i, j] = np.sum(self(
                (vectors[i] - np.atleast_2d(vectors[j]).T).flatten()))
        return D


class KernelFromFunction(Kernel):
    """ Creates a kernel from a function. Please note, that not all methods for
    such a kernel are implemented.
    """

    def __init__(self, kernel_func, sigma):
        Kernel.__init__(self, sigma)
        self._evaluate = kernel_func

    def is_symmetric(self):
        return False


def as_kernel_of_size(obj, sigma):
    """ Returns a kernel of desired size.

    :param obj: Either an existing kernel or a kernel function. A kernel
        function takes two arguments. First a `Quantity 1D` of evaluation time
        points and second a kernel size.
    :type obj: Kernel or func
    :param sigma: Desired standard deviation of the kernel.
    :type sigma: Quantity scalar
    :returns: A :class:`Kernel` with the desired kernel size. If `obj` is
        already a :class:`Kernel` instance, a shallow copy of this instance with
        changed kernel size will be returned. If `obj` is a function it will be
        wrapped in a :class:`Kernel` instance.
    :rtype: :class:`Kernel`
    """

    if isinstance(obj, Kernel):
        obj = copy.copy(obj)
        obj.sigma = sigma
    else:
        obj = KernelFromFunction(obj, sigma)
    return obj


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

    def summed_dist_matrix(self, vectors, presorted=False):
        D = np.empty((len(vectors), len(vectors)))
        if len(vectors) > 0:
            might_have_units = self(vectors[0])
            if hasattr(might_have_units, 'units'):
                D = D * might_have_units.units

        for i in xrange(len(vectors)):
            for j in xrange(i, len(vectors)):
                D[i, j] = D[j, i] = np.sum(self(
                    (vectors[i] - np.atleast_2d(vectors[j]).T).flatten()))
        return D


## TODO:
## The following started from a copy of class SymmetricKernel; to be adjusted
## In an ideal case here to be realized the kernel direction flip
class AsymmetricKernel(Kernel):
    """ Base class for asymmetric kernels. """

    def __init__(self, sigma, direction):
        """
        :param sigma: Standard deviation of the kernel.
        :type sigma: Quantity scalar
        """
        Kernel.__init__(self, sigma, direction)

    ## TODO: Necessary?
    def is_symmetric(self):
        return False

    ## TODO:
    ## def _evaluate(self, t):
    ##    if self.direction == -1:
    ##        \tilde(f)(t)=f(-t)

    ## TODO:
    ## Does summed_dist_matrix make sense in context of asymmetric kernels?
    def summed_dist_matrix(self, vectors, presorted=False):
        D = np.empty((len(vectors), len(vectors)))
        if len(vectors) > 0:
            might_have_units = self(vectors[0])
            if hasattr(might_have_units, 'units'):
                D = D * might_have_units.units

        for i in xrange(len(vectors)):
            for j in xrange(i, len(vectors)):
                D[i, j] = D[j, i] = np.sum(self(
                    (vectors[i] - np.atleast_2d(vectors[j]).T).flatten()))
        return D


class GaussianKernel(SymmetricKernel):
    """ :math:`K(t) = (\frac{1}{\sigma \sqrt{2 \pi}}) \exp(-\frac{t^2}{2 \sigma^2})`
    with :math:`\sigma` being the standard deviation.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma):
        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma.rescale(t.units))) * np.exp(
            -0.5 * (t / sigma.rescale(t.units)).magnitude ** 2)

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma)

    def boundary_enclosing_at_least(self, fraction):
        return self.sigma * np.sqrt(2.0) * \
            scipy.special.erfinv(fraction + scipy.special.erf(0.0))


class LaplacianKernel(SymmetricKernel):
    """ :math:`K(t) = \frac{1}{2 \tau} \exp(-|\frac{t}{\tau}|)`
    with :math:`\tau = \sigma / \sqrt{2}`.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma):
        return (1 / (np.sqrt(2.0) * sigma.rescale(t.units))) * np.exp(
            -(np.absolute(t) * np.sqrt(2.0) / sigma.rescale(t.units)).magnitude)

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma)

    def boundary_enclosing_at_least(self, fraction):
        ## return -self.kernel_size * np.log(1.0 - fraction)
        ## TODO:
        return -self.sigma * np.log(1.0 - fraction) / np.sqrt(2.0)

    def summed_dist_matrix(self, vectors, presorted=False):
        # This implementation is based on
        #
        # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
        # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
        # 48-58.
        #
        # Note that the cited paper contains some errors: In formula (9) the
        # left side of the equation should be divided by two and in the last
        # sum in this equation it should say `j|v_i >= u_i` instead of
        # `j|v_i > u_i`. Also, in equation (11) it should say `j|u_i >= v_i`
        # instead of `j|u_i > v_i`.
        #
        # Given N vectors with n entries on average the run-time complexity is
        # O(N^2 * n). O(N^2 + N * n) memory will be needed.

        if len(vectors) <= 0:
            return np.zeros((0, 0))

        if not presorted:
            vectors = [v.copy() for v in vectors]
            for v in vectors:
                v.sort()

        sizes = np.asarray([v.size for v in vectors])
        values = np.empty((len(vectors), max(1, sizes.max())))
        values.fill(np.nan)
        for i, v in enumerate(vectors):
            if v.size > 0:
                values[i, :v.size] = \
                    (v / self.kernel_size * pq.dimensionless).simplified

        exp_diffs = np.exp(values[:, :-1] - values[:, 1:])
        markage = np.zeros(values.shape)
        for u in xrange(len(vectors)):
            markage[u, 0] = 0
            for i in xrange(sizes[u] - 1):
                markage[u, i + 1] = (markage[u, i] + 1.0) * exp_diffs[u, i]

        # Same vector terms
        D = np.empty((len(vectors), len(vectors)))
        D[np.diag_indices_from(D)] = sizes + 2.0 * np.sum(markage, axis=1)

        # Cross vector terms
        for u in xrange(D.shape[0]):
            all_ks = np.searchsorted(values[u], values, 'left') - 1
            for v in xrange(u):
                js = np.searchsorted(values[v], values[u], 'right') - 1
                ks = all_ks[v]
                slice_j = np.s_[np.searchsorted(js, 0):sizes[u]]
                slice_k = np.s_[np.searchsorted(ks, 0):sizes[v]]
                D[u, v] = np.sum(
                    np.exp(values[v][js[slice_j]] - values[u][slice_j]) *
                    (1.0 + markage[v][js[slice_j]]))
                D[u, v] += np.sum(
                    np.exp(values[u][ks[slice_k]] - values[v][slice_k]) *
                    (1.0 + markage[u][ks[slice_k]]))
                D[v, u] = D[u, v]

        return D


class RectangularKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} \frac{1}{2 \tau}, & |t| < \tau \\
    0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{3} \sigma` corresponding to the half width of the kernel.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma):
        return (0.5 / (np.sqrt(3.0) * sigma.rescale(t.units))) * (np.absolute(t) < np.sqrt(3.0) * sigma.rescale(t.units))

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma)

    def boundary_enclosing_at_least(self, fraction):
        return np.sqrt(3.0) * self.sigma


class TriangularKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{ \begin{array}{ll} \frac{1}{\tau} (1
    - \frac{|t|}{\tau}), & |t| < \tau \\ 0, & |t| \geq \tau \end{array} \right`
    with :math:`\tau = \sqrt{6} \sigma` corresponding to the half width of the kernel.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma):
        return (1.0 / (np.sqrt(6.0) * sigma.rescale(t.units))) * np.maximum(
            0.0,
            (1.0 - (np.absolute(t) / (np.sqrt(6.0) * sigma.rescale(t.units))).magnitude))

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma)

    def boundary_enclosing_at_least(self, fraction):
        return np.sqrt(6.0) * self.sigma


class EpanechnikovLikeKernel(SymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (3 /(4 d)) (1 - (t / d)^2), & |t| < d \\
    0, & |t| \geq d \end{array} \right`
    with :math:`d = \sqrt{5} \sigma` being the half width of the kernel.

    The Epanechnikov kernel under full consideration of its axioms has a half width of :math:`sqrt{5}`.
    Ignoring one axiom also the respective kernel with half width = 1 can be called Epanechnikov kernel.
    ( https://de.wikipedia.org/wiki/Epanechnikov-Kern )
    However, arbitrary width of this type of kernel is here preferred to be called 'Epanechnikov-like' kernel.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma):
        return (3.0 / (4.0 * np.sqrt(5.0) * sigma.rescale(t.units))) * np.maximum(
            0.0,
            1 - (t / (np.sqrt(5.0) * sigma.rescale(t.units))).magnitude ** 2)

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma)

    def boundary_enclosing_at_least(self, fraction):
        return np.sqrt(5.0) * self.sigma


## Potential further symmetric kernels from Wiki Kernels (statistics):
## Quartic (biweight), Triweight, Tricube, Cosine, Logistics, Silverman


## class ExponentialKernel(Kernel):
class ExponentialKernel(AsymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (1 / \tau) \exp{-t / \tau}, & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma`.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma, direction):
        if direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: (1.0 / sigma.rescale(t.units)) * np.exp(
                        (-t / sigma.rescale(t.units)).magnitude)])
        elif direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: (1.0 / sigma.rescale(t.units)) * np.exp(
                        (t / sigma.rescale(t.units)).magnitude),
                    lambda t: 0])
        return kernel

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma, self.direction)

    def boundary_enclosing_at_least(self, fraction):
        return -self.sigma * np.log(1.0 - fraction)


## class AlphaKernel(Kernel):
class AlphaKernel(AsymmetricKernel):
    """ :math:`K(t) = \left\{\begin{array}{ll} (1 / (\tau)^2) t \exp{-t / \tau}, & t > 0 \\
    0, & t \leq 0 \end{array} \right`
    with :math:`\tau = \sigma / \sqrt{2}`.
    """

    def __init__(self, sigma=1.0 * pq.s, direction=1):
        Kernel.__init__(self, sigma, direction)

    @staticmethod
    def evaluate(t, sigma, direction):
        if direction == 1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: 0,
                    lambda t: (2.0 / (sigma.rescale(t.units))**2) * t * np.exp((-t * np.sqrt(2.0) / sigma.rescale(t.units)).magnitude)])
        elif direction == -1:
            kernel = np.piecewise(
                t, [t < 0, t >= 0], [
                    lambda t: (2.0 / (sigma.rescale(t.units))**2) * (-1) * t * np.exp((t * np.sqrt(2.0) / sigma.rescale(t.units)).magnitude),
                    lambda t: 0 ])
        return kernel

    def _evaluate(self, t):
        return self.evaluate(t, self.sigma, self.direction)

    ## TODO:
    def boundary_enclosing_at_least(self, fraction):
        return -self.sigma * np.log(1.0 - fraction)


def discretize_kernel(
        kernel, sampling_rate, area_fraction=default_kernel_area_fraction,
        num_bins=None, ensure_unit_area=False):
    """ Discretizes a kernel.

    :param kernel: The kernel or kernel function. If a kernel function is used
        it should take exactly one 1-D array as argument.
    :type kernel: :class:`Kernel` or function
    :param sampling_rate: Sampling rate for the discretization. The unit will
        typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param float area_fraction: Fraction between 0 and 1 (exclusive)
        of the integral of the kernel which will be at least covered by the
        discretization. Will be ignored if `num_bins` is not `None`. If
        `area_fraction` is used, the kernel has to provide a method
        :meth:`boundary_enclosing_at_least` (see
        :meth:`.Kernel.boundary_enclosing_at_least`).
    :param int num_bins: Number of bins to use for the discretization.
    :param bool ensure_unit_area: If `True`, the area of the discretized
        kernel will be normalized to 1.0.
    :rtype: Quantity 1D
    """

    t_step = 1.0 / sampling_rate

    if num_bins is not None:
        start = -num_bins // 2
        stop = num_bins // 2
    elif area_fraction is not None:
        boundary = kernel.boundary_enclosing_at_least(area_fraction)
        if hasattr(boundary, 'rescale'):
            boundary = boundary.rescale(t_step.units)
        start = np.ceil(-boundary / t_step)
        stop = np.floor(boundary / t_step) + 1
    else:
        raise ValueError(
            "One of area_fraction and num_bins must not be None.")

    k = kernel(np.arange(start, stop) * t_step)
    if ensure_unit_area:
        k /= np.sum(k) * t_step
    return k


def smooth(
        binned, kernel, sampling_rate, mode='same',
        **kernel_discretization_params):
    """ Smoothes a binned representation (e.g. of a spike train) by convolving
    with a kernel.

    :param binned: Bin array to smooth.
    :type binned: 1-D array
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to discretize the
        kernel. It should be equal to the sampling rate used to obtain `binned`.
        The unit will typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array of the same size as
          `binned`
        * 'full': Returns an array with a bin for each shift where `binned` and
          the discretized kernel overlap by at least one bin.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and `binned` completely overlap.

        See also `numpy.convolve
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html>`_.
    :type mode: {'same', 'full', 'valid'}
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :func:`.discretize_kernel`.
    :returns: The smoothed representation of `binned`.
    :rtype: Quantity 1D
    """
    k = discretize_kernel(
        kernel, sampling_rate=sampling_rate, **kernel_discretization_params)
    return scipy.signal.convolve(binned, k, mode) * k.units


def st_convolve(
        train, kernel, sampling_rate, mode='same', binning_params={},
        kernel_discretization_params={}):
    """ Convolves a :class:`neo.core.SpikeTrain` with a kernel.

    :param train: Spike train to convolve.
    :type train: :class:`neo.core.SpikeTrain`
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train. The unit will typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array covering the whole
          duration of the spike train `train`.
        * 'full': Returns an array with additional discretization bins in the
          beginning and end so that for each spike the whole discretized
          kernel is included.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and spike train completely overlap.

        See also :func:`scipy.signal.convolve`.
    :type mode: {'same', 'full', 'valid'}
    :param dict binning_params: Additional discretization arguments which will
        be passed to :func:`.tools.bin_spike_trains`.
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :func:`.discretize_kernel`.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (Quantity 1D, Quantity 1D with the inverse units of `sampling_rate`)
    """

    binned, bins = tools.bin_spike_trains(
        {0: [train]}, sampling_rate, **binning_params)
    binned = binned[0][0]
    #sampling_rate = binned.size / (bins[-1] - bins[0])
    result = smooth(
        binned, kernel, sampling_rate, mode, **kernel_discretization_params)

    assert (result.size - binned.size) % 2 == 0
    num_additional_bins = (result.size - binned.size) // 2
    bins = np.linspace(
        bins[0] - num_additional_bins / sampling_rate,
        bins[-1] + num_additional_bins / sampling_rate,
        result.size + 1)

    return result, bins
