import neo
import quantities as pq
import numpy as np
from elephant.statistics import isi
from .base import Analysis
from .helpers import _check_list_size_and_types, _check_ndarray_size_and_types, _check_quantities_size_and_unit


class SingleISI(Analysis):
    """
    Computes the interspike interval for a single spike train.

    Parameters
    ----------
    spiketrain: neo.SpikeTrain or list of int/float or Quantity array or 1D NumPy ndarray

    params: dict
        May contain the following input parameters:

            1. 'time_unit': time Quantity
                Required if input is not Quantity array or neo.SpikeTrain
                Optional if input is Quantity array or neo.SpikeTrain. If informed, data will be rescaled to that unit.

    Attributes
    ----------
    intervals
    time_unit

    Methods
    -------
    median
    """

    _name = "Interspike intervals (for a single spike train)"
    _description = "elephant.statistics.isi"

    _required_types = {'time_unit': (pq.Quantity,)}

    _isi = None
    _data = None

    _ALLOWED_NDARRAY_TYPES = (np.int, np.float)
    _ALLOWED_LIST_TYPES = (int, float)

    def __init__(self, spiketrain, params={}, **kwargs):
        super().__init__(spiketrain, params=params, **kwargs)

    def _validate_data(self, spiketrain, **kwargs):
        """
        Validates the given data to check if is one of the accepted formats.
        Checks the contents of each format.

        Parameters
        ----------
        spiketrain: list of int/float or Quantity array or NumPy ndarray or neo.SpikeTrain
            If passing list or NumPy ndarray with spike times, the 'time_unit' parameter must be passed.
            If passing Quantity array, it must be a time quantity.
            NumPy ndarrays must be 1D.

        Raises
        ------
        ValueError
            If input is invalid: empty list/array, list/array with non-allowed types,
            Quantity array with non-time quantity.

        NameError
            If passing list/array of spike times without specifying 'time_unit' input parameter

        """
        if isinstance(spiketrain, list):
            # List of spike times.
            # Check if list is not empty and the items are of the allowed types
            _check_list_size_and_types(spiketrain, self._ALLOWED_LIST_TYPES)

            # `time_unit` input parameter is required
            if self.get_input_parameter('time_unit') is None:
                raise NameError("Input parameter 'time_unit' must be passed when using spike times as input")

        elif isinstance(spiketrain, pq.Quantity):
            # Quantities array.
            # It must not be empty and must be a 1-D time quantity array
            _check_quantities_size_and_unit(spiketrain, ndim=1, unit=pq.s)

        elif isinstance(spiketrain, np.ndarray):
            # NumPy array.
            # It must not be empty and must be a 1-D array of ints or floats.
            _check_ndarray_size_and_types(spiketrain, ndim=1, dtypes=self._ALLOWED_NDARRAY_TYPES)

            # Time unit input parameter is required
            if self.get_input_parameter('time_unit') is None:
                raise NameError("Input parameter 'time_unit' must be passed when using spike times as input")
        else:
            # Must be a neo.SpikeTrain object
            if not isinstance(spiketrain, neo.SpikeTrain):
                raise ValueError(f"Input must be neo.SpikeTrain object, not {type(spiketrain)}")

    def _validate_parameters(self):
        """
        Validates the following input parameters:
            1. 'time_unit': must be a time Quantity

        Raises
        ------
        ValueError
            If any input parameter is invalid
        """
        time_unit = self.get_input_parameter('time_unit')
        if time_unit is not None:
            if not time_unit.simplified.dimensionality == pq.s.dimensionality:
                raise ValueError("Parameter 'time_unit' must be a time Quantity. Dimensionality"
                                 f"'{time_unit.simplified.dimensionality}' was given")

    def _process(self, spiketrain, **kwargs):
        """
        Calculates the ISI for the spike train.
        If `spiketrain` is a list or NumPy ndarray, times will be converted to Quantity array using `time_unit` input
            parameter.
        If `spiketrain` is neo.SpikeTrain object or Quantity array, and `time_unit` is specified and different than the
            data, the `spiketrain` will be rescaled.

        Parameters
        ----------
        spiketrain: list of int/float or Quantity array or NumPy ndarray or neo.SpikeTrain
            The times in the spike train

        kwargs: dict
            Additional data
        """
        self._data = spiketrain

        self._isi = isi(self._data)

        time_unit = self.get_input_parameter('time_unit')
        if time_unit is not None:
            if self._isi.units != time_unit:
                self._isi = self._isi.rescale(time_unit)

    @property
    def intervals(self):
        """
        Interspike intervals for the source spike train.

        Returns
        -------
        Quantity array
        """
        return self._isi

    @property
    def time_unit(self):
        """
        The unit of the intervals (time dimension)

        Returns
        -------
        pq.Quantity
        """
        return self.intervals.units

    # TODO: add individual statistics? Which?
    def median(self):
        """
        Calculates the median ISI time.

        Returns
        -------
        pq.Quantity
        """
        return np.median(self._isi)
