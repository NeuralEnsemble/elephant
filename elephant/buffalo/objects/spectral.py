from copy import deepcopy
from .base import AnalysisObject
from collections import namedtuple

_PSDObjectTuple = namedtuple('PSDObject', 'frequencies psd')


class PSDObject(AnalysisObject, _PSDObjectTuple):
    """
    Class to store outputs of Elephant functions that compute power spectrum
    density (PSD) estimations (e.g.: `elephant.spectral.welch_psd`).

    Parameters
    ----------
    frequencies : (F,) array-like
        Vector with the frequencies for which the PSD was computed.
    psd : (N, F) array-like or (F,) array-like
        Value of the PSD for each frequency. If multichannel data, `N` is the
        number of channels and `F` the number of frequencies. Dimension `F`
        must be the same as `frequencies`.
    method : str
        Name of the method that was used to compute the PSD.
    params : dict, optional
        All the parameters that were used to compute the PSD.
        Default: None

    Raises
    ------
    ValueError
        If the dimension `F` of `frequencies` and `psd` differ.
    """

    def __new__(cls, frequencies, psd, method, params=None, *args, **kwargs):
        if psd.shape[-1] != frequencies.shape[0]:
            raise ValueError("PSD array must have the same length for the"
                             "frequency dimension")
        return super().__new__(cls, frequencies, psd)

    def __init__(self, frequencies, psd, method, params=None):
        super().__init__(self, frequencies, psd)
        self.method = method
        self.params = deepcopy(params)
