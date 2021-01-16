from copy import deepcopy
from .base import AnalysisObject
from collections import namedtuple

_PSDObjectTuple = namedtuple('PSDObject', 'freqs psd')

class PSDObject(AnalysisObject, _PSDObjectTuple):
    """AnalysisObject"""

    def __new__(cls, freqs, psd, method, params=None, *args, **kwargs):
        return super().__new__(cls, freqs, psd)

    def __init__(self, freqs, psd, method, params=None, *args, **kwargs):
        super().__init__(self, freqs, psd)
        self.method = method
        self.params = deepcopy(params)