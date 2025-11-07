import numpy as np
import quantities as pq
import neo
import elephant
from enum import Enum
from typing import Any
import warnings

def get_length(obj) -> int:
    """
    Return the length (number of elements) of various supported datatypes:
    - list
    - numpy.ndarray
    - pq.Quantity
    - neo.SpikeTrain

    Returns
    -------
    int
        The number of elements or spikes in the object.

    Raises
    ------
    TypeError
        If the object type is not supported.
    """
    if obj is None:
        raise ValueError("Cannot get length of None")

    if isinstance(obj, elephant.trials.Trials):
        return len(obj.trials)
    elif isinstance(obj, elephant.conversion.BinnedSpikeTrain):
        return obj.n_bins
    elif isinstance(obj, neo.SpikeTrain):
        return len(obj)
    elif isinstance(obj, pq.Quantity):
        return obj.size
    elif isinstance(obj, np.ndarray):
        return obj.size
    elif isinstance(obj, (list,tuple)):
        return len(obj)


    
    else:
        raise TypeError(
            f"Unsupported type for length computation: {type(obj).__name__}"
        )
    
def is_sorted(obj) -> bool:
    if obj is None:
        raise ValueError("Cannot check sortedness of None")
    
    if isinstance(obj, (list, np.ndarray, pq.Quantity)):
        arr = np.asarray(obj)
        return np.all(arr[:-1] <= arr[1:])
    elif isinstance(obj, neo.SpikeTrain):
        arr = obj.magnitude  # Get the underlying numpy array of spike times
        return np.all(arr[:-1] <= arr[1:])
    return False

def is_matrix(obj) -> bool:
    if obj is None:
        raise ValueError("Cannot check matrix of None")
    if isinstance(obj, (list, np.ndarray, pq.Quantity)):
        arr = np.asarray(obj)
        return arr.ndim >= 2
    elif isinstance(obj, neo.SpikeTrain):
        arr = obj.magnitude  # Get the underlying numpy array of spike times
        return arr.ndim >= 2
    return False

def validate_covariance_matrix_rank_deficient(obj, info):
    """
    Check if the covariance matrix of the given object is rank deficient.
    Should work for elephant.trials.Trials, list of neo.core.spiketrainlist.SpikeTrainList or list of list of neo.core.SpikeTrain.
    """
    return obj

def validate_type(
    value,
    info,
    allowed_types: tuple,
    allow_none: bool,
):
    """Generic type validation helper."""
    if value is None:
        if allow_none:
            return value
        raise ValueError(f"{info.field_name} cannot be None")

    if not isinstance(value, allowed_types):
        raise TypeError(f"{info.field_name} must be one of {allowed_types}, not {type(value).__name__}")
    return value

def validate_length(
        value,
        info: str,
        min_length: int,
        warning: bool
):
    if min_length>0:
        if get_length(value) < min_length:
            if warning:
                warnings.warn(f"{info.field_name} has less than {min_length} elements", UserWarning)
            else:
                raise ValueError(f"{info.field_name} must contain at least {min_length} elements")
    return value

def validate_type_length(value, info, allowed_types: tuple, allow_none: bool, min_length: int, warning: bool = False):
    validate_type(value, info, allowed_types, allow_none)
    if value is not None:
        validate_length(value, info, min_length, warning)
    return value

def validate_array_content(value, info, allowed_types: tuple, allow_none: bool, min_length: int, allowed_content_types: tuple, min_length_content: int = 0):
    validate_type_length(value, info, allowed_types, allow_none, min_length)
    for i, item in enumerate(value):
        if not isinstance(item, allowed_content_types):
            raise TypeError(f"Element {i} in {info.field_name} must be {allowed_content_types}, not {type(item).__name__}")
        if min_length_content > 0 and get_length(item) >= min_length_content:
            hasContentLength = True
    if(min_length_content > 0 and not hasContentLength):
        raise ValueError(f"{info.field_name} must contain at least one element with at least {min_length_content} elements")
        
    return value

# ---- Specialized validation helpers ----

def validate_spiketrain(value, info, allowed_types=(list, neo.SpikeTrain, pq.Quantity, np.ndarray), allow_none = False, min_length = 1, check_sorted = False):
    validate_type_length(value, info, allowed_types, allow_none, min_length)
    if(check_sorted):
        if value is not None and not is_sorted(value):
            warnings.warn(f"{info.field_name} is not sorted", UserWarning)
    if(isinstance(value, neo.SpikeTrain)):
        if value.t_start is not None and value.t_stop is not None:
            if value.t_start > value.t_stop:
                raise ValueError(f"{info.field_name} has t_start > t_stop")
    return value

def validate_spiketrains(value, info, allowed_types = (list,), allow_none = False, min_length = 1, allowed_content_types = (list, neo.SpikeTrain, pq.Quantity, np.ndarray), min_length_content = 0):
    validate_array_content(value, info, allowed_types, allow_none, min_length, allowed_content_types, min_length_content)
    return value

def validate_spiketrains_matrix(value, info, allowed_types = (elephant.trials.Trials, list[neo.core.spiketrainlist.SpikeTrainList], list[list[neo.core.SpikeTrain]]), allow_none = False, min_length = 1, check_rank_deficient = False):
    if isinstance(value, list):
        validate_spiketrains(value, info, allowed_content_types=(neo.core.spiketrainlist,list[neo.core.SpikeTrain],))
    else:
        validate_type(value, info, (elephant.trials.Trials,), allow_none=False)
    if check_rank_deficient:
        return validate_covariance_matrix_rank_deficient(value, info)
    return value

def validate_time(value, info, allowed_types=(float, pq.Quantity) ,allow_none=True):
    if(isinstance(value, np.ndarray) and value.size==1):
        value = value.item()
    
    validate_type(value, info, allowed_types, allow_none)
    return value

def validate_quantity(value, info, allow_none=False):
    validate_type(value, info, (pq.Quantity,), allow_none)
    return value

def validate_time_intervals(value, info, allowed_types = (list, pq.Quantity, np.ndarray), allow_none = False, min_length=0, check_matrix = False):
    validate_type_length(value, info, allowed_types, allow_none, min_length)
    if check_matrix:
        if value is not None and is_matrix(value):
            raise ValueError(f"{info.field_name} is not allowed to be a matrix")
    return value

def validate_array(value, info, allowed_types=(list, np.ndarray) , allow_none=False, min_length=1, allowed_content_types = None, min_length_content = 0):
    if allowed_content_types is None:
        validate_type_length(value, info, allowed_types, allow_none, min_length)
    else:
        validate_array_content(value, info, allowed_types, allow_none, min_length, allowed_content_types, min_length_content)
    return value

def validate_binned_spiketrain(value, info, allowed_types=(elephant.conversion.BinnedSpikeTrain,), allow_none=False, min_length=1):
    validate_type_length(value, info, allowed_types, allow_none, min_length, warning=True)
    if value is not None and isinstance(value, elephant.conversion.BinnedSpikeTrain):
        spmat = value.sparse_matrix

        # Check for empty spike trains
        n_spikes_per_row = spmat.sum(axis=1)
        if n_spikes_per_row.min() == 0:
            warnings.warn(
                f'Detected empty spike trains (rows) in the {info.field_name}.', UserWarning)
    return value

def validate_dict_enum_types(value : dict[Enum, Any], info, typeDictionary: dict[Enum, type]):
    for key, val in value.items():
        if not isinstance(val, typeDictionary[key]):
            raise TypeError(f"Value for key {key} in {info.field_name} must be of type {typeDictionary[key].__name__}, not {type(val).__name__}")
    return value
        
def validate_key_in_tuple(value : str, info, t: tuple):
    if value not in t:
        raise ValueError(f"{info}:{value} is not in the options {t}")
    return value


# ---- Model validation helpers ----

def model_validate_spiketrains_same_t_start_stop(spiketrain, t_start, t_stop, name: str = "spiketrains", warning: bool = False):
    if(t_start is None or t_stop is None):
        first = True
        for i, item in enumerate(spiketrain):
            if first:
                t_start = item.t_start
                t_stop = item.t_stop
                first = False
            else:
                if t_start is None and item.t_start != t_start:
                    if warning:
                        warnings.warn(f"{name} has different t_start values among its elements", UserWarning)
                    else:
                        raise ValueError(f"{name} has different t_start values among its elements")
                if t_stop is None and item.t_stop != t_stop:
                    if warning:
                        warnings.warn(f"{name} has different t_stop values among its elements", UserWarning)
                    else:
                        raise ValueError(f"{name} has different t_stop values among its elements")
    else:
        if t_start>t_stop:
            raise ValueError(f"{name} has t_start > t_stop")
                
def model_validate_spiketrains_sam_t_start_stop(spiketrain_i, spiketrain_j):
    if spiketrain_i.t_start != spiketrain_j.t_start:
            raise ValueError("spiketrain_i and spiketrain_j need to have the same t_start")
    if spiketrain_i.t_stop != spiketrain_j.t_stop:
            raise ValueError("spiketrain_i and spiketrain_j need to have the same t_stop")
                
def model_validate_time_intervals_with_nan(time_intervals , with_nan, name: str = "time_intervals"):
    if get_length(time_intervals)<2:
        if(with_nan):
            warnings.warn(f"{name} has less than two entries so a np.Nan will be generated", UserWarning)
        else:
            raise ValueError(f"{name} has less than two entries")
        
def model_validate_binned_spiketrain_fast(binned_spiketrain, fast, name: str = "binned_spiketrain"):
    if(fast and np.max(binned_spiketrain.shape) > np.iinfo(np.int32).max):
        raise MemoryError(f"{name} is too large for fast=True option")
        