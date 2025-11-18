import quantities as pq
import numpy as np
from typing import (
    Any,
    Union,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    field_serializer
)
import neo
from enum import Enum
import elephant

from elephant.kernels import Kernel
import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

import warnings

class PydanticMeanFiringRate(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.mean_firing_rate function
    with additional type checking and json_schema by PyDantic.
    """
    spiketrain: Any = Field(None, description="SpikeTrain Object")
    t_start: Optional[Any] = Field(None, description="Start time")
    t_stop: Optional[Any] = Field(None, description="Stop time")
    axis: Optional[int] = Field(None, description="Axis of calculation")

    @field_validator("spiketrain")
    @classmethod
    def validate_spiketrain(cls, v, info):
        return fv.validate_spiketrain(v, info, allow_none=True)
    
    @field_validator("t_start", "t_stop")
    @classmethod
    def validate_time(cls, v, info):
        return fv.validate_time(v, info)

    @model_validator(mode="after")
    def validate_model(self):             
        if isinstance(self.spiketrain, (np.ndarray, list)):
            if isinstance(self.t_start, pq.Quantity) or isinstance(self.t_stop, pq.Quantity):
                raise TypeError("spiketrain is a np.ndarray or list but t_start or t_stop is pq.Quantity")
        elif not (isinstance(self.t_start, pq.Quantity) and isinstance(self.t_stop, pq.Quantity)):
            raise TypeError("spiketrain is a neo.SpikeTrain or pq.Quantity but t_start or t_stop is not pq.Quantity")
        return self
    
class PydanticInstantaneousRate(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.instantaneous_rate function
    with additional type checking and json_schema by PyDantic.
    """

    class KernelOptions(Enum):
        auto = "auto"

    spiketrains: Any = Field(..., description="Input spike train(s)")
    sampling_period: Any = Field(..., gt=0, description="Time stamp resolution of spike times")
    kernel: Union[KernelOptions, Any] = Field(KernelOptions.auto, description="Kernel for convolution")
    cutoff: Optional[float] = Field(5.0, ge=0, description="cutoff of probability distribution")
    t_start: Optional[Any] = Field(None, description="Start time")
    t_stop: Optional[Any] = Field(None, description="Stop time")
    trim: Optional[bool] = Field(False, description="Only return region of convolved signal")
    center_kernel: Optional[bool] = Field(True, description="Center the kernel on spike")
    border_correction: Optional[bool] = Field(False, description="Apply border correction")
    pool_trials: Optional[bool] = Field(False, description="Calc firing rates averaged over trials when spiketrains is Trials object")
    pool_spike_trains: Optional[bool] = Field(False, description="Calc firing rates averaged over spiketrains")


    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        if(isinstance(v, (list, neo.core.spiketrainlist.SpikeTrainList))):
            return fv.validate_spiketrains(v, info, allowed_types=(list, neo.core.spiketrainlist.SpikeTrainList), allowed_content_types=(neo.SpikeTrain,))
        if(isinstance(v, neo.SpikeTrain)):
            return fv.validate_spiketrain(v, info, allowed_types=(neo.SpikeTrain,))
        return fv.validate_spiketrains_matrix(v, info)

    @field_validator("sampling_period")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    
    @field_validator("kernel")
    @classmethod
    def validate_kernel(cls, v, info):
        if v == cls.KernelOptions.auto.value:
            return v
        return fv.validate_type(v, info, allowed_types=(Kernel), allow_none=False)
    
    @field_validator("t_start", "t_stop")
    @classmethod
    def validate_time(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        if(isinstance(self.kernel, Kernel) and self.cutoff < self.kernel.min_cutoff):
            warnings.warn(f"cutoff {self.cutoff} is smaller than the minimum cutoff {self.kernel.min_cutoff} of the kernel", UserWarning)
        if isinstance(self.spiketrains, list):
            fv.model_validate_spiketrains_same_t_start_stop(self.spiketrains, self.t_start, self.t_stop, warning=True)
        return self

class PydanticTimeHistogram(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.time_histogram function
    with additional type checking and json_schema by PyDantic.
    """

    class OutputOptions(Enum):
        counts = "counts"
        mean = "mean"
        rate = "rate"

    spiketrains: list = Field(..., description="List of Spiketrains")
    bin_size: Any = Field(..., description="Width histogram's time bins")
    t_start: Optional[Any] = Field(None, description="Start time")
    t_stop: Optional[Any] = Field(None, description="Stop time")
    output: Optional[OutputOptions] = Field(OutputOptions.counts, description="Normalization")
    binary: Optional[bool] = Field(False, description="To binary")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.SpikeTrain,))
    
    @field_validator("bin_size")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    
    @field_validator("t_start", "t_stop")
    @classmethod
    def validate_quantity_none(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        fv.model_validate_spiketrains_same_t_start_stop(self.spiketrains, self.t_start, self.t_stop, warning=True)
        return self
    
class PydanticOptimalKernelBandwidth(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.optimal_kernel_bandwidth function
    with additional type checking and json_schema by PyDantic.
    """
    
    spiketimes: Any = Field(..., description="Sequence of spike times(ASC)")
    times: Optional[Any] = Field(None, description="Time at which kernel bandwidth")
    bandwidth: Optional[Any] = Field(None, description="Vector of kernal bandwidth")
    bootstrap: Optional[bool] = Field(False, description="Use Bootstrap")

    @field_validator("spiketimes")
    @classmethod
    def validate_ndarray(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,))
    
    @field_validator("times", "bandwidth")
    @classmethod
    def validate_ndarray_none(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=True)

class PydanticIsi(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.isi function
    with additional type checking and json_schema by PyDantic.
    """
    spiketrain: Any = Field(..., description="SpikeTrain Object (sorted)")
    axis: Optional[int] = Field(-1, description="Difference Axis")

    @field_validator("spiketrain")
    @classmethod
    def validate_spiketrain_sorted(cls, v, info):
        return fv.validate_spiketrain(v, info, check_sorted=True)

class PydanticCv(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.cv function
    with additional type checking and json_schema by PyDantic.
    """
    class NanPolicyOptions(Enum):
        propagate = "propagate"
        omit = "omit"
        _raise = "raise"

    args: Any = Field(..., description="Input array")
    axis: Union[int, None] = Field(0, description="Compute statistic axis")
    nan_policy: NanPolicyOptions = Field(NanPolicyOptions.propagate, description="How handle input NaNs")
    ddof: Optional[int] = Field(0, ge=0, description="Delta Degrees Of Freedom")
    keepdims: Optional[bool] = Field(False, description="leave reduced axes in one-dimensional result")

    @field_validator("args")
    @classmethod
    def validate_array(cls, v, info):
        return fv.validate_array(v, info)
    
class PydanticCv2(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.cv2 function
    with additional type checking and json_schema by PyDantic.
    """

    time_intervals: Any = Field(..., description="Vector of time intervals")
    with_nan: Optional[bool] = Field(False, description="Do not Raise warning on short spike train")

    @field_validator("time_intervals")
    @classmethod
    def validate_time_intervals(cls, v, info):
       return fv.validate_time_intervals(v, info, check_matrix=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        fv.model_validate_time_intervals_with_nan(self.time_intervals, self.with_nan)
        return self
    
class PydanticLv(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.lv function
    with additional type checking and json_schema by PyDantic.
    """

    time_intervals: Any = Field(..., description="Vector of time intervals")
    with_nan: Optional[bool] = Field(False, description="Do not Raise warning on short spike train")

    @field_validator("time_intervals")
    @classmethod
    def validate_time_intervals(cls, v, info):
       return fv.validate_time_intervals(v, info, check_matrix=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        fv.model_validate_time_intervals_with_nan(self.time_intervals, self.with_nan)
        return self
    
class PydanticLvr(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.lvr function
    with additional type checking and json_schema by PyDantic.
    """

    time_intervals: Any = Field(..., description="Vector of time intervals (default units: ms)")
    R: Any = Field(default_factory=lambda: 5. * pq.ms, ge=0, description="Refractoriness constant (default quantity: ms)")
    with_nan: Optional[bool] = Field(False, description="Do not Raise warning on short spike train")

    @field_serializer("R", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)

    @field_validator("time_intervals")
    @classmethod
    def validate_time_intervals(cls, v, info):
       return fv.validate_time_intervals(v, info, check_matrix=True)
    
    @field_validator("R")
    @classmethod
    def validate_R(cls, v, info):
        fv.validate_type(v, info, (pq.Quantity, int, float), allow_none=False)
        if(not isinstance(v, pq.Quantity)):
            warnings.warn("R does not have any units so milliseconds are assumed", UserWarning)
        return v
    
    @model_validator(mode="after")
    def validate_model(self):             
        fv.model_validate_time_intervals_with_nan(self.time_intervals, self.with_nan)
        return self
    
class PydanticFanofactor(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.fanofactor function
    with additional type checking and json_schema by PyDantic.
    """
    spiketrains: list = Field(..., description="List of Spiketrains")
    warn_tolerance: Any = Field(default_factory=lambda: 0.1 * pq.ms, ge=0, description="Warn tolerence of variations")

    @field_serializer("warn_tolerance", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, min_length=0)
    
    @field_validator("warn_tolerance")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)

class PydanticComplexityPdf(BaseModel):
    """
    PyDantic Class to wrap the elephant.statistics.complexity_pdf function
    with additional type checking and json_schema by PyDantic.
    """
    spiketrains: list = Field(..., description="List of Spiketrains")
    bin_size: Any = Field(..., description="Width histogram's time bins")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        fv.validate_spiketrains(v, info, allowed_content_types=(neo.SpikeTrain,))
        fv.model_validate_spiketrains_same_t_start_stop(v, None, None)
        return v
    
    @field_validator("bin_size")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)

class PydanticComplexityInit(BaseModel):
    spiketrains: list = Field(..., description="List of neo.SpikeTrain objects with common t_start/t_stop")
    sampling_rate: Optional[Any] = Field(None, description="Sampling rate (1/time)")
    bin_size: Optional[Any] = Field(None, description="Width of histogram bins")
    binary: Optional[bool] = Field(True, description="If True count neurons, else total spikes")
    spread: Optional[int] = Field(0, ge=0, description="Number of bins for synchronous spikes (>=0)")
    tolerance: Optional[float] = Field(1e-8, description="Tolerance for rounding errors")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        fv.validate_spiketrains(v, info, allowed_content_types=(neo.SpikeTrain,))
        fv.model_validate_spiketrains_same_t_start_stop(v, None, None)
        return v

    @field_validator("bin_size")
    @classmethod
    def validate_bin_size(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    
    @field_validator("sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v, info):
        fv.validate_quantity(v, info, allow_none=True)
        if v is None:
            warnings.warn("no sampling rate is supplied. This may lead to rounding errors when using the epoch to slice spike trains", UserWarning)
        return v

    @model_validator(mode="after")
    def check_rate_or_bin(self):
        if self.sampling_rate is None and self.bin_size is None:
            raise ValueError("Either sampling_rate or bin_size must be set")
        return self