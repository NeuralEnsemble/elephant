import quantities as pq
import numpy as np
from typing import (
    Any,
    Union,
    Self,
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

import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

class PydanticCovariance(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_correlation.covariance function
    with additional type checking and json_schema by PyDantic.
    """

    binned_spiketrain: Any = Field(..., description="Binned spike train")
    binary: Optional[bool] = Field(False, description="Use binary binned vectors")
    fast: Optional[bool] = Field(True, description="Use faster implementation")

    @field_validator("binned_spiketrain")
    @classmethod
    def validate_binned_spiketrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)
    
    @model_validator(mode="after")
    def validate_model(self) -> Self:             
        fv.model_validate_binned_spiketrain_fast(self.binned_spiketrain, self.fast)
        return self


class PydanticCorrelationCoefficient(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_correlation.correlation_coefficient function
    with additional type checking and json_schema by PyDantic.
    """

    binned_spiketrain: Any = Field(..., description="Binned spike train")
    binary: Optional[bool] = Field(False, description="Use binary binned vectors")
    fast: Optional[bool] = Field(True, description="Use faster implementation")

    @field_validator("binned_spiketrain")
    @classmethod
    def validate_binned_spiketrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)
    
    @model_validator(mode="after")
    def validate_model(self) -> Self:             
        fv.model_validate_binned_spiketrain_fast(self.binned_spiketrain, self.fast)
        return self


class PydanticCrossCorrelationHistogram(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_correlation.cross_correlation_histogram function
    with additional type checking and json_schema by PyDantic.
    """

    class WindowOptions(Enum):
        full = "full"
        valid = "valid"

    class MethodOptions(Enum):
        speed = "speed"
        memory = "memory"

    binned_spiketrain_i: Any = Field(..., description="Binned spike train i")
    binned_spiketrain_j: Any = Field(..., description="Binned spike train j")
    window: Optional[Union[WindowOptions, list[int]]] = Field(WindowOptions.full, description="Window")
    border_correction: Optional[bool] = Field(False, description="Correct border effect")
    binary: Optional[bool] = Field(False, description="Count spike falling same bin as one")
    kernel: Optional[Any] = Field(None, description="array containing a smoothing kernel")
    method: Optional[MethodOptions] = Field(MethodOptions.speed, description="Method of calculating")
    cross_correlation_coefficient: Optional[bool] = Field(False, description="Normalize CCH")

    @field_validator("binned_spiketrain_i", "binned_spiketrain_j")
    @classmethod
    def validate_binned_spiketrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)

    @field_validator("kernel")
    @classmethod
    def validate_kernel(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=True)


class PydanticSpikeTimeTilingCoefficient(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_correlation.spike_time_tiling_coefficient function
    with additional type checking and json_schema by PyDantic.
    """

    spiketrain_i: Any = Field(..., description="Spike train Object i")
    spiketrain_j: Any = Field(..., description="Spike train Object j (same T_start and same t_stop)")
    dt: Any = Field(default_factory=lambda: 0.005 * pq.s, description="Synchronicity window")

    @field_serializer("dt", mode='plain')
    def serialize_quantity(self, value: pq.Quantity):
        return fs.serialize_quantity(value)

    @field_validator("spiketrain_i", "spiketrain_j")
    @classmethod
    def validate_spiketrain(cls, v, info):
        # require specifically neo.core.SpikeTrain for this validator
        return fv.validate_spiketrain(v, info, allowed_types=(neo.core.SpikeTrain,))

    @field_validator("dt")
    @classmethod
    def validate_dt(cls, v, info):
        return fv.validate_quantity(v, info)

    @model_validator(mode="after")
    def check_correctTypeCombination(self) -> Self:
        fv.model_validate_two_spiketrains_same_t_start_stop(self.spiketrain_i, self.spiketrain_j)
        return self


class PydanticSpikeTrainTimescale(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_correlation.spike_train_timescale function
    with additional type checking and json_schema by PyDantic.
    """

    binned_spiketrain: Any = Field(..., description="Binned spike train")
    max_tau: Any = Field(..., description="Maximal integration time")

    @field_validator("binned_spiketrain")
    @classmethod
    def validate_binned_spiketrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)

    @field_validator("max_tau")
    @classmethod
    def validate_max_tau(cls, v, info):
        return fv.validate_quantity(v, info)

    @model_validator(mode="after")
    def check_correctTypeCombination(self) -> Self:
        if self.max_tau % self.binned_spiketrain.bin_size > 0.00001:
            raise ValueError("max_tau has to be a multiple of binned_spiketrain.bin_size")
        return self


