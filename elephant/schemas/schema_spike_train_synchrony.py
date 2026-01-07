import quantities as pq
import numpy as np
from typing import (
    Any,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer
)
import neo
from enum import Enum
from elephant.schemas.schema_statistics import PydanticComplexityInit

import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs


class PydanticSpikeContrast(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_synchrony.spike_contrast function
    with additional type checking and json_schema by PyDantic.
    """

    spiketrains: list = Field(..., description="List of Spiketrains")
    t_start: Optional[Any] = Field(None, description="Start time")
    t_stop: Optional[Any] = Field(None, description="Stop time")
    min_bin: Optional[Any] = Field(default_factory=lambda: 10. * pq.ms, description="Min value for bin_min")
    bin_shrink_factortime: Optional[float] = Field(0.9, description="Shrink bin size multiplier", ge=0., le=1.)
    return_trace: Optional[bool] = Field(False, description="Return history of spike-contrast synchrony")

    @field_serializer("min_bin", mode='plain')
    def serialize_quantity(self, value: pq.Quantity):
        return fs.serialize_quantity(value)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.SpikeTrain,), min_length=2, min_length_content=2)

    @field_validator("t_start", "t_stop")
    @classmethod
    def validate_time(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)

    @field_validator("min_bin")
    @classmethod
    def validate_min_bin(cls, v, info):
        return fv.validate_quantity(v, info)
    

class PydanticSynchrotoolInit(PydanticComplexityInit):
    pass

class PydanticSynchrotoolDeleteSynchrofacts(BaseModel):
    class ModeOptions(Enum):
        delete = "delete"
        extract = "extract"

    threshold: int = Field(..., gt=1, description="Threshold for deletion of spikes")
    in_place: Optional[bool] = Field(False, description="Make modification in place")
    mode: Optional[ModeOptions] = Field(ModeOptions.delete, description="Inversion of mask for deletion")