
import pytest
import quantities as pq
import neo
import numpy as np

import elephant

from pydantic import ValidationError
from elephant.schemas.function_validator import deactivate_validation, activate_validation

from elephant.schemas.schema_statistics import *
from elephant.schemas.schema_spike_train_correlation import *
from elephant.schemas.schema_spike_train_synchrony import *


def test_model_json_schema():
	# Just test that json_schema generation runs without error for all models
	model_classes = [
		PydanticCovariance,
		PydanticCorrelationCoefficient,
		PydanticCrossCorrelationHistogram,
		PydanticSpikeTimeTilingCoefficient,
		PydanticSpikeTrainTimescale,
		PydanticMeanFiringRate,
		PydanticInstantaneousRate,
		PydanticTimeHistogram,
		PydanticOptimalKernelBandwidth,
		PydanticIsi,
		PydanticCv,
		PydanticCv2,
		PydanticLv,
		PydanticLvr,
		PydanticFanofactor,
		PydanticComplexityPdf,
		PydanticSpikeContrast,
		PydanticComplexityInit,
		PydanticSynchrotoolInit,
		PydanticSynchrotoolDeleteSynchrofacts
	]
	for cls in model_classes:
		schema = cls.model_json_schema()
		assert isinstance(schema, dict)


"""
Checking for consistent behavior between Elephant functions and Pydantic models.
Tests bypass validate_with decorator if it is already implemented for that function
so consistency is checked correctly
"""

# Deactivate validation happening in the decorator of the elephant functions before all tests in this module to keep checking consistent behavior. Activates it again after all tests in this module have run.

@pytest.fixture(scope="module", autouse=True)
def module_setup_teardown():
    deactivate_validation()

    yield

    activate_validation()

@pytest.fixture
def make_list():
	return [0.01, 0.02, 0.05]

@pytest.fixture
def make_ndarray(make_list):
	return np.array(make_list)

@pytest.fixture
def make_pq_single_quantity():
	return 0.05 * pq.s

@pytest.fixture
def make_pq_multiple_quantity(make_ndarray):
	return make_ndarray * pq.s

@pytest.fixture
def make_spiketrain(make_pq_multiple_quantity):
	return neo.core.SpikeTrain(make_pq_multiple_quantity, t_start=0 * pq.s, t_stop=0.1 * pq.s)

@pytest.fixture
def make_spiketrains(make_spiketrain):
	return [make_spiketrain, make_spiketrain]

@pytest.fixture
def make_binned_spiketrain(make_spiketrain):
	return elephant.conversion.BinnedSpikeTrain(make_spiketrain, bin_size=0.01 * pq.s)

@pytest.fixture
def make_analog_signal():
	n2 = 300
	n0 = 100000 - n2
	return neo.AnalogSignal(np.array([10] * n2 + [0] * n0).reshape(n0 + n2, 1) * pq.dimensionless, sampling_period=1 * pq.s)

@pytest.fixture
def fixture(request):
	return request.getfixturevalue(request.param)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.mean_firing_rate, PydanticMeanFiringRate),
	(elephant.statistics.isi, PydanticIsi),
])
@pytest.mark.parametrize("fixture", [
	"make_list",
	"make_spiketrain",
    "make_ndarray",
    "make_pq_multiple_quantity",
], indirect=["fixture"])
def test_valid_spiketrain_input(elephant_fn, model_cls, fixture):
	valid = {"spiketrain": fixture}
	assert(isinstance(model_cls(**valid), model_cls))
	# just check it runs without error
	elephant_fn(**valid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.mean_firing_rate, PydanticMeanFiringRate),
	(elephant.statistics.isi, PydanticIsi),
])
@pytest.mark.parametrize("spiketrain", [
	5,
	"hello",
])
def test_invalid_spiketrain(elephant_fn, model_cls, spiketrain):
	invalid = {"spiketrain": spiketrain}
	with pytest.raises(TypeError):
		model_cls(**invalid)
	with pytest.raises((TypeError, ValueError)):
		elephant_fn(**invalid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.time_histogram, PydanticTimeHistogram),
	(elephant.statistics.complexity_pdf, PydanticComplexityPdf),
])
def test_valid_pq_quantity(elephant_fn, model_cls, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity}
	assert(isinstance(model_cls(**valid), model_cls))
	# just check it runs without error
	elephant_fn(**valid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.time_histogram, PydanticTimeHistogram),
	(elephant.statistics.complexity_pdf, PydanticComplexityPdf),
])
@pytest.mark.parametrize("pq_quantity", [
	5,
	"hello",
	[0.01, 0.02]
])
def test_invalid_pq_quantity(elephant_fn, model_cls, make_spiketrains, pq_quantity):
	invalid = {"spiketrains": make_spiketrains, "bin_size": pq_quantity}
	with pytest.raises(TypeError):
		model_cls(**invalid)
	with pytest.raises(AttributeError):
		elephant_fn(**invalid)



@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.instantaneous_rate, PydanticInstantaneousRate),
])
@pytest.mark.parametrize("fixture", [
	"make_list",
    "make_ndarray",
    "make_pq_multiple_quantity",
], indirect=["fixture"])
def test_invalid_spiketrains(elephant_fn, model_cls, fixture, make_pq_single_quantity):
	invalid = {"spiketrains": fixture, "sampling_period": make_pq_single_quantity}
	with pytest.raises(TypeError):
		model_cls(**invalid)
	with pytest.raises(TypeError):
		elephant_fn(**invalid)

@pytest.mark.parametrize("output", [
	"counts",
	"mean",
	"rate",
])
def test_valid_enum(output, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "output": output}
	assert(isinstance(PydanticTimeHistogram(**valid), PydanticTimeHistogram))
	# just check it runs without error
	elephant.statistics.time_histogram(**valid)

@pytest.mark.parametrize("output", [
	"countsfagre",
	5,
	"Counts",
	"counts ",
	" counts",
	"counts\n"
])
def test_invalid_enum(output, make_spiketrains, make_pq_single_quantity):
	invalid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "output": output}
	with pytest.raises(ValidationError):
		PydanticTimeHistogram(**invalid)
	with pytest.raises(ValueError):
		elephant.statistics.time_histogram(**invalid)


def test_valid_binned_spiketrain(make_binned_spiketrain):
	valid = {"binned_spiketrain": make_binned_spiketrain}
	assert(isinstance(PydanticCovariance(**valid), PydanticCovariance))
	# just check it runs without error
	elephant.spike_train_correlation.covariance(**valid)

def test_invalid_binned_spiketrain(make_spiketrain):
	invalid = {"binned_spiketrain": make_spiketrain}
	with pytest.raises(TypeError):
		PydanticCovariance(**invalid)
	with pytest.raises(AttributeError):
		elephant.spike_train_correlation.covariance(**invalid)

@pytest.mark.parametrize("elephant_fn,model_cls,invalid", [
	(elephant.statistics.instantaneous_rate, PydanticInstantaneousRate, {"spiketrains": [], "sampling_period": 0.01 * pq.s}),
	(elephant.statistics.optimal_kernel_bandwidth, PydanticOptimalKernelBandwidth, {"spiketimes": np.array([])}),
	(elephant.statistics.cv2, PydanticCv2, {"time_intervals": np.array([])*pq.s}),
])
def test_invalid_empty_input(elephant_fn, model_cls, invalid):

	with pytest.raises(ValueError):
		model_cls(**invalid)
	with pytest.raises((ValueError,TypeError)):
		elephant_fn(**invalid)

@pytest.mark.parametrize("elephant_fn,model_cls,parameter_name,empty_input", [
	(elephant.spike_train_correlation.covariance, PydanticCovariance, "binned_spiketrain", elephant.conversion.BinnedSpikeTrain(neo.core.SpikeTrain(np.array([])*pq.s, t_start=0*pq.s, t_stop=1*pq.s), bin_size=0.01*pq.s)),
])
def test_warning_empty_input(elephant_fn, model_cls, parameter_name, empty_input):
	warning = {parameter_name: empty_input}
	with pytest.warns(UserWarning):
		model_cls(**warning)
	with pytest.warns(UserWarning):
		elephant_fn(**warning)


def test_valid_Complexity(make_spiketrains, make_pq_single_quantity):
	valid = { "spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity }
	assert(isinstance(PydanticComplexityInit(**valid), PydanticComplexityInit))
	# just check it runs without error
	elephant.statistics.Complexity(**valid)