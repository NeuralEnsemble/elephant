
import pytest
import quantities as pq
import neo
import numpy as np

import elephant

from pydantic import ValidationError

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
		PydanticComplexityInit,
		PydanticSpikeContrast,
		PydanticSynchrotoolInit,
		PydanticSynchrotoolDeleteSynchrofacts,
	]
	for cls in model_classes:
		schema = cls.model_json_schema()
		assert isinstance(schema, dict)

"""
Checking for consistent behavior between Elephant functions and Pydantic models.
Tests bypass validate_with decorator if it is already implemented for that function
so consistency is checked correctly
"""

def call_elephant_function(elephant_fn, kwargs):
	if hasattr(elephant_fn, "_is_validate_with"):
		kwargs["not_validate"]=True
		elephant_fn(**kwargs)
	else:
		elephant_fn(**kwargs)

def assert_both_succeed_consistently(elephant_fn, model_cls, kwargs):
	"""Call both the Elephant function and the Pydantic model with the same kwargs.
	Assert both complete without raising exceptions.

	Parameters
	- elephant_fn: callable to invoke with kwargs
	- model_cls: Pydantic model class to instantiate with kwargs
	- kwargs: dict of keyword arguments to pass to both
	"""
	try:
		call_elephant_function(elephant_fn, kwargs)
	except Exception as e:
		assert False, f"Elephant function raised an exception: {e}"

	try:
		model_cls(**kwargs)
	except Exception as e:
		assert False, f"Pydantic model raised an exception: {e}"

def assert_both_warn_consistently(elephant_fn, model_cls, kwargs):
	"""Call both the Elephant function and the Pydantic model with the same kwargs.
	Assert both raise warnings.

	Parameters
	- elephant_fn: callable to invoke with kwargs
	- model_cls: Pydantic model class to instantiate with kwargs
	- kwargs: dict of keyword arguments to pass to both
	"""
	with pytest.warns(Warning) as w1:
		call_elephant_function(elephant_fn, kwargs)
	with pytest.warns(Warning) as w2:
		model_cls(**kwargs)


def assert_both_raise_consistently(elephant_fn, model_cls, kwargs, *, same_type=False, expected_exception=None):
	"""Call both the Elephant function and the Pydantic model with the same kwargs.
	Assert both raise, and if requested assert they raise the same exception type.

	Uses pytest.raises to capture exceptions so failures are reported with pytest's
	native formatting while still allowing comparison of exception objects.

	Parameters
	- elephant_fn: callable to invoke with kwargs
	- model_cls: Pydantic model class to instantiate with kwargs
	- kwargs: dict of keyword arguments to pass to both
	- same_type: if True assert the raised exception classes are identical
	- expected_exception: optional exception type that both must be instances of
	"""
	with pytest.raises(Exception) as e1:
		call_elephant_function(elephant_fn, kwargs)
	with pytest.raises(Exception) as e2:
		model_cls(**kwargs)

	exc1 = e1.value
	exc2 = e2.value

	if expected_exception is not None:
		assert isinstance(exc1, expected_exception), (
			f"Elephant raised {type(exc1)}, expected {expected_exception}")
		assert isinstance(exc2, expected_exception), (
			f"Pydantic raised {type(exc2)}, expected {expected_exception}")

	if same_type:
		if(type(exc1) is type(exc2)):
			return

		if (isinstance(exc1, (ValueError, TypeError)) and isinstance(exc2, (ValidationError, AttributeError))):
			return

		assert False, (
			f"Different exception types: Elephant={type(exc1)}, Pydantic={type(exc2)}. "
			f"Elephant exc: {exc1}; Pydantic exc: {exc2}")

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
	assert_both_succeed_consistently(elephant_fn, model_cls, valid)


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
	assert_both_raise_consistently(elephant_fn, model_cls, invalid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.time_histogram, PydanticTimeHistogram),
	(elephant.statistics.complexity_pdf, PydanticComplexityPdf),
])
def test_valid_pq_quantity(elephant_fn, model_cls, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity}
	assert_both_succeed_consistently(elephant_fn, model_cls, valid)


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
	valid = {"spiketrains": make_spiketrains, "bin_size": pq_quantity}
	assert_both_raise_consistently(elephant_fn, model_cls, valid)



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
	assert_both_raise_consistently(elephant_fn, model_cls, invalid)

@pytest.mark.parametrize("output", [
	"counts",
	"mean",
	"rate",
])
def test_valid_enum(output, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "output": output}
	assert_both_succeed_consistently(elephant.statistics.time_histogram, PydanticTimeHistogram, valid)

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
	assert_both_raise_consistently(elephant.statistics.time_histogram, PydanticTimeHistogram, invalid)


def test_valid_binned_spiketrain(make_binned_spiketrain):
	valid = {"binned_spiketrain": make_binned_spiketrain}
	assert_both_succeed_consistently(
		elephant.spike_train_correlation.covariance,
		PydanticCovariance,
		valid
	)

def test_invalid_binned_spiketrain(make_spiketrain):
	invalid = {"binned_spiketrain": make_spiketrain}
	assert_both_raise_consistently(
		elephant.spike_train_correlation.covariance,
		PydanticCovariance,
		invalid,
	)

@pytest.mark.parametrize("elephant_fn,model_cls,parameter_name,empty_input", [
	(elephant.statistics.instantaneous_rate, PydanticInstantaneousRate, "spiketrains", []),
	(elephant.statistics.optimal_kernel_bandwidth, PydanticOptimalKernelBandwidth, "spiketimes", np.array([])),
	(elephant.statistics.cv2, PydanticCv2, "time_intervals", np.array([])*pq.s),
])
def test_invalid_empty_input(elephant_fn, model_cls, parameter_name, empty_input):
	invalid = {parameter_name: empty_input}
	assert_both_raise_consistently(elephant_fn, model_cls, invalid)

@pytest.mark.parametrize("elephant_fn,model_cls,parameter_name,empty_input", [
	(elephant.spike_train_correlation.covariance, PydanticCovariance, "binned_spiketrain", elephant.conversion.BinnedSpikeTrain(neo.core.SpikeTrain(np.array([])*pq.s, t_start=0*pq.s, t_stop=1*pq.s), bin_size=0.01*pq.s)),
])
def test_warning_empty_input(elephant_fn, model_cls, parameter_name, empty_input):
	warning = {parameter_name: empty_input}
	assert_both_warn_consistently(elephant_fn, model_cls, warning)


def test_valid_Complexity(make_spiketrains, make_pq_single_quantity):
	valid = { "spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity }
	assert_both_succeed_consistently(
		elephant.statistics.Complexity,
		PydanticComplexityInit,
		valid,
	)