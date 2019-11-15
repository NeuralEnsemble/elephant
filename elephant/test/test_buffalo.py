# -*- coding: utf-8 -*-
"""
Unit tests for the Buffalo analysis objects.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import print_function, unicode_literals

from itertools import chain
import unittest
import elephant.buffalo as buf


# #####################################################################################################################
# We create local Analysis subclasses for testing the superclass basic behaviors
#######################################################################################################################

class BasicClassBehavior(buf.base.Analysis):
    _required_params = ['low_cutoff', 'high_cutoff']

    _required_types = {'low_cutoff': (int, float),
                       'high_cutoff': (int, float),
                       'method': (str,)               # optional parameter
                       }

    def _process(self, *args, **kwargs):
        return


class BasicClassNoProcess(buf.base.Analysis):
    """Should throw NotImplementedError"""


class BasicNoParameters(buf.base.Analysis):

    def _process(self, *args, **kwargs):
        return

class BasicClassWrongRequiredParams(buf.base.Analysis):
    _required_params = None

    def _process(self, *args, **kwargs):
        return


class BasicClassWrongRequiredParamsItems(buf.base.Analysis):
    _required_params = [1, 2]

    def _process(self, *args, **kwargs):
        return


class BasicClassWrongRequiredTypes(buf.base.Analysis):
    _required_types = None

    def _process(self, *args, **kwargs):
        return


class BasicClassWrongRequiredTypesItems(buf.base.Analysis):
    # no commas. Values are going to be evaluated as int or str directly, not tuple
    _required_types = {'low_cutoff': (int),
                       'high_cutoff': (int),
                       'method': (str)
                       }

    def _process(self, *args, **kwargs):
        return


class BasicClassNoRequiredTypes(buf.base.Analysis):
    _required_params = ['low_cutoff', 'high_cutoff']

    def _process(self, *args, **kwargs):
        return


class BasicClassNoRequiredParams(buf.base.Analysis):
    _required_types = {'low_cutoff': (int,)}

    def _process(self, *args, **kwargs):
        return


########################################################################################################################
# Test cases
########################################################################################################################

class AnalysisBaseClassTestCase(unittest.TestCase):

    def test_input_parameter_checks(self):
        print("Input validation check")

        # Params is not a dict, should throw exception
        with self.assertRaises(buf.exceptions.BuffaloWrongParametersDictionary):
            BasicClassBehavior(params=True)

        # Params is empty dict, should throw exception
        with self.assertRaises(buf.exceptions.BuffaloMissingRequiredParameters):
            BasicClassBehavior()

        # Params has only non-required items, should throw exception
        missing_analysis_params = {'cutoff': 10}
        with self.assertRaises(buf.exceptions.BuffaloMissingRequiredParameters):
            BasicClassBehavior(params=missing_analysis_params)

        # Params has only one required item, should throw exception
        only_one_required_analysis_params = {'high_cutoff': 20,
                                             'extra': "raw"}
        with self.assertRaises(buf.exceptions.BuffaloMissingRequiredParameters):
            BasicClassBehavior(params=only_one_required_analysis_params)

        # Params has all required items and a non-validated item, should pass
        extra_non_validated_analysis_params = {'low_cutoff': 10,
                                               'high_cutoff': 20,
                                               'extra_non_validated': "raw"}
        extra_non_validated = BasicClassBehavior(params=extra_non_validated_analysis_params)
        self.assertIsInstance(extra_non_validated, BasicClassBehavior)

        # Creating the next two objects with either of these parameters types should work
        analysis_1_params = {'low_cutoff': 10,
                             'high_cutoff': 20}

        analysis_2_params = {'low_cutoff': 10.5,
                             'high_cutoff': 20.5}

        analysis_1 = BasicClassBehavior(params=analysis_1_params)
        analysis_2 = BasicClassBehavior(params=analysis_2_params)
        self.assertIsInstance(analysis_1, BasicClassBehavior)
        self.assertIsInstance(analysis_2, BasicClassBehavior)

        # One of the required parameters has wrong type, should throw exception
        wrong_required_type_analysis_params = {'low_cutoff': '10',
                                               'high_cutoff': 20}
        with self.assertRaises(buf.exceptions.BuffaloWrongParameterType):
            BasicClassBehavior(params=wrong_required_type_analysis_params)

        # All required parameters, one non-required but validated parameter with the right type. Should pass
        extra_analysis_params = {'low_cutoff': 10,
                                 'high_cutoff': 20.0,
                                 'method': "raw"}
        extra_analysis = BasicClassBehavior(params=extra_analysis_params)
        self.assertIsInstance(extra_analysis, BasicClassBehavior)

        # All required parameters, one non-required but validated parameter has the wrong type. Should throw exception
        error_extra_analysis_params = {'low_cutoff': 10,
                                       'high_cutoff': 20.0,
                                       'method': 1}
        with self.assertRaises(buf.exceptions.BuffaloWrongParameterType):
            BasicClassBehavior(params=error_extra_analysis_params)

    def test_inheritance_errors(self):
        print("Inheritance check")

        analysis_params = {'low_cutoff': 10,
                           'high_cutoff': '20'}
        types_analysis_params = {'low_cutoff': 10}
        types_error_analysis_params = {'low_cutoff': 10.0}

        # Class without validation, but without custom `_process`. Should accept empty input parameters but throw
        # NotImplementedError
        with self.assertRaises(NotImplementedError):
            BasicClassNoProcess()

        # This should work despite not giving input parameters
        self.assertIsInstance(BasicNoParameters(), BasicNoParameters)

        # Wrong implementations of `_required_types` and `_required_args`
        with self.assertRaises(AssertionError):
            BasicClassWrongRequiredParams(params=analysis_params)
        with self.assertRaises(AssertionError):
            BasicClassWrongRequiredParamsItems(params=analysis_params)
        with self.assertRaises(AssertionError):
            BasicClassWrongRequiredParamsItems(params=analysis_params)
        with self.assertRaises(AssertionError):
            BasicClassWrongRequiredTypes(params=analysis_params)
        with self.assertRaises(AssertionError):
            BasicClassWrongRequiredTypesItems(params=analysis_params)

        # Despite string, should work because not checking types
        self.assertIsInstance(BasicClassNoRequiredTypes(params=analysis_params), BasicClassNoRequiredTypes)

        # Should fail because checking only required parameters
        with self.assertRaises(buf.exceptions.BuffaloMissingRequiredParameters):
            BasicClassNoRequiredTypes(params=types_analysis_params)

        # Should work because types are correct
        self.assertIsInstance(BasicClassNoRequiredParams(params=types_analysis_params), BasicClassNoRequiredParams)

        # Should fail because doing type checking only
        with self.assertRaises(buf.exceptions.BuffaloWrongParameterType):
            BasicClassNoRequiredParams(params=types_error_analysis_params)


if __name__ == '__main__':
    unittest.main()
