# -*- coding: utf-8 -*-
"""
This module implements base super classes from which all Buffalo objects are derived.

These classes support a standardized flow for processing a data input, producing standardized outputs and provenance
information.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from __future__ import print_function
from datetime import datetime
from copy import deepcopy
from neo.core.baseneo import _check_annotations

from .exceptions import (BuffaloMissingRequiredParameters, BuffaloWrongParametersDictionary, BuffaloWrongParameterType,
                         BuffaloInputValidationFailed)


class Analysis(object):
    """
    This is the superclass for the Buffalo analysis objects.

    It supports basic input validation and storage of basic provenance information.
    Input parameters are passed as the `params` keyed argument in class initialization.
    Any other initialization argument/keyed-argument is treated as input data.

    Methods
    -------

    annotate
        Stores custom values as items in a dictionary inside the class


    Properties
    ----------

    params: dict
        The analysis input parameters that are passed during class initializaition (`params` keyed argument)

    name: str
        Required class attribute that provides a name for the object

    description: str
        Required class attribute that provides a human-readable description for the object

    cno_name: str
        Optional class attribute that describes the object according to the Common Neuroscience Ontology schema

    annotations: dict
        Any custom annotations that are stored in the object

    creation_time: datetime
        Time the instance was created (UTC)

    execution_time: datetime
        Time that the analysis main function was run (UTC)

    finish_time: datetime
        Time that the analysis main function finished (UTC)


    Notes
    -----

    Child classes can be easily derived by just overriding a few protected variables or functions.

    _required_params: list of str
        Defines the keys that MUST be present in the `params` dictionary

    _required_types: dict
        Items take the form of 'str: tuple', where the string defines a key name in the `params` dictionary.
        The tuple should define any type that is valid for the parameter.
        Any parameter (required or not) may be listed for type validation.
        Therefore, expected optional parameters may be readily checked.

        IMPORTANT: for unique types, the tuple must end with a trailing comma, otherwise Python just returns the element

        Example:
            param X must be only integer --> _required_types = {'X': (int,)}

            param Y can be either integer or float --> _required_types = {'Y': (int, float)}

    _process: function
        This is the function that contains the logic of the analysis.
        Must be overridden, otherwise NotImplementedError is raised.

    _validate: function
        This is an optional function that is used to validate input data and parameters values.
        It happens after the basic type validation of input parameters was done.
        Returns True if not overridden.
    """

    _params = None                    # Internal copy of the dictionary of input parameters for the analysis

    _ts_create = None                 # UTC time of instance creation
    _ts_execute = None                # UTC time of analysis execution
    _ts_end = None                    # UTC time of analysis execution end

    _custom_annotations = None        # Dictionary with custom annotations

    _name = None                      # Name of the analysis
    _cno_name = None                  # Name according to the CNO ontology TODO: check against official schema
    _description = None               # Human readable description of the object

    _required_params = []    # List of keys that must be present in the 'params' dictionary during class initialization

    # Dictionary with tuples describing the accepted types for each item in the 'params' dictionary. May include
    # additional, non required parameters known in advance, and that can be automatically checked
    _required_types = {}

    def __init__(self, *args, params={}, **kwargs):
        """
        Initializes the Analysis object.
        Automatic validation is performed, by checking if all required parameters were passed and if inputs are correct.
        Finally, it initializes basic provenance information.

        Parameters
        ----------

        params : dict
            All input parameters relevant for the analysis.
            An input parameter is defined as any input that controls the analysis, but that is not the data that is the
            subject of the analysis.

        args : list
            Input **data** necessary for the analysis.
            Should be validated accordingly by each subclass.

        kwargs: dict
            Keyed-arguments describing input **data** necessary for the analysis.
            Should be validated accordingly by each subclass.

        Raises
        ------

        BuffaloWrongParametersDictionary
            If `params` is not a dictionary.

        BuffaloMissingRequiredParameters
            If any of the required parameters specified in the `_required_params` list is missing

        BuffaloWrongParameterType
            If any of the input parameters has a different type than specified in `_required_types`

        BuffaloInputValidationFailed
            If the '_validate' function returns False (i.e., the input parameters and data are not adequate for the
                analysis.

        AssertionError
            If there was an error in the implementation of `_required_params` or `_required_types`, or the required
            class attributes `name` and `description`

        NotImplementedError
            If `_process` is not overriddn by the child class

        """
        self._ts_create = datetime.utcnow()    # Stores the timestamp when the instance was created

        # Check required class attributes are present
        assert isinstance(self.name, str) and len(self.name), "Must define 'name' attribute with non-empty string"
        assert isinstance(self.description, str) is not None and len(self._description), ("Must define 'description'"
                                                                                          "attribute with non-empty"
                                                                                          "string")

        # Check if optional class attributes are valid if present
        if self.cno_name is not None:
            assert isinstance(self.cno_name, str) and len(self.cno_name), "CNO name must be non-empty string"

        # Check that protected variables are correct
        assert isinstance(self._required_params, list), "Required params must be 'list'"
        assert isinstance(self._required_types, dict), "Required types must be 'dict'"

        # Check that analysis parameters were passed, and store
        if not isinstance(params, dict):
            raise BuffaloWrongParametersDictionary("The analysis parameters must be passed as a dictionary in the "
                                                   "'params' keyed argument")
        self._params = deepcopy(params)

        # Verify that analysis parameters are valid.
        # This is a type validation, not value validation.
        if not self._has_required_params():
            raise BuffaloMissingRequiredParameters(f"Check that '{', '.join(self._required_params)} are present in the"
                                                   "parameters dictionary")
        if not self._params_are_valid():
            raise BuffaloWrongParameterType("Check that the analysis input parameters types are correct")

        # Performs the optional validation on the input parameters/data
        if not self._validate(*args, **kwargs):
            raise BuffaloInputValidationFailed("Check that parameters/data are correct for the analysis")

        # Run the analysis for the provided data
        # 'params' are removed as any other arg/kwarg is supposed to be data, which is the only input for the '_process'
        # function.
        if 'params' in kwargs:
            kwargs.pop('params')

        self._ts_execute = datetime.utcnow()   # Stores the timestamp when '_process' function is called
        self._process(*args, **kwargs)         # Run the analysis on the given input data
        self._ts_end = datetime.utcnow()       # Stores the timestamp when analysis finished

    def _validate(self, *args, **kwargs):
        """
        This functions performs a check in the input parameters and input data, to assert that they are correct
        according to the analysis algorithm.
        This is optional. If not overridden, the function always returns True.

        Parameters
        ----------

        args : list
            Input **data** necessary for the analysis.

        kwargs: dict
            Keyed-arguments describing input **data** necessary for the analysis.

        Returns
        -------

            boolean
                True if validation passed, False if anything happened
        """
        return True

    def _process(self, *args, **kwargs):
        """
        This is the main logic for the analysis.
        Input data is passed from the class initialization
        """
        raise NotImplementedError

    def _has_required_params(self):
        """
        Internal function that checks if the parameters dictionary stored inside the class has the necessary items.

        Returns
        -------

        boolean
            True if check is OK, False if missing any of the parameters.
            If required params list is empty, returns True.
        """
        parameters = list(self.params.keys())
        if len(self._required_params):
            if len(parameters) >= len(self._required_params):
                for required in self._required_params:
                    assert isinstance(required, str), "Required params must contain strings"
                    if required not in parameters:
                        print(f"Missing parameter '{required}'")
                        return False
                return True
            print("Fewer than the number of required parameters passed")
            return False
        return True

    def _params_are_valid(self):
        """
        Internal function that checks if the input parameters types are acceptable

        Returns
        -------

        boolean
            True if check is OK, False if wrong type was passed.
            If type checking dictionary is empty, returns True
        """
        for key, value in self.params.items():
            if key in self._required_types.keys():
                assert isinstance(self._required_types[key], tuple), "Values of required types dict must be tuples"
                if type(value) not in self._required_types[key]:
                    print(f"Parameter '{key}' should be one of: {', '.join(map(str, self._required_types[key]))}")
                    return False
        return True

    @property
    def creation_time(self):
        return self._ts_create

    @property
    def execution_time(self):
        return self._ts_execute

    @property
    def finish_time(self):
        return self._ts_end

    @property
    def annotations(self):
        return self._custom_annotations

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def cno_name(self):
        return self._cno_name

    @property
    def params(self):
        return self._params

    def annotate(self, **annotations):
        """
        Inserts annotations in the object.

        Parameters
        ----------
        annotations: dict
            Arguments passed as key=value pairs are stored inside `_custom_annotations` dictionary.
            It can be accessed by the `annotations` class property.

        """
        _check_annotations(annotations)      # Use Neo constraints to check each individual annotation item
        self._custom_annotations.update(annotations)

    def describe(self):
        """
        This method must be implemented by each analysis to save provenance information to an object that supports
        W3C PROV model.
        """
        return NotImplementedError
