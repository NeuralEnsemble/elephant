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
                         BuffaloInputValidationFailed, BuffaloImplementationError, BuffaloDataValidationFailed)


class Analysis(object):
    """
    This is the superclass for the Buffalo analysis objects.

    It supports basic input validation and storage of basic provenance information.
    Input parameters are passed as the `params` keyed argument in class initialization.
    Any other initialization argument/keyed-argument is treated as input data.

    Attributes
    ----------
        args: list
            Input data as positional arguments

        kwargs: dict
            Input data as keyed arguments

        params: dict
            Input parameters for the analysis


    Methods
    -------
    annotate
        Stores custom values as items in a dictionary inside the class

    get_input_parameter
        Quickly retrieves the value of an input parameter, if it is available


    Properties
    ----------
    input_parameters: dict
        The analysis input parameters that are passed during class initialization (`params` keyed argument)

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

    _validate_parameters: function
        This is an optional function that is used to validate input parameters values.
        It is executed after the basic type validation of input parameters was done.

    _validate_data: function
        This is an optional function that is used to validate input data specifically.
        It is executed after input parameters are validated.
    """

    _input_parameters = None          # Internal copy of the dictionary of input parameters for the analysis

    _ts_create = None                 # UTC time of instance creation
    _ts_execute = None                # UTC time of analysis execution
    _ts_end = None                    # UTC time of analysis execution end

    _annotations = {}                 # Dictionary with custom annotations

    _name = None                      # Name of the analysis
    _cno_name = None                  # Name according to the CNO ontology TODO: check against official schema
    _description = None               # Human readable description of the object

    _required_params = []    # List of keys that must be present in the 'params' dictionary during class initialization

    # Dictionary with tuples describing the accepted types for each item in the 'params' dictionary. May include
    # additional, non required parameters known in advance, and that can be automatically checked
    _required_types = {}

    def __init__(self, *args, params=None, **kwargs):
        """
        Initializes the Analysis object.
        Automatic validation is performed, by checking if all required parameters were passed and if inputs are correct.
        Finally, it initializes basic provenance information.

        Parameters
        ----------
        params : dict, None
            All input parameters relevant for the analysis.
            An input parameter is defined as any input that controls the analysis, but that is not the data that is the
            subject of the analysis.
            If no parameters are required, None should be passed.

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

        BuffaloImplementationError
            If there was an error in the implementation of `_required_params` or `_required_types`, or the required
            class attributes `name` and `description`

        NotImplementedError
            If `_process` is not overridden by the child class

        """
        self._ts_create = datetime.utcnow()    # Stores the timestamp when the instance was created

        if params is None:
            params = dict()

        # Check required class attributes are present
        if not (isinstance(self.name, str) and len(self.name)):
            raise BuffaloImplementationError("Must define 'name' attribute with non-empty string")
        if not (isinstance(self.description, str) and len(self._description)):
            raise BuffaloImplementationError("Must define 'description' attribute with non-empty string")

        # Check if optional class attributes are valid if present
        if self.cno_name is not None:
            if not (isinstance(self.cno_name, str) and len(self.cno_name)):
                raise BuffaloImplementationError("CNO name must be non-empty string")

        # Check that protected variables are correct
        if not isinstance(self._required_params, list):
            raise BuffaloImplementationError("'_required_params' class variable must be 'list'")
        if not isinstance(self._required_types, dict):
            raise BuffaloImplementationError("'_required_types' class variable must be 'dict'")

        # Check that the analysis parameters were passed, and store them in `_input_parameters`
        if not isinstance(params, dict):
            raise BuffaloWrongParametersDictionary("The analysis parameters must be passed as a dictionary in the "
                                                   "'params' keyed argument")
        self._input_parameters = deepcopy(params)

        # Verify that analysis parameters are present and valid.
        # This is a type validation, not value validation.
        # Any value in `_required_params` that is not a string is treated as an implementation error
        # If any key of `_required_types` has a value that is not a tuple, this is also treated as implementation error.
        try:
            self._has_required_input_parameters()
        except NameError as error:
            raise BuffaloMissingRequiredParameters(error)
        except ValueError as error:
            raise BuffaloImplementationError(error)

        try:
            self._input_parameters_are_valid()
        except TypeError as error:
            raise BuffaloWrongParameterType(error)
        except ValueError as error:
            raise BuffaloImplementationError(error)

        # Performs the optional validation on the input parameters
        try:
            self._validate_parameters()
        except ValueError as error:
            raise BuffaloInputValidationFailed(error)

        # Performs the optional validation on the input data
        try:
            self._validate_data(*args, **kwargs)
        except (NameError, ValueError) as error:
            raise BuffaloDataValidationFailed(error)

        # Run the analysis for the provided data
        # 'params' are removed as any other arg/kwarg is supposed to be data, which is the only input for the '_process'
        # function.
        if 'params' in kwargs:
            kwargs.pop('params')

        self._ts_execute = datetime.utcnow()   # Stores the timestamp when '_process' function is called
        self._process(*args, **kwargs)         # Run the analysis on the given input data
        self._ts_end = datetime.utcnow()       # Stores the timestamp when analysis finished

    def _has_required_input_parameters(self):
        """
        Internal function that checks if the parameters dictionary stored inside the class has the necessary items.

        Raises
        -------
        NameError
            If any of the parameters listed in `_required_params` are missing.

        ValueError
            If `_required_params` contains an item that is not a string.
        """
        parameters = list(self.input_parameters.keys())
        if len(self._required_params):
            if len(parameters) >= len(self._required_params):
                for required in self._required_params:
                    if not isinstance(required, str):
                        raise ValueError(f"Required params must contain strings. '{type(required)}' was passed")
                    if required not in parameters:
                        raise NameError(f"Missing parameter '{required}'")
            else:
                raise NameError(f"Parameters '{', '.join(self._required_params)}' are expected.")

    def _input_parameters_are_valid(self):
        """
        Internal function that checks if the input parameters types are acceptable.

        Raises
        -------
        TypeError
            If the type of the parameter is not any listed for that key in `_required_types`

        ValueError
            If the value of the item in `_required_types` dict is not a tuple
        """
        for key, value in self.input_parameters.items():
            if key in self._required_types.keys():
                if not isinstance(self._required_types[key], tuple):
                    raise ValueError(f"Values for dictionary item '{key}' must be passed as a tuple")
                if type(value) not in self._required_types[key]:
                    raise TypeError(f"Parameter '{key}' should be one of: "
                                    ', '.join(map(str, self._required_types[key])))

    def _validate_parameters(self):
        """
        This function performs a check in the input parameters, to assert that they are correct
        according to the analysis algorithm.
        This is optional. If overridden, the function should raise ValueError exceptions when a validation fails.

        Raises
        -------
            ValueError
                If input parameters are not correct
        """
        pass

    def _validate_data(self, *args, **kwargs):
        """
        This function performs a check in the input data, to assert that they are correct
        according to the analysis algorithm.
        This is optional. If overridden, the function should raise ValueError or NameError exceptions when a validation
        fails.

        Raises
        -------
            ValueError
                If input parameters are not correct

            NameError
                If missing input data
        """
        pass

    def _process(self, *args, **kwargs):
        """
        This is the main logic for the analysis.
        Input data is passed from the class initialization.
        """
        raise NotImplementedError

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
        return self._annotations

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
    def input_parameters(self):
        """
        All parameters that were given for the analysis.
        """
        return self._input_parameters

    def annotate(self, **annotations):
        """
        Inserts annotations in the object.

        Parameters
        ----------
        annotations: dict
            Arguments passed as key=value pairs are stored inside `_annotations` dictionary.
            It can be accessed by the `annotations` class property.

        """
        _check_annotations(annotations)      # Use Neo constraints to check each individual annotation item
        self._annotations.update(annotations)

    def get_input_parameter(self, parameter):
        """
        Fetches the value of any input parameter passed using the `params` argument.
        If that parameter is not found (optional), returns None

        Parameters
        ----------
        parameter: str
            Name of the parameter

        Returns
        -------
            object, None
                Value of the parameter.
                None if the parameter is not found in the `_input_parameters` dictionary.
        """
        if parameter in self.input_parameters:
            return self.input_parameters[parameter]
        return None

    def describe(self):
        """
        This method must be implemented by each analysis to save provenance information to an object that supports
        W3C PROV model.
        """
        #TODO: implement according to `prov` package objects
        return NotImplementedError
