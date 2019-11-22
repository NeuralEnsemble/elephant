# -*- coding: utf-8 -*-
"""
This module implements exception definitions to be used by Buffalo analysis objects.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""


class BuffaloException(Exception):
    """
    Generic class for exceptions in Buffalo objects.

    Attributes
    ----------
    source: Exception, str
        If source is an Exception object, its message will be used.
        If source is a string, this will be the exception message.
        If no source is passed, BuffaloException will use the default message '(no details)'
    """

    def __init__(self, source):
        """
        Initializes the exception class.

        Parameters
        ----------
        source: Exception, str
            Any object derived from Exception class or string to be used as message.
        """
        message = "(no details)"

        if isinstance(source, Exception):
            if isinstance(source.args[0], str):
                message = source.args[0]
        elif isinstance(source, str):
            message = source

        super().__init__(message)


class BuffaloImplementationError(BuffaloException):
    """Raised if a Buffalo object was implemented without required class attributes/variables"""


class BuffaloWrongParametersDictionary(BuffaloException):
    """Raised if `params` keyed argument is not a dictionary"""


class BuffaloMissingRequiredParameters(BuffaloException):
    """Raised if any of the required input parameters are missing"""


class BuffaloWrongParameterType(BuffaloException):
    """Raised if any of the input parameters has a wrong type"""


class BuffaloInputValidationFailed(BuffaloException):
    """Raised if input parameters are invalid for the analysis"""


class BuffaloDataValidationFailed(BuffaloException):
    """Raised if input data are invalid for the analysis"""
