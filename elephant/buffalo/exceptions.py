# -*- coding: utf-8 -*-
"""
This module implements exceptions definitions to be used by Buffalo analysis objects.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""


class BuffaloWrongParametersDictionary(Exception):
    """Raised if `params` keyed argument is not a dictionary"""


class BuffaloMissingRequiredParameters(Exception):
    """Raised if any of the required input parameters are missing"""


class BuffaloWrongParameterType(Exception):
    """Raised if any of the input parameters has a wrong type"""


class BuffaloInputValidationFailed(Exception):
    """Raised if input parameters or data are invalid for the analysis"""
