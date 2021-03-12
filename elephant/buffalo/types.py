"""
This module defines named tuples that are used to structure the informatoin
throughout Buffalo.

:copyright: Copyright 2014-2021 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from collections import namedtuple


# Named tuples to store provenance information of each call
# In `AnalysisStep`, the `function` value is a `FunctionInfo` named tuple

AnalysisStep = namedtuple('AnalysisStep', ('function',
                                           'input',
                                           'params',
                                           'output',
                                           'arg_map',
                                           'kwarg_map',
                                           'call_ast',
                                           'code_statement',
                                           'time_stamp_start',
                                           'time_stamp_end',
                                           'return_targets',
                                           'vis'))


FunctionInfo = namedtuple('FunctionInfo', ('name', 'module', 'version'))


# Named tuple to store variable arguments

VarArgs = namedtuple('VarArgs', 'args')


# Named tuples to store hashes

ObjectInfo = namedtuple('ObjectInfo', ('hash', 'type', 'id', 'details'))

FileInfo = namedtuple('FileInfo', ('hash', 'hash_type', 'path'))
