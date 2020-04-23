# -*- coding: utf-8 -*-
"""
This module implements a provenance object that supports provenance capture
using the W3C PROV standard.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from functools import wraps
import inspect
import joblib
import ast
from collections import namedtuple

from elephant.buffalo.prov import BuffaloProvDocument
from elephant.buffalo.graph import BuffaloProvGraph

from pprint import pprint


AnalysisStep = namedtuple('AnalysisStep', ('function',
                                           'input',
                                           'params',
                                           'output',
                                           'arg_map',
                                           'kwarg_map',
                                           'call_ast'))


FunctionDefinition = namedtuple('FunctionDefinition', ('name',
                                                       'module',
                                                       'version'))


class BuffaloProvObject(object):

    _id = None
    _type = None
    _value = None

    @staticmethod
    def _get_object_info(obj):
        class_name = "{}.{}".format(type(obj).__module__,
                                    type(obj).__name__)
        return id(obj), class_name, obj

    def __init__(self, obj):
        self._id, self._type, self._value = self._get_object_info(obj)

    def __hash__(self):
        return hash((self._id, self._type, joblib.hash(self._value)))

    def __eq__(self, other):
        if isinstance(other, BuffaloProvObject):
            return hash(self) == hash(other)
        else:
            object_id, class_name, value = self._get_object_info(other)
            if value is self._value:
                return True
            else:
                return (object_id, class_name, value) == (
                    self._id, self._type, self._value
                )

    def __repr__(self):
        return "{}: {} = {}".format(self._id, self._type, self._value)

    def get_md_string(self):
        value = "(id: {})".format(self._id)
        return '{}["{}<br>{}"]\n'.format(hash(self), self._type, value)


class Provenance(object):

    active = False
    history = []
    objects = dict()
    inputs = None

    prov_document = BuffaloProvDocument()

    def __init__(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")
        self.inputs = inputs

    def __call__(self, function):

        @wraps(function)
        def wrapped(*args, **kwargs):
            function_output = function(*args, **kwargs)

            # If capturing provenance...
            if Provenance.active:

                # 1. Capture AST of the call to the function

                frame = inspect.getouterframes(inspect.currentframe())[1]
                tree = ast.parse(frame.code_context[0].strip())

                # 2. Extract function name and information
                # TODO: fetch version information

                function_name = FunctionDefinition(
                    function.__name__, function.__module__, None)

                # 3. Extract parameters passed to function and store in
                #    `input_data` dictionary
                #    Two separate lists with the names according to the
                #    arg/kwarg order are also constructed, to map to the
                #    `args` and `keywords` fields of AST nodes

                # 3.1 Positional arguments

                input_data = {}
                input_args_names = []
                params = tuple(inspect.signature(function).parameters.keys())
                for arg_id, arg_val in enumerate(args):
                    arg_name = params[arg_id]
                    input_data[arg_name] = arg_val
                    input_args_names.append(arg_name)

                # 3.2 Add keyword arguments

                input_kwargs_names = []
                for kwarg_id, kwarg_name in enumerate(kwargs.keys()):
                    input_data[kwarg_name] = kwargs[kwarg_name]
                    input_kwargs_names.append(kwarg_name)

                # 4. Create parameters/input descriptions for the graph
                #    Here the inputs, but not the parameters passed to the
                #    function are transformed in the hashable type
                #    BuffaloProvObject. Inputs are defined as the parameter
                #    `inputs` when initializing the class, and stored as the
                #    class attribute `inputs`

                parameters = {}
                inputs = {}
                for key, value in input_data.items():
                    if key in self.inputs:
                        inputs[key] = self.add(value)
                    else:
                        parameters[key] = value

                # 5. Create hashable BuffaloProvObject for the output

                output = self.add(function_output)

                # TODO: do static analysis
                # self._insert_static_information(tree, inputs, output)

                # 6. Create tuple with the analysis step information

                step = AnalysisStep(function_name, inputs, parameters, output,
                                    input_args_names, input_kwargs_names,
                                    tree)

                # 7. Add to history graph / PROV document

                Provenance.history.append(step)
                Provenance.prov_document.add_step(step)

            return function_output

        return wrapped

    @classmethod
    def get_prov_graph(cls, **kwargs):
        return cls.prov_document.get_dot_graph(**kwargs)

    @classmethod
    def print_graph(cls):
        graph = BuffaloProvGraph(cls.objects, cls.history)
        graph.print_graph()

    @classmethod
    def add(cls, obj):
        prov_object = BuffaloProvObject(obj)
        if prov_object not in cls.objects:
            cls.objects[prov_object] = prov_object
        return cls.objects[prov_object]


# Interface functions

def activate():
    Provenance.active = True


def deactivate():
    Provenance.active = False


def print_history():
    pprint(Provenance.history)


def print_graph():
    Provenance.print_graph()


def save_prov_graph(**kwargs):
    dot = Provenance.get_prov_graph(**kwargs)
    dot.write_png("prov_graph.png")
