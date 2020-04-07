# -*- coding: utf-8 -*-
"""
This module implements a provenance object that supports provenance capture
using the W3C PROV standard.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from functools import wraps
import inspect
import ast
from collections import namedtuple
from collections.abc import Iterable

from elephant.buffalo.prov import BuffaloProvDocument
from elephant.buffalo.graph import BuffaloProvGraph

from pprint import pprint


AnalysisStep = namedtuple('AnalysisStep', ('function',
                                           'input',
                                           'params',
                                           'output'))

FunctionDefinition = namedtuple('FunctionDefinition', ('name',
                                                       'module',
                                                       'qualname',
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
        if "__hash__" in dir(self._value):
            value = self._value if self._value.__hash__ is not None \
                else self.__repr__()
        else:
            value = self.__repr__()
        return hash((self._id, self._type, value))

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
        if isinstance(self._value, Iterable):
            value = "(id: {})".format(self._id)
        else:
            value = self._value.__repr__()
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

            # frame = inspect.getouterframes(inspect.currentframe())[1]
            # tree = ast.parse(frame.code_context[0].strip())
            # print(ast.dump(tree))

            # If capturing provenance...
            if Provenance.active:

                # 1. Extract function name
                function_name = FunctionDefinition(
                    function.__name__, function.__module__,
                    function.__qualname__, None)

                # 2. Extract parameters passed to function and store in
                #    `input_data` dictionary

                # 2.1 Positional arguments
                input_data = {}
                input_syntax = {}
                params = tuple(inspect.signature(function).parameters.keys())
                for arg_id, arg_val in enumerate(args):
                    arg_name = params[arg_id]
                    input_data[arg_name] = arg_val

                # 2.2 Add keyword arguments
                input_data.update(kwargs)

                # 3. Create parameters/input description for the graph

                parameters = {}
                inputs = {}
                for key, value in input_data.items():
                    if key in self.inputs:
                        inputs[key] = self.add(value)
                    else:
                        parameters[key] = value

                output = self.add(function_output)

                # 4. Create tuple with the analysis step information
                step = AnalysisStep(function_name, inputs, parameters, output)

                # 5. Add to history graph / PROV document
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
