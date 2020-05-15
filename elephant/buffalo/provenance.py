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

from elephant.buffalo.object_hash import BuffaloObjectHash
from elephant.buffalo.prov import BuffaloProvDocument
from elephant.buffalo.graph import BuffaloProvenanceGraph
from elephant.buffalo.ast import CallAST

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


class Provenance(object):
    """
    Class to capture and store provenance information in analysis workflows
    using Elephant.

    The class is a callable object, to be used as a decorator to every function
    of the workflow that will be tracked.

    Parameters
    ----------
    inputs : list of str
        Names of the arguments that are considered inputs to the function.
        An input is a variable or value with which the function will perform
        some computation or action. Arguments that only control the behavior
        of the function are considered parameters. The names can be for both
        positional or keyword arguments. Every argument that is not named in
        `inputs` will be considered as a parameter.

    Attributes
    ----------
    active : bool
        If True, provenance tracking is active.
        If False, provenance tracking is suspended.
        This attribute is set using the :func:`activate`/:func:`deactivate`
        interface functions.
    history : list of AnalysisStep
        All events that were tracked. Each function call is structured in a
        named tuple that stores:
        * 'function': `FunctionDefinition` named tuple;
        * 'inputs': list of the `BuffaloObjectHash` objects associated with
          every value;
        * 'params': `dict` with the positional/keyword argument names as keys,
          and their respective values passed to the function;
        * 'output': `BuffaloObjectHash` object associated with the returned
          value;
        * 'arg_map': names of the positional arguments;
        * 'kwarg_map': names of the keyword arguments;
        * 'call_ast': `ast.Call` object containing the Abstract Syntax Tree
          of the code that generated the function call.
    objects : dict
        Dictionary where the keys are the hash values of every input and
        output object tracked during the workflow. The hashes are obtained
        by the `:class:BuffaloObjectHash` class.

    Raises
    ------
    ValueError
        If `inputs` is not a list.
    """

    active = False
    history = []
    objects = dict()
    inputs = None
    calling_frame = None

    prov_document = BuffaloProvDocument()

    def __init__(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")
        self.inputs = inputs

    def _insert_static_information(self, tree, inputs, output):
        # Use a NodeVisitor to find the Call node that corresponds to the
        # current AnalysisStep. It will fetch static relationships between
        # variables and attributes, and link to the inputs and outputs of the
        # function
        ast_visitor = CallAST(self, inputs, output)
        ast_visitor.visit(tree)

    def __call__(self, function):

        @wraps(function)
        def wrapped(*args, **kwargs):
            function_output = function(*args, **kwargs)

            # If capturing provenance...
            if Provenance.active:

                # 1. Capture Abstract Syntax Tree (AST) of the call to the
                # function

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
                #    function, are transformed in the hashable type
                #    `BuffaloObjectHash`. Inputs are defined by the parameter
                #    `inputs` when initializing the class, and stored as the
                #    class attribute `inputs`

                parameters = {}
                inputs = {}
                for key, value in input_data.items():
                    if key in self.inputs:
                        inputs[key] = self.add(value)
                    else:
                        parameters[key] = value

                # 5. Create hashable `BuffaloObjectHash` for the output

                output = self.add(function_output)

                # 6. Analyze AST and fetch static relationships in the
                # input/output and other variables/objects in the script
                self._insert_static_information(tree, inputs, output)

                # 7. Create tuple with the analysis step information

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
        """
        Returns the W3C PROV graph representation of the captured provenance
        information.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments forwarded to the
            :func:`BuffaloProvDocument.get_dot_graph` function.

        Returns
        -------
        pydot.Dot
            Dot notation graph representing provenance information in W3C PROV
            standard.

        See Also
        --------
        elephant.buffalo.prov.BuffaloProvDocument

        """
        return cls.prov_document.get_dot_graph(**kwargs)

    @classmethod
    def print_graph(cls):
        graph = BuffaloProvenanceGraph(cls.objects, cls.history)
        graph.print_graph()

    @classmethod
    def add(cls, obj):
        """
        Hashes and insert a given Python object into the internal dictionary
        (:attr:`objects`), if the hash is new.

        Parameters
        ----------
        obj : object
            Python object to be added to `objects`.

        Returns
        -------
        BuffaloObjectHash
            Hash to the object that was added.

        """
        object_hash = BuffaloObjectHash(obj)
        if object_hash not in cls.objects:
            cls.objects[object_hash] = object_hash
        return cls.objects[object_hash]

    @classmethod
    def add_script_variable(cls, name):
        """
        Hashes an object stored as a variable in the namespace where provenance
        tracking was activated. Then add the hash to the internal dictionary.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        object
            Python object referenced by `name`.
        object_hash
            `BuffaloObjectHash` instance with the hash of the object.

        """
        instance = cls.calling_frame.f_locals[name]
        object_hash = cls.add(instance)
        return instance, object_hash


##############################################################################
# Interface functions
##############################################################################

def activate():
    """
    Activates provenance tracking within Elephant.
    """
    # To access variables in the same namespace where the function is called,
    # the previous frame in the stack need to be saved
    if Provenance.calling_frame is None:
        Provenance.calling_frame = inspect.currentframe().f_back

    Provenance.active = True


def deactivate():
    """
    Deactivates provenance tracking within Elephant.
    """
    Provenance.active = False


def print_history():
    """
    Print all steps in the provenance track.
    """
    pprint(Provenance.history)


def print_graph():
    Provenance.print_graph()


def save_prov_graph(filename, **kwargs):
    """
    Saves a PNG file with the provenance track described using the W3C PROV
    model.

    Parameters
    ----------
    filename : str or Path
        Destination of the PROV graph.
    kwargs : dict
        Keyword arguments forwarded to the
        :func:`BuffaloProvDocument.get_dot_graph` function.
    """
    dot = Provenance.get_prov_graph(**kwargs)
    dot.write_png(filename)
