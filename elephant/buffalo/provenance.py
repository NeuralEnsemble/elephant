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
import datetime

from elephant.buffalo.object_hash import BuffaloObjectHash
from elephant.buffalo.graph import BuffaloProvenanceGraph
from elephant.buffalo.ast_analysis import CallAST
from elephant.buffalo.code_lines import SourceCodeAnalyzer

from os.path import splitext

from pprint import pprint

# Python 2.7 compatibility
if 'signature' in dir(inspect):
    signature = inspect.signature
else:
    import funcsigs
    signature = funcsigs.signature


AnalysisStep = namedtuple('AnalysisStep', ('function',
                                           'input',
                                           'params',
                                           'output',
                                           'arg_map',
                                           'kwarg_map',
                                           'call_ast',
                                           'code_statement',
                                           'time_stamp',
                                           'return_targets'))


FunctionDefinition = namedtuple('FunctionDefinition', ('name',
                                                       'module',
                                                       'version'))


VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
VarArgs = namedtuple('VarArgs', 'value')


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
        * 'output': list of the `BuffaloObjectHash` objects associated with
          the returned values;
        * 'arg_map': names of the positional arguments;
        * 'kwarg_map': names of the keyword arguments;
        * 'call_ast': `ast.AST` object containing the Abstract Syntax Tree
          of the code that generated the function call.
        * 'code_statement': `str` with the code statement calling the function.
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
    objects = set()
    inputs = None

    calling_frame = None
    source_code = None
    frame_ast = None
    source_lineno = None
    source_file = None
    source_name = None
    code_analyzer = None

    def __init__(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")
        self.inputs = inputs

    def _insert_static_information(self, tree, inputs, output):
        # Use a NodeVisitor to find the Call node that corresponds to the
        # current AnalysisStep. It will fetch static relationships between
        # variables and attributes, and link to the inputs and outputs of the
        # function
        ast_visitor = CallAST(self)
        ast_visitor.visit(tree)

    def __call__(self, function):

        @wraps(function)
        def wrapped(*args, **kwargs):

            # For functions that are used inside other decorated functions, or
            # recursively, check if the calling frame is the one being
            # tracked. We do this by fetching the calling line number if
            # this comes from the calling frame. Otherwise, the line number
            # will be None, and therefore the provenance tracking loop will
            # be skipped.
            # For list comprehensions, we need to check the frame above, as
            # this creates a frame named <listcomp>
            lineno = None
            if Provenance.active:
                try:
                    frame = inspect.currentframe().f_back
                    frame_info = inspect.getframeinfo(frame)
                    function_name = frame_info.function
                    if function_name == '<listcomp>':
                        while function_name == '<listcomp>':
                            frame = frame.f_back
                            frame_info = inspect.getframeinfo(frame)
                            function_name = frame_info.function

                    if (frame_info.filename == self.source_file and
                            frame_info.function == self.source_name):
                        lineno = frame.f_lineno
                finally:
                    del frame_info
                    del frame

            # Call the function
            function_output = function(*args, **kwargs)
            time_stamp = datetime.datetime.utcnow().isoformat()

            # If capturing provenance...
            if Provenance.active and lineno is not None:

                # 1. Capture Abstract Syntax Tree (AST) of the call to the
                # function. We need to check the source code in case the
                # call spans multiple lines. In this case, we fetch the
                # full statement.
                source_line = self.code_analyzer.extract_multiline_statement(
                    lineno)
                ast_tree = ast.parse(source_line)

                # 2. Check if there is an assignment to one or more variables
                # This will be used to identify if there are multiple output
                # nodes. This is needed because just checking if
                # `function_output` is tuple does not work if the function is
                # actually returning a tuple
                return_targets = []
                if isinstance(ast_tree.body[0], ast.Assign):
                    assign_target = ast_tree.body[0].targets[0]
                    if isinstance(assign_target, ast.Tuple):
                        return_targets = [target.id for target in
                                          assign_target.elts]
                    elif isinstance(assign_target, ast.Name):
                        return_targets = [assign_target.id]
                    else:
                        raise ValueError("Unknown assign target!")

                # 3. Extract function name and information
                # TODO: fetch version information

                function_name = FunctionDefinition(
                    function.__name__, function.__module__, None)

                # 4. Extract parameters passed to function and store in
                #    `input_data` dictionary
                #    Two separate lists with the names according to the
                #    arg/kwarg order are also constructed, to map to the
                #    `args` and `keywords` fields of AST nodes

                input_data = {}
                input_args_names = []
                input_kwargs_names = []

                func_parameters = inspect.signature(function).bind(*args,
                                                                   **kwargs)
                for arg_name, arg_value in func_parameters.arguments.items():
                    cur_parameter = func_parameters.signature.parameters[
                        arg_name]
                    if cur_parameter.kind != VAR_POSITIONAL:
                        input_data[arg_name] = arg_value
                    else:
                        input_data[arg_name] = VarArgs(arg_value)
                    if arg_name in kwargs:
                        input_kwargs_names.append(arg_name)
                    else:
                        input_args_names.append(arg_name)

                # 5. Create parameters/input descriptions for the graph
                #    Here the inputs, but not the parameters passed to the
                #    function, are transformed in the hashable type
                #    `BuffaloObjectHash`. Inputs are defined by the parameter
                #    `inputs` when initializing the class, and stored as the
                #    class attribute `inputs`

                parameters = {}
                inputs = {}
                for key, value in input_data.items():
                    if key in self.inputs:
                        if isinstance(value, VarArgs):
                            var_input_list = []
                            for var_arg in value.value:
                                var_input_list.append(self.add(var_arg))
                            inputs[key] = VarArgs(tuple(var_input_list))
                        else:
                            inputs[key] = self.add(value)
                    else:
                        parameters[key] = value

                # 6. Create hashable `BuffaloObjectHash` for the output
                # objects to follow individual returns
                outputs = {}
                if len(return_targets) > 1:
                    for index, item in enumerate(function_output):
                        outputs[index] = self.add(item)
                else:
                    outputs[0] = self.add(function_output)

                # 7. Analyze AST and fetch static relationships in the
                # input/output and other variables/objects in the script
                self._insert_static_information(ast_tree, inputs, outputs)

                # 8. Create tuple with the analysis step information

                step = AnalysisStep(function_name, inputs, parameters, outputs,
                                    input_args_names, input_kwargs_names,
                                    ast_tree, source_line, time_stamp,
                                    return_targets)

                # 7. Add to history
                # The history will be the base to generate the graph / PROV
                # document
                Provenance.history.append(step)

            return function_output

        return wrapped

    @classmethod
    def set_calling_frame(cls, frame):
        cls.calling_frame = frame

        cls.source_file = inspect.getfile(frame)
        cls.source_name = inspect.getframeinfo(frame).function

        if cls.source_name == '<module>':
            cls.source_lineno = 1
        else:
            cls.source_lineno = inspect.getlineno(frame)

        code_lines = inspect.getsourcelines(frame)[0]

        # Clean decorators
        cur_line = 0
        while code_lines[cur_line].strip().startswith('@'):
            cur_line += 1

        cls.source_code = code_lines[cur_line:]

        cls.frame_ast = ast.parse("".join(cls.source_code).strip())

        cls.code_analyzer = SourceCodeAnalyzer(cls.source_code,
                                               cls.frame_ast,
                                               cls.source_lineno,
                                               cls.source_name)

    @classmethod
    def get_prov_graph(cls, **kwargs):
        """
        Returns the W3C PROV graph representation of the captured provenance
        information.
        """
        raise NotImplementedError

    @classmethod
    def save_graph(cls, filename, show=False):
        """
        Save an interactive graph with the provenance track.

        Parameters
        ----------
        filename : str
            HTML file to save the graph.

        Raises
        ------
        ValueError
            If `filename` is not an HTML file.

        """
        name, ext = splitext(filename)
        if not ext.lower() in ['.html', '.htm']:
            raise ValueError("Filename must have HTML extension (.html, "
                             ".htm)!")

        graph = BuffaloProvenanceGraph()
        for step in Provenance.history:
            graph.add_step(step)
        graph.to_pyvis(filename, show=show)

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
            cls.objects.add(object_hash)
        return object_hash

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
    # the previous frame in the stack need to be saved. We also extract
    # extended information regarding the frame code.
    Provenance.set_calling_frame(inspect.currentframe().f_back)
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


def save_graph(filename, show=False):
    """
    Saves an interactive graph to disk.

    Parameters
    ----------
    filename : str
        Destination of the saved graph (HTML file).
    show : bool
        If True, displays the graph in the browser after saving.
        Default: False.
    """
    Provenance.save_graph(filename, show=show)
