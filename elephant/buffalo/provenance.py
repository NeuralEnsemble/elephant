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
import pickle

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
VarArgs = namedtuple('VarArgs', 'args')


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

    Raises
    ------
    ValueError
        If `inputs` is not a list.
    """

    active = False
    history = []
    inputs = None

    calling_frame = None
    source_code = None
    frame_ast = None
    source_lineno = None
    source_file = None
    source_name = None
    code_analyzer = None

    def __init__(self, inputs, container_output=False):
        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")
        self.inputs = inputs
        self.container_output = container_output
        self.initialized = False
        self.has_return = True

    def _insert_static_information(self, tree, function, time_stamp):
        # Use a NodeVisitor to find the Call node that corresponds to the
        # current AnalysisStep. It will fetch static relationships between
        # variables and attributes, and link to the inputs and outputs of the
        # function
        ast_visitor = CallAST(self, function, time_stamp)
        ast_visitor.visit(tree)

    def _analyze_function(self, function):
        try:
            source_code = inspect.getsourcelines(function)[0]
            source_code = source_code[1:]    # Strip decorator
            code_string = "".join(source_code)
            ast_tree = ast.parse(code_string)

            has_return = False
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Return):
                    has_return = True
                    break
        except:
            has_return = True

        print(function, has_return)
        return True, has_return

    def __call__(self, function):

        @wraps(function)
        def wrapped(*args, **kwargs):

            # Call the function
            function_output = function(*args, **kwargs)
            time_stamp = datetime.datetime.utcnow().isoformat()

            # If capturing provenance...
            if Provenance.active:

                # In the first run, analyze the function code to check if there
                # are returns. If no return, we won't track the output as this
                # is automatically the None object
                if not self.initialized:
                    self.initialized, self.has_return = \
                        self._analyze_function(function)

                # For functions that are used inside other decorated functions,
                # or recursively, check if the calling frame is the one being
                # tracked. We do this by getting the line number if this comes
                # from the calling frame. Otherwise, the line number will be
                # None, and the provenance tracking block will be skipped.
                # For list comprehensions, we need to check the frame above,
                # as this creates a frame named <listcomp>
                lineno = None
                try:
                    frame = inspect.currentframe().f_back
                    frame_info = inspect.getframeinfo(frame)
                    function_name = frame_info.function
                    if function_name == '<listcomp>':
                        while function_name == '<listcomp>':
                            frame = frame.f_back
                            frame_info = inspect.getframeinfo(frame)
                            function_name = frame_info.function
                    elif function_name == 'wrapper':
                        frame = frame.f_back
                        frame_info = inspect.getframeinfo(frame)
                        function_name = frame_info.function

                    if (frame_info.filename == self.source_file and
                            frame_info.function == self.source_name):
                        lineno = frame.f_lineno
                finally:
                    del frame_info
                    del frame

                # Capture provenance information
                if lineno is not None:

                    # 1. Capture Abstract Syntax Tree (AST) of the call to the
                    # function. We need to check the source code in case the
                    # call spans multiple lines. In this case, we fetch the
                    # full statement.
                    source_line = \
                        self.code_analyzer.extract_multiline_statement(lineno)
                    ast_tree = ast.parse(source_line)
                    print(source_line)

                    # 2. Check if there is an assignment to one or more
                    # variables. This will be used to identify if there are
                    # multiple output nodes. This is needed because just
                    # checking if `function_output` is tuple does not work if
                    # the function is actually returning a tuple.
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

                    try:
                        module = function.__module__
                    except:
                        module = None
                    function_name = FunctionDefinition(
                        function.__name__, module, None)

                    # 4. Extract parameters passed to the function and store
                    # in `input_data` dictionary. Two separate lists with the
                    # names according to the arg/kwarg order are also
                    # constructed, to map to the `args` and `keywords` fields
                    # of the AST nodes

                    input_data = {}
                    input_args_names = []
                    input_kwargs_names = []

                    try:
                        func_parameters = \
                            inspect.signature(function).bind(*args, **kwargs)

                        for arg_name, arg_value in \
                                func_parameters.arguments.items():
                            cur_parameter = \
                                func_parameters.signature.parameters[arg_name]

                            if cur_parameter.kind != VAR_POSITIONAL:
                                input_data[arg_name] = arg_value
                            else:
                                # Variable positional arguments are stored as
                                # the namedtuple VarArgs.
                                input_data[arg_name] = VarArgs(arg_value)

                            if arg_name in kwargs:
                                input_kwargs_names.append(arg_name)
                            else:
                                input_args_names.append(arg_name)

                    except ValueError:
                        # Can't inspect signature. Append args/kwargs by
                        # order
                        for arg_index, arg in enumerate(args):
                            input_data[arg_index] = arg
                            input_args_names.append(arg_index)

                        kwarg_start = len(input_data)
                        for kwarg_index, kwarg in enumerate(kwargs,
                                                            start=kwarg_start):
                            input_data[kwarg_index] = kwarg
                            input_kwargs_names.append(kwarg_index)

                    # 5. Create parameters/input descriptions for the graph.
                    # Here the inputs, but not the parameters passed to the
                    # function, are transformed in the hashable type
                    # `BuffaloObjectHash`. Inputs are defined by the parameter
                    # `inputs` when initializing the class, and stored as the
                    # class attribute `inputs`.

                    parameters = {}
                    inputs = {}
                    for key, input_value in input_data.items():
                        if key in self.inputs:
                            if isinstance(input_value, VarArgs):
                                var_input_list = []
                                for var_arg in input_value.args:
                                    var_input_list.append(
                                        BuffaloObjectHash(var_arg).info())
                                inputs[key] = VarArgs(tuple(var_input_list))
                            else:
                                inputs[key] = \
                                    BuffaloObjectHash(input_value).info()
                        else:
                            parameters[key] = input_value

                    # 6. Create hashable `BuffaloObjectHash` for the output
                    # objects to follow individual returns
                    outputs = {}
                    if self.has_return:
                        if len(return_targets) > 1:
                            for index, item in enumerate(function_output):
                                outputs[index] = BuffaloObjectHash(item).info()
                        else:
                            outputs[0] = BuffaloObjectHash(function_output).info()

                    # 7. Analyze AST and fetch static relationships in the
                    # input/output and other variables/objects in the script
                    self._insert_static_information(ast_tree,
                                                    function_name.name,
                                                    time_stamp)

                    # 8. Create tuple with the analysis step information.
                    step = AnalysisStep(function_name,
                                        inputs,
                                        parameters,
                                        outputs,
                                        input_args_names, input_kwargs_names,
                                        ast_tree, source_line, time_stamp,
                                        return_targets)

                    # 9. Add to the history.
                    # The history will be the base to generate the graph and
                    # PROV document.
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
    def get_graph(cls, history=None):
        if history is None:
            history = Provenance.history
        graph = BuffaloProvenanceGraph()

        for step in history:
            graph.add_step(step)
        return graph

    @classmethod
    def dump_history(cls, filename):
        pickle.dump(Provenance.history, open(filename, "wb"))

    @classmethod
    def save_graph(cls, filename, source=None, show=False):
        """
        Save an interactive graph with the provenance track.

        Parameters
        ----------
        filename : str
            HTML file to save the graph.
        show : bool, optional
            If True, shows the graph in the browser after saving.
            Default: False.

        Raises
        ------
        ValueError
            If `filename` is not an HTML file.

        """
        name, ext = splitext(filename)
        if not ext.lower() in ['.html', '.htm']:
            raise ValueError("Filename must have HTML extension (.html, "
                             ".htm)!")

        if source is None:
            print("Getting graph")
            source = cls.get_graph()

        print("Converting graph")
        source.to_pyvis(filename, show=show)

    @classmethod
    def get_script_variable(cls, name):
        return cls.calling_frame.f_locals[name]


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


def save_graph(filename, source=None, show=False):
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
    Provenance.save_graph(filename, source=None, show=show)


def get_graph(history=None):
    return Provenance.get_graph(history)


def dump_provenance(filename):
    Provenance.dump_history(filename)

