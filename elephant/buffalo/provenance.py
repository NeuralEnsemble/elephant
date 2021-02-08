"""
This module implements a function decorator to support provenance capture
during the execution of analysis scripts using Elephant.

:copyright: Copyright 2014-2021 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from functools import wraps
import inspect
import ast
from collections import namedtuple
import datetime

from elephant.buffalo.object_hash import (BuffaloObjectHash, BuffaloFileHash,
                                          hash_memoizer)
from elephant.buffalo.graph import BuffaloProvenanceGraph
from elephant.buffalo.ast_analysis import _CallAST
from elephant.buffalo.code_lines import _BuffaloCodeAnalyzer

from os.path import splitext

from pprint import pprint


# Python 2.7 compatibility
if 'signature' in dir(inspect):
    signature = inspect.signature
else:
    import funcsigs
    signature = funcsigs.signature

import dill

AnalysisStep = namedtuple('AnalysisStep', ('function',
                                           'input',
                                           'params',
                                           'output',
                                           'arg_map',
                                           'kwarg_map',
                                           'call_ast',
                                           'code_statement',
                                           'time_stamp',
                                           'return_targets',
                                           'vis'))


FunctionInfo = namedtuple('FunctionInfo', ('name', 'module', 'version'))


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
        `inputs`, `file_input` or `file_output` will be considered as a
        parameter.
    file_input : list of str, optional
        Names of the arguments that represent file(s) read from the disk by
        the function. Their SHA256 hashes will be computed and stored.
    file_output : list of str, optional
        Names of the arguments that represent file(s) write to the disk by
        the function. The SHA256 hashes will be computed and stored.

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
        * 'function': `FunctionInfo` named tuple;
        * 'inputs': list of the `ObjectInfo` or `FileInfo` named tuples
          associated with every input value to the function;
        * 'params': `dict` with the positional/keyword argument names as keys,
          and their respective values passed to the function;
        * 'output': list of the `ObjectInfo` or `FileInfo` named tuples
          associated with the values returned by the function or files written
          to the disk;
        * 'arg_map': names of the positional arguments;
        * 'kwarg_map': names of the keyword arguments;
        * 'call_ast': `ast.AST` object containing the Abstract Syntax Tree
          of the code that generated the function call.
        * 'code_statement': `str` with the code statement calling the function.
        * 'time_stamp': `datetime` with the execution time of the statement;
        * 'return_targets': names of the variables that store the function
          output(s) in the source code;
        * 'vis': tuple of integers, with the X/Y positions for the function
          node in the visualization graph.

    Raises
    ------
    ValueError
        If `inputs` is not a list.
    """

    active = False
    history = []

    inputs = None
    file_inputs = None
    file_outputs = None

    calling_frame = None
    source_code = None
    frame_ast = None
    source_lineno = None
    source_file = None
    source_name = None
    code_analyzer = None

    call_order = list()
    call_count = dict()

    def __init__(self, inputs, file_input=None, file_output=None):
        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")

        self.file_inputs = list()
        self.file_outputs = list()

        # Iterate over the list of arguments that are either file input/output,
        # and store in the appropriate class attribute
        for arg, file_list in zip((file_input, file_output),
                                  (self.file_inputs, self.file_outputs)):
            if arg is not None:
                file_list.extend(arg)

        # Store the list of arguments that are inputs
        self.inputs = inputs

        # Initialize other variables
        self.initialized = False
        self.has_return = True

    def _insert_static_information(self, tree, function, time_stamp):
        # Use a NodeVisitor to find the Call node that corresponds to the
        # current AnalysisStep. It will fetch static relationships between
        # variables and attributes, and link to the inputs and outputs of the
        # function
        ast_visitor = _CallAST(self, function, time_stamp)
        ast_visitor.visit(tree)

    def _analyze_function(self, function):
        # Check if the function code has a Return node.
        # If it has, returns True.
        try:
            source_code = inspect.getsourcelines(function)[0]
            source_code = source_code[1:]       # Strip decorator
            code_string = "".join(source_code)
            ast_tree = ast.parse(code_string)

            has_return = False
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Return):
                    has_return = True
                    break
        except:
            has_return = True

        return has_return

    def _capture_provenance(self, lineno, function, args, kwargs,
                            function_output, time_stamp):

        # 1. Capture Abstract Syntax Tree (AST) of the call to the
        # function. We need to check the source code in case the
        # call spans multiple lines. In this case, we fetch the
        # full statement.
        print(lineno)
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

        module = getattr(function, '__module__')
        function_info = FunctionInfo(function.__name__, module, None)

        # 4. Extract parameters passed to the function and store
        # in `input_data` dictionary. Two separate lists with the
        # names according to the arg/kwarg order are also
        # constructed, to map to the `args` and `keywords` fields
        # of the AST nodes

        input_data = {}
        input_args_names = []
        input_kwargs_names = []

        try:
            fn_sig = inspect.signature(function)
            func_parameters = fn_sig.bind(*args, **kwargs)

            # Get default arguments in case they were not passed
            default_args = {k: v.default
                            for k, v in fn_sig.parameters.items()
                            if v.default is not inspect.Parameter.empty}

            for arg_name, arg_value in \
                    func_parameters.arguments.items():
                cur_parameter = \
                    func_parameters.signature.parameters[arg_name]

                if arg_name in default_args:
                    default_args.pop(arg_name)

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

            # Add default arguments to kwargs
            input_kwargs_names.extend(list(default_args.keys()))

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

            default_args = {}

        # 5. Create parameters/input descriptions for the graph.
        # Here the inputs, but not the parameters passed to the
        # function, are transformed in the hashable type
        # `BuffaloObjectHash`. Inputs are defined by the parameter
        # `inputs` when initializing the class, and stored as the
        # class attribute `inputs`. If one parameter is defined
        # as a `file_input` when initializing the class, a hash
        # to the file is obtained using the `BuffaloFileHash`.

        # Initialize parameter list with all default arguments
        # that were not passed to the function
        parameters = default_args

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
            elif key in self.file_inputs:
                inputs[key] = BuffaloFileHash(input_value).info()
            elif not key in self.file_outputs:
                parameters[key] = input_value

        # 6. Create hashable `BuffaloObjectHash` for the output
        # objects to follow individual returns, if the function
        # is not returning None
        outputs = {}
        if self.has_return:
            if len(return_targets) > 1:
                for index, item in enumerate(function_output):
                    outputs[index] = BuffaloObjectHash(item).info()
            else:
                outputs[0] = BuffaloObjectHash(function_output).info()

        # If there is a file output as defined in the class
        # initialization, create the hash and add as output,
        # using `BuffaloFileHash`
        if len(self.file_outputs):
            for idx, file_output in enumerate(self.file_outputs):
                outputs[f"file.{idx}"] = \
                    BuffaloFileHash(input_data[file_output]).info()

        # 7. Analyze AST and fetch static relationships in the
        # input/output and other variables/objects in the script
        self._insert_static_information(ast_tree,
                                        function_info.name,
                                        time_stamp)

        # 8. Use a call counter to organize the nodes in the output
        # graph
        if not function_info.name in self.call_order:
            self.call_order.append(function_info.name)
            self.call_count[function_info.name] = 0

        self.call_count[function_info.name] += 1
        vis_position = (self.call_count[function_info.name],
                        self.call_order.index(function_info.name))

        # 9. Create tuple with the analysis step information.
        return AnalysisStep(function_info, inputs, parameters,
                            outputs,
                            input_args_names, input_kwargs_names,
                            ast_tree, source_line, time_stamp,
                            return_targets, vis_position)

    def _get_calling_line_number(self, frame):
        # Get the line number of the current call.
        # For that, we need to find the frame containing the call, starting
        # from `frame`, which is the current frame being executed.
        lineno = None

        # Extract information and calling function name in `frame`
        frame_info = inspect.getframeinfo(frame)
        function_name = frame_info.function

        if function_name == '<listcomp>':
            # For list comprehensions, we need to check the frame above,
            # as this creates a function named <listcomp>. We use a while loop
            # in case of nested list comprehensions.
            while function_name == '<listcomp>':
                frame = frame.f_back
                frame_info = inspect.getframeinfo(frame)
                function_name = frame_info.function

        # If the frame corresponds to the script file and the tracked function,
        # we get the line number
        if (frame_info.filename == self.source_file and
                function_name == self.source_name):
            lineno = frame.f_lineno

        return lineno

    def __call__(self, function):

        @wraps(function)
        def wrapped(*args, **kwargs):

            # Call the function and get the execution time stamp
            function_output = function(*args, **kwargs)
            time_stamp = datetime.datetime.utcnow().isoformat()

            # If capturing provenance...
            if Provenance.active:

                # Clear previous hash memoizations
                hash_memoizer.clear()

                # In the first run, analyze the function code to check if there
                # are returns. If no return, we won't track the output as this
                # is automatically the None object
                if not self.initialized:
                    self.has_return = self._analyze_function(function)
                    self.initialized = True

                # For functions that are used inside other decorated functions,
                # or recursively, check if the calling frame is the one being
                # tracked. If this call comes from the frame tracked, we will
                # get the line number. Otherwise, the line number will be
                # None, and the provenance tracking block will be skipped.
                try:
                    frame = inspect.currentframe().f_back
                    lineno = self._get_calling_line_number(frame)
                finally:
                    del frame

                # Capture provenance information
                if lineno is not None:

                    # Get AnalysisStep tuple with provenance information
                    step = self._capture_provenance(lineno, function, args,
                                                    kwargs, function_output,
                                                    time_stamp)

                    # Add step to the history.
                    # The history will be the base to generate the graph and
                    # PROV document.
                    Provenance.history.append(step)

            return function_output

        return wrapped

    @classmethod
    def set_calling_frame(cls, frame):
        # This method stores the frame of the code being tracked, and
        # extract several information that is needed for capturing provenance

        # Store the reference to the calling frame
        cls.calling_frame = frame

        # Get the file name and function associated with the frame
        cls.source_file = inspect.getfile(frame)
        cls.source_name = inspect.getframeinfo(frame).function

        # Set code start line. If the `provenance.activate` function was
        # called in the main script body, the name will be <module> and code
        # starts at line 1. If it was called inside a function (e.g. `main`),
        # we need to get the start line from the frame.
        if cls.source_name == '<module>':
            cls.source_lineno = 1
        else:
            cls.source_lineno = inspect.getlineno(frame)

        # Get the list with all the lines of the code being tracked
        code_lines = inspect.getsourcelines(frame)[0]

        # Clean any decorators (this happens when we are tracking inside a
        # function like `main`).
        cur_line = 0
        while code_lines[cur_line].strip().startswith('@'):
            cur_line += 1

        # Store the source code lines
        cls.source_code = code_lines[cur_line:]

        # Get the AST of the code being tracked
        cls.frame_ast = ast.parse("".join(cls.source_code).strip())

        # Create a _BuffaloCodeAnalyzer instance with the frame information,
        # so that we can capture provenance information later
        cls.code_analyzer = _BuffaloCodeAnalyzer(cls.source_code,
                                                 cls.frame_ast,
                                                 cls.source_lineno,
                                                 cls.source_name)

    @classmethod
    def get_prov_info(cls, **kwargs):
        """
        Returns the W3C PROV representation of the captured provenance
        information.
        """
        raise NotImplementedError

    @classmethod
    def get_graph(cls):
        """
        Get the Networkx graph of the provenance history.

        Returns
        -------
        BuffaloProvenanceGraph
        """
        return BuffaloProvenanceGraph(cls.history)

    @classmethod
    def dump_history(cls, filename):
        """
        Save the provenance history to disk.

        Parameters
        ----------
        filename : str or Path-like
            Destination file where the history will be stored.
        """
        #FIXME: this produces some errors with matplotlib objects
        dill.dump(Provenance.history, open(filename, "wb"))

    @classmethod
    def save_graph(cls, filename, show=False):
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

        source = cls.get_graph()
        source.to_pyvis(filename, show=show)

    @classmethod
    def get_script_variable(cls, name):
        """
        Access to variable values in the tracked code by name.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        object
            Python object stored in variable `name`.
        """
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


def get_graph(history=None):
    return Provenance.get_graph(history)


def dump_provenance(filename):
    Provenance.dump_history(filename)

