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

from elephant.buffalo.object_hash import BuffaloObjectHasher, BuffaloFileHash
from elephant.buffalo.graph import BuffaloProvenanceGraph
from elephant.buffalo.ast_analysis import _CallAST
from elephant.buffalo.code_lines import _BuffaloCodeAnalyzer
from elephant.buffalo.serialization import generate_prov_representation

from os.path import splitext

from pprint import pprint


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
        * 'time_stamp_start', 'time_stamp_end': `str` with the ISO
          representation of the start and end times of the function execution;
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
    source_file = None
    calling_frame = None

    call_order = list()
    call_count = dict()

    def __init__(self, inputs, file_input=None, file_output=None):
        if inputs is None:
            inputs = []
        if file_input is None:
            file_input = []
        if file_output is None:
            file_output = []

        if not isinstance(inputs, list):
            raise ValueError("`inputs` must be a list")

        # Store the arguments that are file input/outputs
        self.file_inputs = [f_input for f_input in file_input
                            if f_input is not None]
        self.file_outputs = [f_output for f_output in file_output
                             if f_output is not None]

        # Store the list of arguments that are inputs
        self.inputs = inputs

    def _insert_static_information(self, tree, hasher, function, time_stamp):
        # Use a NodeVisitor to find the Call node that corresponds to the
        # current AnalysisStep. It will fetch static relationships between
        # variables and attributes, and link to the inputs and outputs of the
        # function. The hasher object is passed, to use hash memoization in
        # case the hash of some object is already computed
        ast_visitor = _CallAST(provenance_tracker=self, hasher=hasher,
                               function=function, time_stamp=time_stamp)
        ast_visitor.visit(tree)

    def _process_input_arguments(self, function, args, kwargs):
        # Inspect the arguments to extract the ones defined as inputs.
        # Values are stored in a dictionary with the argument name as key.
        # If signature inspection is not possible, the inputs are stored by
        # order in the function call, with the index as keys.

        # Initialize dictionaries and lists
        input_data = {}
        input_args_names = []
        input_kwargs_names = []

        try:
            # Get the function signature and bind the arguments, obtaining a
            # dictionary with argument name as keys and argument value as
            # values
            fn_sig = inspect.signature(function)
            func_parameters = fn_sig.bind(*args, **kwargs)

            # Get the default argument values, to store them in case they
            # were not passed in the call
            default_args = {k: v.default
                            for k, v in fn_sig.parameters.items()
                            if v.default is not inspect.Parameter.empty}

            # For each item in the bound arguments dictionary...
            for arg_name, arg_value in func_parameters.arguments.items():

                # Get the description of the current argument by its name
                cur_parameter = \
                    func_parameters.signature.parameters[arg_name]

                # If this argument is one of possible default values, remove
                # it, since the user has passed a value explicitly
                if arg_name in default_args:
                    default_args.pop(arg_name)

                # If the argument is variable positional, i.e., *arg, we will
                # store its tuple in the dictionary as the VarArgs named tuple.
                # This signals that this argument's value is multiple.
                # Otherwise, we just store the value
                if cur_parameter.kind != VAR_POSITIONAL:
                    input_data[arg_name] = arg_value
                else:
                    # Variable positional arguments are stored as
                    # the namedtuple VarArgs.
                    input_data[arg_name] = VarArgs(arg_value)

                # Store the argument name in the appropriate list
                if arg_name in kwargs:
                    input_kwargs_names.append(arg_name)
                else:
                    input_args_names.append(arg_name)

            # Add the default argument names to the list of kwargs names
            input_kwargs_names.extend(default_args.keys())

        except ValueError:
            # Can't inspect signature. Append args/kwargs by order
            for arg_index, arg in enumerate(args):
                input_data[arg_index] = arg
                input_args_names.append(arg_index)

            # Keyword arguments index start after the last positional argument
            kwarg_start = len(input_data)
            for kwarg_index, kwarg in enumerate(kwargs,
                                                start=kwarg_start):
                input_data[kwarg_index] = kwarg
                input_kwargs_names.append(kwarg_index)

            # No default arguments
            default_args = {}

        return input_data, input_args_names, input_kwargs_names, default_args

    def _capture_provenance(self, lineno, function, args, kwargs,
                            function_output, time_stamp_start,
                            time_stamp_end):

        # 1. Capture Abstract Syntax Tree (AST) of the call to the
        # function. We need to check the source code in case the
        # call spans multiple lines. In this case, we fetch the
        # full statement.
        # TODO: use logging instead of printing
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
                # This branch should not be reachable
                raise ValueError("Unknown assign target!")

        # 3. Extract function name and information
        # TODO: fetch version information

        module = getattr(function, '__module__')
        function_info = FunctionInfo(name=function.__name__,
                                     module=module, version=None)

        # 4. Extract the parameters passed to the function and store them in
        # the `input_data` dictionary.
        # Two separate lists with the names according to the arg/kwarg order
        # are also constructed, to map to the `args` and `keywords` fields
        # of the AST nodes. Also, the list of all arguments whose values taken
        # are defaults is returned as the `default_args` dictionary.

        input_data, input_args_names, input_kwargs_names, default_args = \
            self._process_input_arguments(function, args, kwargs)

        # 5. Create parameters/input descriptions for the graph.
        # Here the inputs, but not the parameters passed to the function, are
        # hashed using the `BuffaloObjectHasher` object.
        # Inputs are defined by the parameter `inputs` when initializing the
        # decorator, and stored as the attribute `inputs`. If one parameter
        # is defined as a `file_input` in the initialization, a hash to the
        # file is obtained using the `BuffaloFileHash`.
        # After this step, all hashes of input parameters/files are going to
        # be stored in the dictionary `inputs`.

        hasher = BuffaloObjectHasher()

        # Initialize parameter list with all default arguments that were not
        # passed to the function
        parameters = default_args

        inputs = {}
        for key, input_value in input_data.items():
            if key in self.inputs:
                if isinstance(input_value, VarArgs):
                    # If the argument is multiple, hash each value of the
                    # tuple and store them inside a `VarArgs` namedtuple so
                    # that we know this is a multiple input
                    var_input_list = []
                    for var_arg in input_value.args:
                        var_input_list.append(hasher.info(var_arg))
                    inputs[key] = VarArgs(tuple(var_input_list))
                else:
                    inputs[key] = hasher.info(input_value)
            elif key in self.file_inputs:
                # Input is from a file. Hash using `BuffaloFileHash`
                inputs[key] = BuffaloFileHash(input_value).info()
            elif key not in self.file_outputs:
                # The remainder argument is also not an output file, so this
                # is an actual input to the function defined when initializing
                # the decorator.
                parameters[key] = input_value

        # 6. Create hash for the output using `BuffaloObjectHasher` to follow
        # individual returns. The hashes will be stored in the `outputs`
        # dictionary, with the index as the order of each returned object.
        outputs = {}
        if len(return_targets) == 1:
            function_output = [function_output]
        for index, item in enumerate(function_output):
            outputs[index] = hasher.info(item)

        # If there is a file output as defined in the decorator
        # initialization, create the hash and add as output using
        # `BuffaloFileHash`. These outputs will be identified by the key
        # `file.X`, where X is an integer with the order of the file output
        if len(self.file_outputs):
            for idx, file_output in enumerate(self.file_outputs):
                outputs[f"file.{idx}"] = \
                    BuffaloFileHash(input_data[file_output]).info()

        # 7. Analyze AST and fetch static relationships in the
        # input/output and other variables/objects in the script
        self._insert_static_information(tree=ast_tree, hasher=hasher,
                                        function=function_info.name,
                                        time_stamp=time_stamp_start)

        # 8. Use a call counter to organize the nodes in the output
        # graph
        if function_info.name not in self.call_order:
            self.call_order.append(function_info.name)
            self.call_count[function_info.name] = 0

        self.call_count[function_info.name] += 1
        vis_position = (self.call_count[function_info.name],
                        self.call_order.index(function_info.name))

        # 9. Create tuple with the analysis step information and return.
        return AnalysisStep(function=function_info,
                            input=inputs,
                            params=parameters,
                            output=outputs,
                            arg_map=input_args_names,
                            kwarg_map=input_kwargs_names,
                            call_ast=ast_tree,
                            code_statement=source_line,
                            time_stamp_start=time_stamp_start,
                            time_stamp_end=time_stamp_end,
                            return_targets=return_targets,
                            vis=vis_position)

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
        elif function_name == 'wrapper':
            # For the Elephant deprecations, we need to skip the decorator
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

            # Call the function and get the execution time stamps
            time_stamp_start = datetime.datetime.utcnow().isoformat()
            function_output = function(*args, **kwargs)
            time_stamp_end = datetime.datetime.utcnow().isoformat()

            # If capturing provenance...
            if Provenance.active:

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
                    step = self._capture_provenance(
                        lineno=lineno,
                        function=function, args=args,
                        kwargs=kwargs,
                        function_output=function_output,
                        time_stamp_start=time_stamp_start,
                        time_stamp_end=time_stamp_end)

                    # Add step to the history.
                    # The history will be the base to generate the graph and
                    # PROV document.
                    Provenance.history.append(step)

            return function_output

        return wrapped

    @classmethod
    def set_calling_frame(cls, frame):
        """
        This method stores the frame of the code being tracked, and
        extract several information that is needed for capturing provenance

        Parameters
        ----------
        frame : inspect.frame
            Frame object returned by the `inspect` module. This must
            correspond to the namespace where provenance tracking was
            activated. This is automatically fetched by the interface function
            :func:`provenance.activate` defined in this module.
        """

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
                                                 cls.source_lineno)

    @classmethod
    def get_prov_info(cls):
        """
        Returns the W3C PROV representation of the captured provenance
        information.
        """
        return generate_prov_representation(cls.source_file,
                                            cls.history)

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
        # TODO: this method will be removed, any visualization will be part
        # of BuffaloProvDocument
        name, ext = splitext(filename)
        if not ext.lower() in ['.html', '.htm']:
            raise ValueError("Filename must have HTML extension (.html, "
                             ".htm)!")

        source = BuffaloProvenanceGraph(cls.history)
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
    Provenance.calling_frame = None
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


def save_provenance(filename=None, file_format='rdf'):
    """
    Serialized provenance information according to the W3C Provenance Data
    Model (PROV).

    Parameters
    ----------
    filename : str or path-like, optional
        Destination file to serialize the provenance information.
        If None, the function will return a string containing the provenance
        information in the specified format.
        Default: None
    file_format : {'json', 'rdf', 'prov', 'xml'}, optional
        Serialization format. Formats currently supported are:
        * 'json' : PROV-JSON
        * 'rdf' : PROV-O
        * 'prov' : PROV-N
        * 'xml : PROV-XML
        Default: 'rdf'

    Returns
    -------
    str or None
        If `filename` is None, the function returns the PROV information as
        a string. If a file destination was informed, the return is None.

    Notes
    -----
    For details regarding the serialization formats, please check the
    specification on the W3C website (https://www.w3.org/TR/prov-primer/).

    """
    prov_document = Provenance.get_prov_info()
    prov_data = prov_document.serialize(filename, format=file_format)
    return prov_data
