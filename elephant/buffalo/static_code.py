"""
This module implements classes to identify Python objects through the analysis
of nodes from an Abstract Syntax Tree, and generates the relationships
between them. The classes support building a hierarchical tree describing
the child/parent relationships between the objects.
"""

import ast
from elephant.buffalo.types import AnalysisStep, FunctionInfo


class _StaticStep(object):
    """
    Base class for analysis steps extracted from static code analysis.

    The information from a single Abstract Syntax Tree node is extracted and
    the reference to the actual Python object that the node represents is
    stored, as well as the hash.

    Parameters
    ----------
    node : ast.AST
        Abstract Syntax Tree node that represents the object.
        The subclass varies for each operation.
    child : _StaticStep, optional
        A `_StaticStep` object that is owned by the one being created. If a
        node in the Abstract Syntax Tree contains other nodes that describe a
        Python object, this node is the parent.
        Default: None.

    Attributes
    ----------
    object_hash : BuffaloObjectHash
        Hash object describing the Python object associated with this
        `StaticStep` instance.
    parent : _StaticStep
        `StaticStep` object that owns this instance.
    value : object
        Reference to the actual Python object associated with this
        `StaticStep` instance.

    Raises
    ------
    TypeError
        If `node` is not of the type describing the operation represented
        by the `_StaticStep` object.
    """

    _operation = None
    _node_type = None

    def __init__(self, node, time_stamp, child=None):
        if not isinstance(node, self._node_type):
            raise TypeError("AST node must be of type '"
                            f"{type(self._node_type)}'")
        self.parent = None
        self._node = node
        if child is not None:
            child.set_parent(self)
        self.object_hash = None
        self.time_stamp = time_stamp

    def set_parent(self, parent):
        self.parent = parent

    @property
    def value(self):
        """
        Returns the Python object associated with this instance.
        """
        raise NotImplementedError

    def _get_params(self):
        raise NotImplementedError

    def get_analysis_step(self):
        """
        Returns an `AnalysisStep` named tuple describing the relationships
        between parent and child nodes.
        """
        import elephant.buffalo.decorator as provenance

        params = self._get_params()
        input_object = self.parent.object_hash if self.parent is not None \
            else None
        output_object = self.object_hash

        return AnalysisStep(
            function=FunctionInfo(name=self._operation,
                                  module="",
                                  version=""),
            input={0: input_object},
            params=params,
            output={0: output_object},
            arg_map=None,
            kwarg_map=None,
            call_ast=self._node,
            code_statement=None,
            time_stamp_start=self.time_stamp,
            time_stamp_end=self.time_stamp,
            return_targets=[],
            vis=(None, None))


class _NameStep(_StaticStep):
    """
    Analysis step that represents an `ast.Name` Abstract Syntax Tree node.

    This step is supposed to be the first level of a tree describing the
    dependencies between the objects, and maps to a variable in the script.

    The node must be previously modified to include the reference to the
    Python object associated with the variable, and a `BuffaloObjectHash`
    hash.

    """

    _operation = 'variable'
    _node_type = ast.Name

    def __init__(self, node, time_stamp, child=None):
        super(_NameStep, self).__init__(node, time_stamp, child)
        self.object_hash = node.object_hash

    @property
    def value(self):
        return self._node.instance

    def _get_params(self):
        return None


class _SubscriptStep(_StaticStep):
    """
    Analysis step that represents an `ast.Subscript` Abstract Syntax Tree node.

    This step represents a subscripting operation in the script.
    """

    _operation = 'subscript'
    _node_type = ast.Subscript

    def __init__(self, node, time_stamp, child):
        super(_SubscriptStep, self).__init__(node, time_stamp, child)
        self._slice, self._params = self._get_slice(node.slice)

    @staticmethod
    def _get_slice(slice_node):
        # Extracts index or slice information from an `ast.Slice` or
        # `ast.Index` nodes that are the `slice` attribute of `ast.Subscript`.
        # Returns the slice/index value and a dictionary to be stored as
        # parameters in the `AnalysisStep` named tuple.

        params = {}

        if isinstance(slice_node, ast.Index):

            # Integer or string indexing
            if isinstance(slice_node.value, ast.Num):
                index_value = int(slice_node.value.n)
            elif isinstance(slice_node.value, ast.Str):
                index_value = slice_node.value.s
            elif isinstance(slice_node.value, ast.Name):
                from elephant.buffalo.decorator import Provenance
                index_value = Provenance.get_script_variable(slice_node.value.id)
            else:
                raise TypeError("Operation not supported")

            params['index'] = index_value
            return index_value, params

        # Required for newer Python versions
        if isinstance(slice_node, ast.Constant):
            index_value = slice_node.value
            params['index'] = index_value
            return index_value, params

        if isinstance(slice_node, ast.Name):
            from elephant.buffalo.decorator import Provenance
            index_value = Provenance.get_script_variable(slice_node.id)
            params['index'] = index_value
            return index_value, params

        if isinstance(slice_node, ast.Slice):

            # Slicing
            stop = int(slice_node.upper.n)
            start = getattr(slice_node, 'lower', None)
            step = getattr(slice_node, 'step', None)

            start = int(start.n) if start is not None else None
            step = int(step.n) if step is not None else None

            params['slice'] = f":{stop}"
            if start is not None:
                params['slice'] = f"{start}{params['slice']}"
            if step is not None:
                params['slice'] += f":{step}"

            return slice(start, stop, step), params

    def _get_params(self):
        return self._params

    @property
    def value(self):
        return self.parent.value[self._slice]


class _AttributeStep(_StaticStep):
    """
    Analysis step that represents an `ast.Attribute` Abstract Syntax Tree
    node.

    This step represents accessing an object attribute using dot '.' operation
    in the script.
    """

    _operation = 'attribute'
    _node_type = ast.Attribute

    def __init__(self, node, time_stamp, child=None):
        super(_AttributeStep, self).__init__(node, time_stamp, child)

    def _get_params(self):
        return {'name': self._node.attr}

    @property
    def value(self):
        return getattr(self.parent.value, self._node.attr)
