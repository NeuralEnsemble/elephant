"""
This module implements classes to extract and work with the information
obtained from the nodes of an Abstract Syntax Tree describing the code
associated with a given function call within the script.

# TODO: add informative example

>>> isi_times = isi(block.segments[0].spiketrains[0])
"""

import ast
import itertools

from elephant.buffalo.static_code import (_AttributeStep, _NameStep,
                                          _SubscriptStep)


class _NameAST(ast.NodeTransformer):
    """
    NodeTransformer to find all root variables that are loaded in an
    Abstract Syntax Tree.

    The reference to the actual Python object is stored in the new node as
    the :attr:`instance`, and the object hash is stored in the
    :attr:`object_hash` attribute.

    Parameters
    ----------
    provenance_tracker : elephant.buffalo.Provenance
        Reference to the provenance tracker decorator.
    """

    provenance_tracker = None

    def __init__(self, provenance_tracker, hasher):
        super(_NameAST, self).__init__()
        self.provenance_tracker = provenance_tracker
        self.hasher = hasher

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # If this is a node where the variable is accessed, get the
            # actual object from the tracked frame, and store the object
            # reference and hash
            instance = self.provenance_tracker.get_script_variable(node.id)
            setattr(node, 'instance', instance)
            setattr(node, 'object_hash', self.hasher.info(instance))
            return node
        return node


class _CallAST(ast.NodeVisitor):
    """
    NodeVisitor to inspect and fetch subscript/attribute relationships
    in the call to the function that is being tracked.

    Parameters
    ----------
    provenance_tracker : elephant.buffalo.Provenance
        Reference to the provenance tracker decorator.
    function : str
        Name of the function being tracked.
    time_stamp : datetime
        Timestamp of the current function execution.
    """

    def __init__(self, provenance_tracker, hasher, function, time_stamp):
        super(_CallAST, self).__init__()
        self.provenance_tracker = provenance_tracker
        self.function = function
        self.hasher = hasher
        self.time_stamp = time_stamp

    def visit_Call(self, node):

        if isinstance(node.func, ast.Name) and node.func.id == self.function:

            # Fetch static information of Attribute and Subscript nodes that
            # were inputs. This should capture provenance hierarchical
            # information for inputs that are class members or items in
            # iterables
            for position, arg_node in enumerate(
                    itertools.chain(node.args, node.keywords)):

                if isinstance(arg_node, (ast.Subscript, ast.Attribute)):
                    _process_subscript_or_attribute(arg_node,
                                                    self.provenance_tracker,
                                                    self.hasher,
                                                    self.time_stamp)
        else:
            self.generic_visit(node)


def _fetch_object_tree(node, time_stamp):
    # Iterate recursively the syntax tree of `node`, to fetch the actual
    # Python objects that are represented in runtime, building an
    # hierarchical tree

    def _extract(node, child=None):
        if isinstance(node, ast.Subscript):
            subscript = _SubscriptStep(node, time_stamp, child)
            _extract(node.value, subscript)
            return subscript

        if isinstance(node, ast.Attribute):
            attribute = _AttributeStep(node, time_stamp, child)
            _extract(node.value, attribute)
            return attribute

        if isinstance(node, ast.Name):
            name = _NameStep(node, time_stamp, child)
            return name

    return _extract(node)


def _build_object_tree_provenance(object_tree, provenance_tracker, hasher):
    # Iterate recursively through an hierarchical tree describing
    # the child/parent relationships between the objects, build the
    # provenance analysis steps associated and store in the provenance
    # tracker history

    def _hash_and_store(tree_node):
        if tree_node.object_hash is None:
            # Hash if needed
            tree_node.object_hash = hasher.info(tree_node.value)
        if tree_node.parent is not None:
            # Insert provenance step
            if tree_node.parent.object_hash is None:
                # Hash if needed
                tree_node.parent.object_hash = hasher.info(
                    tree_node.parent.value)
            provenance_tracker.history.append(
                tree_node.get_analysis_step())
            _hash_and_store(tree_node.parent)

    _hash_and_store(object_tree)


def _process_subscript_or_attribute(node, provenance_tracker, hasher,
                                    time_stamp):
    # Find root variable, hash it and include reference in the
    # node
    name_visitor = _NameAST(provenance_tracker, hasher)
    name_visitor.visit(node.value)

    # Fetch object references from syntax
    object_tree = _fetch_object_tree(node, time_stamp)

    # Insert provenance operations and create hashes if necessary
    _build_object_tree_provenance(object_tree, provenance_tracker, hasher)
