"""
This module implements classes to extract and work with the information
obtained from the nodes of an Abstract Syntax Tree describing the code
associated with a given execution line of the script.
"""

import ast
import itertools

from elephant.buffalo.static_code import (AttributeStep, NameStep,
                                          SubscriptStep)
from elephant.buffalo.object_hash import BuffaloObjectHash


class NameAST(ast.NodeTransformer):
    """
    NodeTransformer to find all root variables that are loaded in an
    Abstract Syntax Tree tree.

    The reference to the actual Python object is stored in the new node, and
    a hash to the object is added to the `Provenance` class decorator internal
    index.
    """

    provenance = None

    def __init__(self, provenance):
        super(NameAST, self).__init__()
        self.provenance = provenance

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            instance = self.provenance.get_script_variable(node.id)
            setattr(node, 'instance', instance)
            setattr(node, 'object_hash', BuffaloObjectHash(instance).info())
            return node
        return node


class CallAST(ast.NodeVisitor):

    provenance_tracker = None
    function = None

    def __init__(self, provenance_tracker, function, time_stamp):
        super(CallAST, self).__init__()
        self.provenance_tracker = provenance_tracker
        self.function = function
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
                                                    self.time_stamp)
        else:
            self.generic_visit(node)


def _fetch_object_tree(node, time_stamp):
    # Iterate recursively the syntax tree of `node`, to fetch the actual
    # Python objects that are represented in runtime, building an
    # hierarchical tree

    def _extract(node, child=None):
        if isinstance(node, ast.Subscript):
            subscript = SubscriptStep(node, time_stamp, child)
            _extract(node.value, subscript)
            return subscript
        elif isinstance(node, ast.Attribute):
            attribute = AttributeStep(node, time_stamp, child)
            _extract(node.value, attribute)
            return attribute
        elif isinstance(node, ast.Name):
            name = NameStep(node, time_stamp, child)
            return name

    return _extract(node)


def _build_object_tree_provenance(object_tree, provenance_tracker):
    # Iterate recursively through an hierarchical tree describing
    # the child/parent relationships between the objects, build the
    # provenance analysis steps associated and store in the provenance
    # tracker history

    def _hash_and_store(tree_node):
        if tree_node.object_hash is None:
            # Hash if needed
            tree_node.object_hash = BuffaloObjectHash(tree_node.value).info()
        if tree_node.parent is not None:
            # Insert provenance step
            if tree_node.parent.object_hash is None:
                # Hash if needed
                tree_node.parent.object_hash = BuffaloObjectHash(
                    tree_node.parent.value).info()
            provenance_tracker.history.append(
                tree_node.get_analysis_step())
            _hash_and_store(tree_node.parent)

    _hash_and_store(object_tree)


def _process_subscript_or_attribute(node, provenance_tracker, time_stamp):
    # Find root variable, hash it and include reference in the
    # node
    name_visitor = NameAST(provenance_tracker)
    name_visitor.visit(node.value)

    # Fetch object references from syntax
    object_tree = _fetch_object_tree(node, time_stamp)

    # Insert provenance operations and create hashes if necessary
    _build_object_tree_provenance(object_tree, provenance_tracker)
