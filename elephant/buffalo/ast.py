"""
This module implements classes to extract and work with the information
obtained from the nodes of an Abstract Syntax Tree describing the code
associated with a given execution line of the script.
"""

import ast
import itertools

from elephant.buffalo.static_code import (AttributeStep, NameStep,
                                          SubscriptStep)


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
        super().__init__()
        self.provenance = provenance

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            instance, object_hash = self.provenance.add_script_variable(node.id)
            setattr(node, 'instance', instance)
            setattr(node, 'object_hash', object_hash)
            return node
        return node


class CallAST(ast.NodeVisitor):

    provenance_tracker = None

    def __init__(self, provenance_tracker, inputs, outputs):
        super().__init__()
        self.provenance_tracker = provenance_tracker
        self._inputs = inputs
        self._outputs = outputs

    def _fetch_object_tree(self, node):
        # Iterate recursively the syntax tree of `node`, to fetch the actual
        # Python objects that are represented in runtime, building an
        # hierarchical tree

        def _extract(node, child=None):
            if isinstance(node, ast.Subscript):
                subscript = SubscriptStep(node, child)
                _extract(node.value, subscript)
                return subscript
            elif isinstance(node, ast.Attribute):
                attribute = AttributeStep(node, child)
                _extract(node.value, attribute)
                return attribute
            elif isinstance(node, ast.Name):
                name = NameStep(node, child)
                return name

        return _extract(node)

    def _build_object_tree_provenance(self, object_tree):
        # Iterate recursively through an hierarchical tree describing
        # the child/parent relationships between the objects, build the
        # provenance analysis steps associated and store in the provenance
        # tracker history

        def _hash_and_store(call_ast, tree_node):
            if tree_node.object_hash is None:
                # Hash if needed
                tree_node.object_hash = self.provenance_tracker.add(
                    tree_node.value)
            if tree_node.parent is not None:
                # Insert provenance step
                if tree_node.parent.object_hash is None:
                    # Hash if needed
                    tree_node.parent.object_hash = self.provenance_tracker.add(
                        tree_node.parent.value)
                call_ast.provenance_tracker.history.append(
                    tree_node.get_analysis_step())
                _hash_and_store(call_ast, tree_node.parent)

        _hash_and_store(self, object_tree)

    def visit_Call(self, node):

        args_tree = {}
        kwargs_tree = {}

        # Fetch static information of Attribute and Subscript nodes that
        # were inputs. This should capture provenance hierarchical information
        # for inputs that are class members or items in iterables
        for position, arg_node in enumerate(
                itertools.chain(node.args, node.keywords)):

            if isinstance(arg_node, (ast.Subscript, ast.Attribute)):

                # Find root variable, hash it and include reference in the
                # node
                name_visitor = NameAST(self.provenance_tracker)
                name_visitor.visit(arg_node.value)

                # Fetch object references from syntax
                object_tree = self._fetch_object_tree(arg_node)

                # Insert provenance operations and create hashes if necessary
                self._build_object_tree_provenance(object_tree)
