"""
This module implements a class to store and fetch information from the source
code of the frame that activated provenance tracking.
The main purpose is to retrieve the full multiline statements that generated
the call to a tracked function.
"""
import numpy as np
import ast
from collections import namedtuple


IterationBlock = namedtuple('IterationBlock', ('iterators',
                                               'values',
                                               'indexes',
                                               'start',
                                               'end'))


class SourceCodeAnalyzer(object):
    """
    Stores the source code of the function that activated provenance tracking,
    and provides methods for retrieving execution statements.

    Parameters
    ----------
    source_code : list
        Extracted source code from the frame that activated provenance.
    ast_tree : ast.Module
        Abstract Syntax Tree of the source code.
    start_line : int
        Line from the source file where the code starts.
    source_name : str
        Name of the frame to which the source code corresponds.
    """

    iteration_blocks = list()

    def __init__(self, source_code, ast_tree, start_line, source_name):
        self.source_code = source_code
        self.ast_tree = ast_tree
        self.start_line = start_line
        self._offset = 0 if source_name == '<module>' else 1
        self.statement_lines = self._build_line_map(ast_tree)
        self.iteration_blocks = list()

    def _insert_iteration_block(self, node):
        iterators = []
        indexes = []
        values = []
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call):
                # Iterator is a function
                function = node.iter.func
                if function == 'enumerate':
                    # Target is a tuple, first element is the index,
                    # second element is the value during iteration
                    indexes.append(node.target.elts[0])
                    values.append(node.target.elts[1])
                    # Argument for the function is the iterator
                    iterators.append(node.iter.args[0].id)
                elif function == 'zip':
                    # Multiple iterators. Need to fetch the variable associated
                    # with each
                    for index, zip_arg in node.iter.args:
                        values.append(node.target.elts[index])
                        iterators.append(zip_arg.id)
            elif isinstance(node.iter, ast.Name):
                # Iterator is a variable
                iterators.append(node.iter.id)
                values.append(node.target.id)
        start_line = node.lineno
        end_lines = [child.lineno for child in ast.walk(node) if 'lineno' in
                     child._attributes]
        end_line = max(end_lines)
        self.iteration_blocks.append(
            IterationBlock(iterators, values, indexes, start_line, end_line)
        )

    def _build_line_map(self, ast_tree):
        is_function = False
        if (len(ast_tree.body) == 1 and
                isinstance(ast_tree.body[0], ast.FunctionDef)):
            # We are tracking inside a function (e.g., `def main():`)
            code_nodes = ast_tree.body[0].body
            is_function = True
        else:
            # We are tracking from the script root
            code_nodes = ast_tree.body

        # Add the line number of each node in the script/function body
        statement_lines = list()
        iteration_nodes = list()
        while len(code_nodes):
            node = code_nodes.pop(0)
            if hasattr(node, 'body'):
                # Another code block (e.g., if, for, while)
                # Just add the nodes in the body for further processing
                code_nodes.extend(node.body)
                if hasattr(node, 'iter'):
                    # This is an iteration node. Store it in the list to
                    # get information later
                    iteration_nodes.append(node)
            else:
                # A statement. Find the maximum line number
                end_lines = [child.lineno for child in ast.walk(node) if
                             'lineno' in child._attributes]
                statement_lines.append((node.lineno, max(end_lines)))

        statement_lines = sorted(statement_lines, key=lambda x: x[0])
        statement_lines = np.asarray(statement_lines)

        # If in a function, the lines will be relative to the function `def`
        # line. We need to correct. The `def` line is line number 1, therefore,
        # codes start on line 2 of the function body.
        if is_function:
            statement_lines += self.start_line - 2

        for node in iteration_nodes:
            self._insert_iteration_block(node)

        return statement_lines

    def _get_statement_lines(self, line_number):
        line_diff = self.statement_lines[:, 0] - line_number
        nearest_number_index = np.argmax(line_diff[line_diff <= 0])
        return self.statement_lines[nearest_number_index, :]

    def is_iteration(self, line_number):
        """
        Returns if the line under execution is inside a iteration block (e.g.,
        for loop).

        Parameters
        ----------
        line_number : int
            Line number from :attr:`source_code`.

        Returns
        -------
        bool
            True if part of an iteration block, False otherwise.
        """
        pass

    def extract_multiline_statement(self, line_number):
        """
        Fetch all code lines in case `line_number` contains a statement that
        is the end of a multiline statement.

        Parameters
        ----------
        line_number : int
            Line number from :attr:`source_code`.

        Returns
        -------
        str
            The code corresponding to the full statement in case it is a
            multiline.

        """
        statement_start, statement_end = self._get_statement_lines(line_number)
        position_offset = -self.start_line + self._offset
        return "".join(
            self.source_code[statement_start + position_offset:
                             statement_end + position_offset + 1]).strip()
