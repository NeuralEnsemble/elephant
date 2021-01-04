"""
This module implements a class to store and fetch information from the source
code of the frame that activated provenance tracking.
The main purpose is to retrieve the full multiline statements that generated
the call to a tracked function.
"""
import numpy as np
import ast
from collections import namedtuple


from elephant.buffalo.ast_analysis import (_process_subscript_or_attribute,
                                            _build_object_tree_provenance)


class IterationBlocks(object):
    """
    Stores blocks of `for` loops, and provides access to the iterator
    variables.
    """

    def __init__(self):
        # This will hold start and end line numbers of the `ast.For` node,
        # the `iter` and `target` values, as well as extracted AST nodes
        # representing the loop iterables, indexes and values.
        self.blocks = None

    def add(self, nodes, offset=0):
        for_statement_lines = list()
        for for_node in nodes:
            if not isinstance(for_node, ast.For):
                raise TypeError("Node is not an `ast.For` iteration!")

            # Find the maximum line of the for loop block
            end_lines = [child.lineno for child in ast.walk(for_node) if
                         'lineno' in child._attributes]
            last_line = max(end_lines) + offset
            start_line = for_node.lineno + offset

            iterator = for_node.iter
            target = for_node.target
            indexes, values, iterables = \
                self._extract_information(iterator, target)
            iteration = (start_line, last_line, iterator, target, indexes,
                         values, iterables)

            for_statement_lines.append(iteration)

        for_statement_lines = sorted(for_statement_lines, key=lambda x: x[0])

        # Final information is stored in a NumPy array with the following
        # columns: start line, last line, iterator AST node, target AST node,
        # list of nodes representing indexes, list of nodes representing
        # values, and list of nodes representing iterables in the loop.
        # Each for iteration block is a row.
        self.blocks = np.asarray(for_statement_lines)

    @staticmethod
    def _extract_information(iterator, target):
        iterables = []
        indexes = []
        values = []

        if isinstance(iterator, ast.Call):
            # Iterator is a function. Check for functions that operate on
            # iterables and fetch relevant information.
            function = iterator.func.id
            if function == 'enumerate':
                # Target is a tuple: first element is the index,
                # second element is the value during iteration
                # TODO: this may still break if not unpacking at the loop
                indexes.append(target.elts[0])
                values.append(target.elts[1])
                # Argument for the function is the iterable
                iterables.append(iterator.args[0])
            elif function == 'zip':
                # Multiple iterators. Need to fetch the variable associated
                # with each
                for index, zip_arg in enumerate(iterator.args):
                    values.append(target.elts[index])
                    iterables.append(zip_arg)
            elif function == 'range':
                # Not Pythonic, but will loop only to fetch a value
                # No iterable object to map
                values.append(target)
        elif isinstance(iterator, (ast.Name, ast.Subscript, ast.Attribute)):
            # Iterator is a variable, attribute or subscript
            iterables.append(iterator)
            values.append(target)

        return indexes, values, iterables

    def get_object(self, position, name, tracker):
        pass

    def within_iteration(self, position):
        # If `position` is within an iteration block, the difference w.r.t
        # the starting line will be positive, and w.r.t to the last line is
        # <= 0. Therefore, the product of the differences will be <= 0.
        # For any position outside the block, either both differences are
        # positive or both differences are negative. Therefore, the product
        # will always be > 0.
        map = self.blocks[:, 0:2] - position
        map = np.prod(map, axis=1)

        # Return the block(s) with iteration information
        return self.blocks[map <= 0]


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

    def __init__(self, source_code, ast_tree, start_line, source_name):
        self.source_code = source_code
        self.ast_tree = ast_tree
        self.start_line = start_line
        self._offset = 0 if source_name == '<module>' else 1
        self.iteration_blocks = IterationBlocks()
        self.statement_lines = self._build_line_map(ast_tree)

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

                # If `else` block is present, add it as well
                if len(node.orelse):
                    code_nodes.extend(node.orelse)

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
        offset = 0
        if is_function:
            offset = self.start_line - 2
            statement_lines += offset

        # If any iteration block was found, process it and prepare to retrieve
        # the relevant objects later
        self.iteration_blocks.add(iteration_nodes, offset)

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
