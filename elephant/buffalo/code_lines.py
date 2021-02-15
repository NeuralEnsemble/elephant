"""
This module implements a class to store and fetch information from the source
code of the frame that activated provenance tracking.
The main purpose is to retrieve the full multiline statements that generated
the call to a tracked function.
"""
import numpy as np
import ast


class _BuffaloCodeAnalyzer(object):
    """
    Stores the source code of the frame that activated provenance tracking,
    and provides methods for retrieving execution statements by line number.

    Parameters
    ----------
    source_code : list
        Extracted source code from the frame that activated provenance.
    ast_tree : ast.Module
        Abstract Syntax Tree of the source code.
    start_line : int
        Line from the source file where the code starts.
    source_name : str
        Name of the frame whose source code corresponds.
    """

    def __init__(self, source_code, ast_tree, start_line, source_name):
        self.source_code = source_code
        self.ast_tree = ast_tree
        self.start_line = start_line - 1   #TODO: check offsets in first line
        self._offset = 0 if source_name == '<module>' else 1
        # self.iteration_blocks = IterationBlocks()
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
        # self.iteration_blocks.add(iteration_nodes, offset)

        return statement_lines

    def _get_statement_lines(self, line_number):
        line_diff = self.statement_lines[:, 0] - line_number
        nearest_number_index = np.argmax(line_diff[line_diff <= 0])
        return self.statement_lines[nearest_number_index, :]

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
