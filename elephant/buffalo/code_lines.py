"""
This module implements a class to store and fetch information from the source
code of the frame that activated provenance tracking.
The main purpose is to retrieve the full multiline statements that generated
the call to a tracked function.
"""
import numpy as np
import ast


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
        self.start_line_map, self.end_line_map = self._build_line_map(ast_tree)

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
        statement_start_lines = list()
        statement_end_lines = list()
        for node in code_nodes:
            statement_start_lines.append(node.lineno)
            end_lines = [child.lineno for child in ast.walk(node) if
                         'lineno' in child._attributes]
            statement_end_lines.append(max(end_lines))

        start_line_map = np.sort(np.array(statement_start_lines))
        end_line_map = np.sort(np.array(statement_end_lines))

        # If in a function, the lines will be relative to the function `def`
        # line. We need to correct. The `def` line is line number 1, therefore,
        # codes start on line 2 of the function body.
        if is_function:
            start_line_map += self.start_line - 2
            end_line_map += self.start_line - 2

        return start_line_map, end_line_map

    def _get_statement_lines(self, line_number):
        line_diff = self.start_line_map - line_number
        nearest_number_index = np.argmax(line_diff[line_diff <= 0])
        return (self.start_line_map[nearest_number_index],
                self.end_line_map[nearest_number_index])

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
