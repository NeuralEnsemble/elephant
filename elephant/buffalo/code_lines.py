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

    def __init__(self, source_code, ast_tree, start_line):
        self.source_code = np.array(source_code)
        self.start_line = start_line
        self.ast_tree = ast_tree
        # self.iteration_blocks = IterationBlocks()
        self.statement_lines, self.source_lines = \
            self._build_line_map(ast_tree)
        pass

    def _build_line_map(self, ast_tree):
        # This function analyzes the AST structure to fetch the start and end
        # lines of each statement, while also fetching loops and conditional
        # blocks. A mapping of each script line to the actual code is also
        # returned, that is used later when fetching the full statements.

        # We extract a stack with all nodes in the script/function body. To
        # correct the starting line if provenance is tracked inside a function
        # (e.g., `def main():`), we set a flag to use later
        is_function = False
        if (len(ast_tree.body) == 1 and
                isinstance(ast_tree.body[0], ast.FunctionDef)):
            # We are tracking inside a function
            code_nodes = ast_tree.body[0].body
            is_function = True
        else:
            # We are tracking from the script root
            code_nodes = ast_tree.body

        # Build the list with line numbers of each main node in the
        # script/function body. These are stored in `statement_lines` array,
        # where column 0 is the starting line of the statement, and column 1
        # the end line. The line information from the AST is relative to the
        # scope of code, i.e., for code inside a function, the first line
        # after `def` is line 2. We correct this later after having the full
        # array.
        statement_lines = list()
        iteration_nodes = list()   # WIP; to fetch flow control

        # We process node by node. Whenever code blocks are identified, all
        # nodes in its body are pushed to the `code_nodes` stack
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

        # Convert list to the final array, allowing easy masking
        statement_lines = sorted(statement_lines, key=lambda x: x[0])
        statement_lines = np.asarray(statement_lines)

        # Create an array with the line number of each line in the source code
        source_lines = np.arange(self.start_line,
                                 self.start_line + self.source_code.shape[0])

        # Correct the line numbers. If in a function, the `def` line is 1, and
        # the code starts on line 2 of the function body. The code in
        # `self.source_code` also starts one line after the number stored in
        # `self.start_line`.
        if is_function:
            offset = self.start_line - 2
            statement_lines += offset
            source_lines -= 1

        # If any iteration block was found, process it and prepare to retrieve
        # the relevant objects later (WIP)
        # self.iteration_blocks.add(iteration_nodes, offset)

        return statement_lines, source_lines

    def extract_multiline_statement(self, line_number):
        """
        Fetch all code lines in case `line_number` contains a statement that
        is the end or part of a multiline statement.

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
        # Find the start and end line of the statement identified by
        # `line_number`
        line_diff = self.statement_lines[:, 0] - line_number
        nearest_number_index = np.argmax(line_diff[line_diff <= 0])
        statement_start, statement_end = \
            self.statement_lines[nearest_number_index, :]

        # Obtain the mask to get the source code between the start and end
        # lines
        line_mask = np.logical_and(self.source_lines >= statement_start,
                                   self.source_lines <= statement_end)

        # Retrieve the lines and join in a single string
        return "".join(
            self.source_code[line_mask]).strip()
