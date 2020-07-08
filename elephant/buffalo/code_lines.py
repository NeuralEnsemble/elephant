from tokenize import (generate_tokens, NEWLINE, OP, COMMENT, RBRACE, LBRACE,
                      RPAR, LPAR, RSQB, LSQB, COLON, INDENT, TokenError)
from six import StringIO


EXACT_TOKEN_TYPES = {
    ')': RPAR,
    '(': LPAR,
    ']': RSQB,
    '[': LSQB,
    ':': COLON,
    '}': RBRACE,
    '{': LBRACE,
}


class SourceCodeAnalyzer(object):
    """
    Stores the source code of the function that activated provenance tracking,
    and provides methods for retrieving execution statements.

    Parameters
    ----------
    source_code : list
        Extracted source code from the frame that activated provenance.
    start_line : int
        Line from the source file where the code starts.
    """

    def __init__(self, source_code, start_line):
        self.source_code = source_code
        self.start_line = start_line

    def _check_line(self, line_number):
        # Verifies if a given line is part of a multiline statement
        try:
            if line_number < (self.start_line + 1):
                return None
            line = self._get_code_line(line_number)
            string_io = StringIO(line)

            # If TokenError is raised, this is part of a multiline
            # statement.
            tokens = generate_tokens(string_io.readline)

            # If no error is raised, then check if there are open brackets.
            # If not, check if the line terminates in a multiline
            # character, such as \ + ( [ { , ".

            # Iterate over tokens and accumulate counts
            last_token = None
            operator_count = {RBRACE: 0, RPAR: 0, RSQB: 0,
                              LBRACE: 0, LPAR: 0, LSQB: 0}
            for token in tokens:
                # Ignore any comments and indentations
                if token[0] == NEWLINE:
                    break
                if token[0] == COMMENT or token[0] == INDENT:
                    continue

                if token[0] == OP:
                    exact_type = token.exact_type \
                        if hasattr(token, 'exact_type') else \
                        EXACT_TOKEN_TYPES[token[1]]
                    if exact_type in [RBRACE, RPAR, RSQB, LBRACE, LPAR,
                                      LSQB]:
                        operator_count[exact_type] += 1
                last_token = token

            # If number of any L brackets are greater than R brackets, then
            # this is part of a multiline.
            for right_bracket, left_bracket in zip([RSQB, RBRACE, RPAR],
                                                   [LSQB, LBRACE, LPAR]):
                if operator_count[left_bracket] < \
                        operator_count[right_bracket]:
                    return line

            # Check for ending operators in multilines
            if last_token[0] == OP:
                exact_type = last_token.exact_type \
                    if hasattr(last_token, 'exact_type') \
                    else EXACT_TOKEN_TYPES[last_token[1]]
                if exact_type in [RBRACE, RPAR, RSQB, COLON]:
                    return None
                return line

            return None

        except TokenError:
            return line

    def _get_code_line(self, line_number):
        return self.source_code[line_number - self.start_line]

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

        statement = []
        cur_line = self._check_line(line_number)
        if cur_line is not None:
            # We know this is already a multiline statement. Add the line
            # above and start checking the previous lines
            statement.append(self._get_code_line(line_number - 1))
            previous_line_number = line_number - 2
        else:
            cur_line = self._get_code_line(line_number)
            previous_line_number = line_number - 1

        previous_line = self._check_line(previous_line_number)
        while previous_line is not None:
            statement.append(previous_line)
            previous_line_number -= 1
            previous_line = self._check_line(previous_line_number)

        return "".join(statement[::-1] + [cur_line]).strip()
