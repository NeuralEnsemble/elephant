"""
Sphinx extension that validates documentation of default values in
functions, methods and class constructors.

If a parameter has a default value, the rules below are applied after
inspecting the signature and the docstring. A class is checked through
the signature of its own `__init__` against the class docstring. Classes
that inherit their constructor are skipped.

Special treatment is given to parameters whose signature default is `None`
and which are reassigned in the function body to a list, dictionary or set
literal, the mutable-default argument idiom. The docstring is expected to
document that literal, not `None`, and the rules are applied to it. A
parameter reassigned to any other expression keeps `None` as its documented
default. The code AST is scanned to detect this idiom.

Rules
-----
1. Optional marker. The parameter type list must end with `, optional`,
   with no space between the type list and the comma. The marker states
   that the caller may omit the argument, which is exactly what a signature
   default means. Parameter entries documented without type specifications
   are not checked.
   Subtype: `defaults.missing_optional`.

2. Default line present. The description block must contain a line that
   starts with `Default:` followed by a space, matched case-sensitively.
   Subtype: `defaults.missing_default`.

3. Default line last. The `Default:` line must be the last non-empty line of
   the description block.
   Subtype: `defaults.default_not_last`.

4. No separate paragraph. A blank line must not precede the `Default:`
   line, unless it closes an indented block or a list, where
   reStructuredText requires it. Such a blank line is not reported. A
   list closes the block only when it is itself opened by a blank line
   or by the start of the description block, as reStructuredText
   requires. A list marker written directly below running text is part
   of that paragraph and does not open a list, so the blank line above
   the `Default:` line is reported.
   Subtype: `defaults.separate_paragraph`.

5. Default value correct. The text following `Default:` must match the
   default expression from the signature, or start with it followed by
   parenthetical extra text. The comparison is exact and is made against
   a canonical rendering of the expression, not against its source text,
   so `15*pq.ms` is documented as `Default: 15 * pq.ms` and
   `spikes="random"` as `Default: 'random'`. An integer literal is
   rendered as it is written. A float literal keeps the notation of the
   signature, canonicalized by these forms:

   - A missing fractional or integer digit is filled with a zero, so
     `1.` is documented as `Default: 1.0` and `.5` as `Default: 0.5`.
   - Scientific notation is preserved instead of expanded, and the
     exponent marker is lowercase, so `1E-5` is documented as
     `Default: 1e-5` and `1e3` as `Default: 1e3`.
   - The exponent carries no `+` sign and no leading zero, so `1e+05`
     is documented as `Default: 1e5`.
   - The mantissa of an exponent literal carries no trailing `.` and no
     trailing `.0`, so `1.e-3` and `1.0e-3` are both documented as
     `Default: 1e-3`, while `1.5e-3` keeps its fractional digit.

   The notation mirrors the signature in both directions: a signature
   written `0.001` rejects `Default: 1e-3`, and a signature written
   `1e-3` rejects `Default: 0.001`.
   Subtype: `defaults.default_mismatch`.

6. Extra text in parentheses. Any text after the default value must be
   enclosed in parentheses immediately after the value, such as
   `Default: -1 (the last axis)`. The form `Default: the last axis (-1)`
   is rejected. The parenthesis must fit on the `Default:` line; a longer
   explanation belongs in the `Notes` section, referenced from the
   parenthesis as in `Default: None (see Notes [1])`.
   Subtype: `defaults.bad_extra_text`.

7. No trailing period. The `Default:` line must not end with a period. The
   period is removed before rules 4 and 5 compare the text, so a value
   that is otherwise correct is reported by this rule alone.
   Subtype: `defaults.trailing_period`.

8. None default is annotated. A parameter whose signature default is
   `None` and which carries a type hint annotation must have a type hint
   annotation that admits `None`. These annotations can be written as
   `Optional[X]`, `X | None` or `Union[X, None]`. The type hint form
   `x: list[int] = None` is a type error under PEP 484, which dropped
   implicit optionality.
   Subtype: `defaults.none_default_not_optional`.

Multi-name headers
------------------
Several parameters might be grouped at once, as in
`t_start, t_stop : pq.Quantity`. Therefore, all elements in the group share
the description block and the `Default:` docstring text. In this extension,
each parameter is checked separately against its own signature default. A
group whose members carry different defaults will be flagged as a mismatch
against the single `Default:` line, which shows that the grouped parameters
should be documented separately.

Validation output
-----------------

Specific checks may be suppressed through `suppress_warnings` in `conf.py`::

    suppress_warnings = [
        'defaults.missing_optional',
        'defaults.missing_default',
        'defaults.default_not_last',
        'defaults.default_mismatch',
        'defaults.bad_extra_text',
        'defaults.separate_paragraph',
        'defaults.trailing_period',
        'defaults.none_default_not_optional',
    ]

Objects can be excluded from every check through `defaults_ignore` in
`conf.py`::

    defaults_ignore = [
        'elephant.statistics.cv',
        'elephant.trials',
    ]

An entry is the fully-qualified name reported by autodoc, matched exactly
or as a dotted prefix, so a module or a class name excludes everything
below it. This differs from `suppress_warnings`, which disables one rule
across all objects.

This extension requires Python 3.9 or newer, which provides `ast.unparse`.
"""

import ast
import copy
import inspect
import re
import textwrap

import sphinx.util.logging

logger = sphinx.util.logging.getLogger(__name__)

# Matches a numpydoc entry header: one or several parameter names separated by
# commas, optionally followed by a colon and the type field.
_HEADER_RE = re.compile(r'^(\w+(?:\s*,\s*\w+)*)\s*(?::\s*(.*))?$')

# Matches a numpydoc section underline of three dashes or more.
# This is used to detect the start of a new section, e.g., Parameters.
_DASHES_RE = re.compile(r'^-{3,}')

# Matches the optional marker at the end of a type field. The `\S` requires
# the comma to follow the type list directly, with no space before it.
_OPTIONAL_SUFFIX_RE = re.compile(r'\S, optional$')

# Matches a parenthetical suffix: optional leading whitespace, then a
# single balanced parenthesized group spanning the rest of the text, with
# at most one level of nesting. This is used to check that any extra text
# after the default value is enclosed in parentheses.
_PAREN_SUFFIX_RE = re.compile(r'^\s*\((?:[^()]|\([^()]*\))*\)\s*$')

# Matches a bullet or enumerated list marker at the start of a line, such
# as `* item`, `- item`, `1. item` or `(1) item`. The enumerator is a
# number or a single letter, so that prose beginning `Fig. ` or `ref. `
# is not taken for a list item. A blank line is required to close such a
# list before the text that follows it.
_LIST_ITEM_RE = re.compile(r'^([*+-]|\(?(?:[0-9]{1,3}|[a-zA-Z])[.)])\s')

# Matches the source spelling of a float literal: a mantissa of digits
# around an optional decimal point, followed by an optional exponent.
# The named groups carry the two parts to the canonical rendering. Digit
# separators are removed before the match, so they never reach a group.
_FLOAT_LITERAL_RE = re.compile(
    r'(?P<mantissa>[0-9_]*\.?[0-9_]*)(?:[eE](?P<exponent>[+-]?[0-9_]+))?')

_PARAM_SECTIONS = frozenset({'Parameters', 'Other Parameters'})


def _canonical_float(text):
    """
    Returns the canonical rendering of the source spelling of one float
    literal.

    The canonical forms are those of rule 5. A literal without an
    exponent gains the zero its integer or fractional part is missing,
    so `1.` becomes `1.0` and `.5` becomes `0.5`. A literal with an
    exponent keeps its scientific notation, written with a lowercase
    `e`, an exponent stripped of a `+` sign and of leading zeros, and a
    mantissa stripped of a trailing `.` or `.0`, so `1.0E+05` becomes
    `1e5`.

    Parameters
    ----------
    text : str
        The source spelling of a float literal, such as `1.0e-3`.

    Returns
    -------
    str or None
        The canonical rendering, or `None` when `text` is not the
        spelling of a float literal.
    """
    match = _FLOAT_LITERAL_RE.fullmatch(text.replace('_', ''))
    if match is None or not match.group('mantissa').strip('.'):
        return None
    mantissa = match.group('mantissa')
    exponent = match.group('exponent')

    if exponent is not None:
        # An integral mantissa carries no decimal separator.
        if mantissa.endswith('.'):
            mantissa = mantissa[:-1]
        elif '.' in mantissa and mantissa.partition('.')[2].rstrip('0') == '':
            mantissa = mantissa.partition('.')[0]
        if mantissa.startswith('.'):
            mantissa = f'0{mantissa}'
        sign = '-' if exponent.startswith('-') else ''
        digits = exponent.lstrip('+-').lstrip('0') or '0'
        return f'{mantissa}e{sign}{digits}'

    if mantissa.startswith('.'):
        mantissa = f'0{mantissa}'
    if mantissa.endswith('.'):
        mantissa = f'{mantissa}0'
    return mantissa


def _unparse_default(node, source):
    """
    Renders a default expression, keeping the notation of its float
    literals.

    `ast.unparse` renders a float constant through `repr`, which drops
    the notation the author wrote: `1e-5` becomes `1e-05` and `1.0e-3`
    becomes `0.001`. Every float constant is therefore replaced by an
    `ast.Name` holding the canonical rendering of its source spelling,
    which `ast.unparse` emits verbatim and never parenthesizes, so the
    surrounding expression keeps the normalization `ast.unparse`
    provides. A constant whose source segment is unavailable, or whose
    spelling `_canonical_float` does not recognize, is left in place and
    keeps the `ast.unparse` rendering.

    Parameters
    ----------
    node : ast.expr
        The default value expression. It is deep-copied, so the tree of
        the caller is not modified.
    source : str
        The source text `node` was parsed from, from which
        `ast.get_source_segment` recovers the spelling of each literal.

    Returns
    -------
    str
        The rendered default expression.
    """
    class _FloatRewriter(ast.NodeTransformer):
        # Swaps every float constant for its canonical source spelling.
        def visit_Constant(self, constant):
            if not isinstance(constant.value, float):
                return constant
            segment = ast.get_source_segment(source, constant)
            canonical = (_canonical_float(segment)
                         if segment is not None else None)
            if canonical is None:
                return constant
            return ast.copy_location(
                ast.Name(id=canonical, ctx=ast.Load()), constant)

    rewritten = _FloatRewriter().visit(copy.deepcopy(node))
    return ast.unparse(ast.fix_missing_locations(rewritten))


def _find_mutable_defaults(func_node, none_params, source):
    """
    Find the parameters of `func_node` that carry a mutable default.

    The mutable-default argument idiom assigns the real default in the
    function body::

        if not param:          # or: if param is None:
            param = <value>

    Only a simple, single-assignment body without an `else` or `elif`
    branch is recognized, and only when the assigned value is a list,
    dictionary or set display whose elements are all literals. These are
    the values that cannot be written as a signature default. A tuple is
    accepted as an element but not as the assigned value itself, since a
    tuple is immutable and belongs in the signature.

    Any other expression computes the default at call time and is skipped
    so that `None` remains the documented default, as in `t_start=None`
    where the body assigns `t_start = spiketrains[0].t_start`. These cases are
    expected to be described in the parameter description for the behavior
    of the function when the parameter takes the `None` value.

    Parameters
    ----------
    func_node : ast.FunctionDef | ast.AsyncFunctionDef
        The function node whose top-level body is scanned.
    none_params : set[str]
        The names of the parameters whose signature default is `None`.
    source : str
        The source text `func_node` was parsed from, passed on to
        `_unparse_default` to recover the spelling of float literals.

    Returns
    -------
    dict[str, str]
        The effective default expression, as rendered by
        `_unparse_default`, per parameter name.
    """
    def _is_literal(node):
        # Returns True when the node is a literal expression.
        # The caller restricts the assigned value to a mutable. This helper
        # checks that everything inside it is a literal.
        if isinstance(node, ast.Constant):
            return True

        # A negative or explicitly signed number, such as `-1`, parses as a
        # unary operation over a constant.
        if (isinstance(node, ast.UnaryOp)
                and isinstance(node.op, (ast.UAdd, ast.USub))
                and isinstance(node.operand, ast.Constant)):
            return True

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(_is_literal(element) for element in node.elts)

        # A dictionary display holds keys and values separately. A `**` entry
        # carries `None` as its key, which is not a literal.
        if isinstance(node, ast.Dict):
            return all(_is_literal(element)
                       for element in node.keys + node.values)

        return False

    # Extract reassignments in the mutable-default argument idiom.
    overrides = {}
    for stmt in func_node.body:
        # Scan only simple if statements without else branches.
        if not isinstance(stmt, ast.If) or stmt.orelse:
            continue

        # Check for a single assignment in the if body.
        if (len(stmt.body) != 1
                or not isinstance(stmt.body[0], ast.Assign)):
            continue
        assign = stmt.body[0]

        # Require a single name target matching one of the parameters with a
        # `None` default.
        if (len(assign.targets) != 1
                or not isinstance(assign.targets[0], ast.Name)):
            continue
        param_name = assign.targets[0].id

        if param_name not in none_params:
            continue

        # Check if the conditional matches the two recognized patterns.
        test = stmt.test

        # Pattern A: if not param:
        matched = (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id == param_name
        )

        # Pattern B: if param is None:
        if not matched:
            matched = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == param_name
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            )

        if not matched:
            continue

        # Record the effective default only when the reassignment builds a
        # mutable literal, the one value that cannot be written in the
        # signature. Any computed expression is a default determined at
        # call time and keeps `None` as the documented value.
        value = assign.value
        if (isinstance(value, (ast.List, ast.Dict, ast.Set))
                and _is_literal(value)):
            overrides[param_name] = _unparse_default(value, source)

    return overrides


def _annotation_admits_none(node):
    """
    Returns `True` when the type hint annotation for the parameter accepts
    `None` as a value.

    The spellings that admit `None` are `Optional[X]`, the PEP 604 union
    `X | None`, `Union[..., None]`, `Literal[..., None]`, and the
    unconstrained `Any` and `object`. `Annotated[X, ...]` resolves to `X`.
    Any other annotation, such as a bare `int` or a `list[int]`,
    excludes `None`. A string forward reference is not resolved and is
    treated as excluding `None`, as Elephant does not use them.

    Parameters
    ----------
    node : ast.expr
        The annotation node to inspect.

    Returns
    -------
    bool
        `True` when `None` is a member of the annotated value domain.
    """
    # Handle subscription-based annotations like `Optional[X]` or
    # `Union[X, Y]` which appear as `ast.Subscript` nodes.
    if isinstance(node, ast.Subscript):
        # Extract the base name (possibly qualified) and consider only the
        # trailing identifier, e.g., `typing.Optional` -> `Optional`.
        base = ast.unparse(node.value).split('.')[-1]

        # The slice is a Tuple for several items and a single node for one.
        elements = (node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice])

        # `Optional[...]` explicitly admits None.
        if base == 'Optional':
            return True

        # `Union[...]` and `Literal[...]` may carry None among their
        # members; check each element recursively.
        if base in ('Union', 'Literal'):
            return any(_annotation_admits_none(el) for el in elements)

        # `Annotated[X, ...]` carries the annotation as its first element
        # and metadata after it.
        if base == 'Annotated':
            return _annotation_admits_none(elements[0])

        # Other subscripted types (e.g., `list[int]`) do not admit None by
        # default under PEP 484 rules.
        return False

    # Handle PEP 604 union syntax `X | None`, represented as a binary
    # operation with the `BitOr` operator. Recurse into both sides.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (_annotation_admits_none(node.left)
                or _annotation_admits_none(node.right))

    # `Any` and `object` admit every value, `None` among them.
    if isinstance(node, (ast.Name, ast.Attribute)):
        return ast.unparse(node).split('.')[-1] in ('Any', 'object')

    # Finally, a literal `None` in the annotation is an `ast.Constant`
    # whose value is `None`.
    return isinstance(node, ast.Constant) and node.value is None


def _read_signature(obj):
    """
    Reads the callable signature and extracts the information on default
    values for all the parameters. Use the AST to facilitate extraction of the
    mutable-default argument idiom and complex expressions in the default
    assignment, such as `0.5 * pq.ms` (that would otherwise evaluate to
    `array(0.5) * ms`).

    Parameters
    ----------
    obj : callable
        The function or method whose source is parsed.

    Returns
    -------
    defaults : dict[str, str]
        The effective default expression per parameter name, as rendered
        by `_unparse_default`. A signature default of `None` is replaced
        by the value obtained from the code in the function body if the
        parameter value is reassigned (i.e., the mutable-default
        argument idiom).
    none_defaulted : set[str]
        The names of the parameters whose default is literally `None` in
        the signature, before that replacement. A parameter that uses the
        mutable-default argument idiom stays in this set. This is used to
        inspect proper type hint annotations for the function.
    annotations : dict[str, ast.expr]
        The AST node representing the type hint annotation per parameter name.

    Notes
    -----
    The empty result `({}, set(), {})` is returned on any failure, such as
    unavailable source code for `obj`, a syntax error, or if `obj` is not a
    Python function/method.
    """
    # Parse `obj` source code into an AST. If source retrieval or
    # parsing fails (builtins, C extensions, or dynamic objects), return
    # the empty result.
    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        logger.debug(f'defaults: no source for '
                     f'{getattr(obj, "__qualname__", obj)} ({error})')
        return {}, set(), {}

    # Walk the AST looking for the first function/async function
    # definition node. We stop at the first encountered function node so
    # the code works when the provided source contains decorators or
    # wrapper code.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Extract the AST nodes representing the default values for positional
        # arguments. `args.defaults` cover the values for the last n
        # positional parameters.
        args = node.args
        positional_args = args.posonlyargs + args.args
        n_pos_defaults = len(args.defaults)
        defaulted = (
            list(zip(positional_args[-n_pos_defaults:], args.defaults))
            if args.defaults else []
        )

        # Extract the AST nodes representing the default values for keyword
        # arguments. `args.kw_defaults` contains the values and is
        # parallel to `args.kwonlyargs`, with `None` marking a keyword-only
        # parameter that has no default.
        defaulted += [(keyword_arg, default)
                      for keyword_arg, default in
                      zip(args.kwonlyargs, args.kw_defaults)
                      if default is not None]

        # Reconstruct the textual default expression per parameter.
        # This renders the actual expression as a string (e.g.,
        # `None` -> 'None', `0.5 * pq.ms` -> '0.5 * pq.ms') for comparison
        # against the docstring text.
        defaults = {arg.arg: _unparse_default(default_value, source)
                    for arg, default_value in defaulted}

        # Collect type hint annotations as AST nodes for later checks.
        annotations = {arg.arg: arg.annotation
                       for arg in positional_args + args.kwonlyargs
                       if arg.annotation}

        # Identify parameters whose default values are literally `None`
        # in the signature, and retrieve values defined using the
        # mutable-default idiom in the function body. These parameters have
        # `None` as value in the signature but are presented with a different
        # value in the docstring. This is used for defaults that are lists,
        # such as `[2, 3, 4, 5]`, as mutables cannot be used safely in
        # default assignments in function signatures.
        none_defaulted = {name for name, value in defaults.items()
                          if value == 'None'}
        if none_defaulted:
            defaults.update(
                _find_mutable_defaults(node, none_defaulted, source))

        return defaults, none_defaulted, annotations

    # No suitable function node found in the source. Return the empty result.
    return {}, set(), {}


def _scan_parameter_entries(lines):
    """
    Scan the docstring lines for the documented parameters.

    Only the `Parameters` and `Other Parameters` sections are scanned. A
    parameter description grouping several parameters (e.g.,
    `param_1, param_2 : types`) yields one entry per name, each
    carrying the shared parameter type list and the parameter description text.

    Parameters
    ----------
    lines : list[str]
        The raw docstring lines handed over by autodoc.

    Returns
    -------
    list[dict]
        One dictionary per documented parameter name, with the keys:

        name
            The parameter name.
        type
            The type list for the parameter, or `''` when it carries none.
        has_default_line
            `True` when the parameter description holds a line that starts
            with `Default:`, matched case-sensitively.
        default_is_last
            `True` when `has_default_line` is `True` and the `Default:`
            line is the last non-empty line of the description block.
        default_value
            The stripped text that follows `Default:` on that line, or
            `None` when the description has no `Default:` line.
        default_line_idx
            The index of the `Default:` line in `lines`, or `None` when
            the description has no such line.
    """
    entries = []
    current_section = None
    # `line_idx` walks the docstring `lines` top to bottom.
    line_idx = 0

    while line_idx < len(lines):
        stripped = lines[line_idx].strip()

        # A non-indented line followed by a dashes underline opens a new
        # section, whatever its name. Advance to the section body.
        if (stripped
                and lines[line_idx][0] not in (' ', '\t')
                and line_idx + 1 < len(lines)
                and lines[line_idx + 1][:1] == '-'
                and _DASHES_RE.match(lines[line_idx + 1].strip())):
            current_section = stripped
            line_idx += 2
            continue

        # If we are in a Parameters section and the line is not empty,
        # check the entry for each parameter. The header of a parameter
        # entry is not indented, while the description block below it is
        # indented.
        if (current_section in _PARAM_SECTIONS and stripped
                and lines[line_idx][0] not in (' ', '\t')):

            # Detect an entry header like "name, other : types".
            match = _HEADER_RE.match(stripped)
            if match:
                # Split the possibly comma-separated parameter names and
                # capture the optional type field following the colon.
                names = [name.strip()
                         for name in match.group(1).split(',')]
                param_type = (match.group(2) or '').strip()

                # Walk the indented description block below the header.
                # Collect information about a `Default:` line if present.
                # `desc_idx` scans the indented description lines below the
                # parameter header.
                desc_idx = line_idx + 1
                has_default_line = False
                default_is_last = False
                default_value = None
                default_line_idx = None
                # `desc_end` tracks the index after the last non-empty
                # line in the description block so we can check whether
                # the `Default:` line is the last one.
                desc_end = line_idx + 1

                while desc_idx < len(lines):
                    line = lines[desc_idx]
                    line_stripped = line.strip()
                    # A non-empty, non-indented line signals the next
                    # header or section; stop scanning the description.
                    if line_stripped and line[0] not in (' ', '\t'):
                        break
                    if line_stripped:
                        # Update the end marker to include this non-empty
                        # description line.
                        desc_end = desc_idx + 1
                        # Detect a `Default:` line case-sensitively and
                        # capture the text that follows the colon.
                        if line_stripped.startswith('Default: '):
                            has_default_line = True
                            default_value = (
                                line_stripped[len('Default: '):].strip())
                            default_line_idx = desc_idx
                    desc_idx += 1

                # Mark whether the `Default:` line is the very last
                # non-empty line of the description block.
                if has_default_line:
                    default_is_last = (default_line_idx + 1 == desc_end)

                # One entry per parameter name sharing this description
                # block. Store the extracted metadata for further checks.
                entries.extend({
                    'name': name,
                    'type': param_type,
                    'has_default_line': has_default_line,
                    'default_is_last': default_is_last,
                    'default_value': default_value,
                    'default_line_idx': default_line_idx,
                } for name in names)

                # Advance the main index past the description we consumed
                # and continue scanning the rest of the docstring.
                line_idx = desc_idx
                continue

        line_idx += 1

    return entries


def _validate_defaults(app, what, name, obj, options, lines):
    """
    Check if the docstring contains correct information regarding any default
    value for a parameter in the signature of a method or function.

    A warning is emitted per violated rule, carrying the `defaults` type
    and the rule-specific subtype.

    Parameters
    ----------
    app : sphinx.application.Sphinx or None
        The Sphinx application emitting the `autodoc-process-docstring`
        event, which supplies the `defaults_ignore` configuration. It is
        None when the handler is called directly, outside a build.
    what : str
        The kind of object being documented. Only `'function'`, `'method'`,
        `'class'` and `'exception'` are processed. The latter two are
        processed through their own `__init__` methods to validate the
        parameters of their constructor.
    name : str
        The fully-qualified name of the documented object.
    obj : object
        The documented object itself.
    options : dict
        The options given to the autodoc directive.
    lines : list[str]
        The raw docstring lines, left unmodified by this handler.
    """
    # Skip objects listed in `defaults_ignore`. An entry matches the
    # fully-qualified name reported by autodoc, either exactly or as a
    # dotted prefix, so a module or a class name covers everything below
    # it. `app` is None when the handler is called outside a build.
    ignored_names = app.config.defaults_ignore if app is not None else ()
    for ignored in ignored_names:
        if name == ignored or name.startswith(f'{ignored}.'):
            return

    # Only validate functions, methods and classes. A class documents the
    # parameters of its constructor in the class docstring, so the defaults
    # are read from `__init__`. A class that inherits its constructor is
    # skipped, as those parameters belong to the parent class docstring.
    if what in ('class', 'exception'):
        if '__init__' not in vars(obj):
            return
        signature_obj = obj.__init__
    elif what in ('function', 'method'):
        signature_obj = obj
    else:
        return

    # Read the signature: effective defaults, which parameters had
    # `None` as the default, and any type hint annotations present.
    defaults, none_defaulted, annotations = _read_signature(signature_obj)

    # Skip validation if this function/method does not have defaults.
    if not defaults:
        return

    # Get the source code information for more informative warnings. The
    # location describes the object whose docstring is validated, not the
    # constructor whose signature was read. The object is unwrapped first
    # because `inspect.getsourcelines` follows `__wrapped__` while
    # `inspect.getfile` does not, so a decorated function would report the
    # line of its definition in the file of its decorator.
    try:
        unwrapped = inspect.unwrap(obj)
        source_file = inspect.getfile(unwrapped)
        source_line = inspect.getsourcelines(unwrapped)[1]
        location = f'{source_file}:{source_line}'
    except (OSError, TypeError):
        location = name

    # Iterate over the parameter entries parsed from the docstring text and
    # check each one against the defaults in the function signature.
    for entry in _scan_parameter_entries(lines):
        param_name = entry['name']
        if param_name not in defaults:
            # This is the documentation for a parameter that has no default in
            # the signature. Exit as there is nothing to validate.
            continue

        default_str = defaults[param_name]
        param_type = entry['type']

        # Rule 1: ", optional" must follow the type list directly. No spaces
        # are allowed between the comma and the type list.
        if param_type and not _OPTIONAL_SUFFIX_RE.search(param_type):
            logger.warning(
                f'[defaults] ({location}) '
                f'{name}: parameter `{param_name}` has '
                f'default `{default_str}` in the signature but its '
                f'docstring type information `{param_type}` does not end '
                f'with `, optional` without spaces between the type list '
                f'and the comma.',
                type='defaults',
                subtype='missing_optional',
            )

        # Rule 2: the description block must contain a `Default:` line.
        if not entry['has_default_line']:
            logger.warning(
                f'[defaults] ({location}) '
                f'{name}: parameter `{param_name}` has '
                f'default `{default_str}` in the signature but its '
                f'description has no valid `Default:` line. The marker '
                f'must be followed by a space.',
                type='defaults',
                subtype='missing_default',
            )
            # Further default-related checks require a `Default:` line.
            continue

        # Rule 3: `Default:` must be the last non-empty line of the
        # parameter description block.
        if not entry['default_is_last']:
            logger.warning(
                f'[defaults] ({location}) '
                f'{name}: parameter `{param_name}` has a '
                f'`Default:` line that is not the last non-empty line '
                f'of its description block.',
                type='defaults',
                subtype='default_not_last',
            )

        # Rule 4: the `Default:` line must not be preceded by a blank line
        # in the docstring. A blank line that closes an indented block or a
        # list is required by reStructuredText and is not reported.
        default_line_idx = entry['default_line_idx']
        if (default_line_idx is not None and default_line_idx > 0
                and not lines[default_line_idx - 1].strip()):
            default_line = lines[default_line_idx]
            indent = len(default_line) - len(default_line.lstrip())

            # Walk back to the last non-empty line above the blank one.
            previous_idx = default_line_idx - 1
            while previous_idx > 0 and not lines[previous_idx].strip():
                previous_idx -= 1
            previous = lines[previous_idx]
            previous_indent = len(previous) - len(previous.lstrip())

            # A deeper indented line belongs to a block, which needs the
            # blank line that closes it.
            closes_block = previous_indent > indent

            # A list marker at the same indentation closes a list only when
            # the list was opened. reStructuredText requires a blank line
            # before the first item, so a marker written directly below
            # running text is plain text within that paragraph. Walk up the
            # run of non-empty lines and check that a list item starts it.
            if not closes_block and _LIST_ITEM_RE.match(previous.lstrip()):
                run_idx = previous_idx
                while (run_idx > 0 and lines[run_idx - 1].strip()
                       and lines[run_idx - 1][0] in (' ', '\t')):
                    run_idx -= 1
                run_start = lines[run_idx]
                run_indent = len(run_start) - len(run_start.lstrip())
                closes_block = (run_indent == indent
                                and bool(_LIST_ITEM_RE.match(
                                    run_start.lstrip())))
            if not closes_block:
                logger.warning(
                    f'[defaults] ({location}) '
                    f'{name}: parameter `{param_name}` has '
                    f'a `Default:` line that is preceded by a blank line.',
                    type='defaults',
                    subtype='separate_paragraph',
                )

        # Capture the documented default text. Rules 5 and 6 compare it
        # without a trailing period, which rule 7 reports on its own.
        doc_default = entry['default_value']
        if doc_default is not None:
            doc_value = doc_default.removesuffix('.')

            # Rule 5: the documented default must match the signature
            # default, or start with it followed by parenthetical extra
            # information.
            value_ok = (doc_value == default_str
                        or doc_value.startswith(f'{default_str} '))
            if not value_ok:
                logger.warning(
                    f'[defaults] ({location}) '
                    f'{name}: parameter `{param_name}` '
                    f'documents `Default: {doc_default}` but the '
                    f'signature default is `{default_str}`.',
                    type='defaults',
                    subtype='default_mismatch',
                )
            else:
                # Rule 6: if there is extra text after the value, it must
                # be enclosed in parentheses immediately following the
                # value.
                suffix = doc_value[len(default_str):]
                if suffix and not _PAREN_SUFFIX_RE.match(suffix):
                    logger.warning(
                        f'[defaults] ({location}) '
                        f'{name}: parameter `{param_name}` '
                        f'has extra text after the default value that is '
                        f'not enclosed in parentheses: '
                        f'`Default: {doc_default}`.',
                        type='defaults',
                        subtype='bad_extra_text',
                    )

            # Rule 7: the `Default:` line should not end with a period.
            if doc_default.endswith('.'):
                logger.warning(
                    f'[defaults] ({location}) '
                    f'{name}: parameter `{param_name}` has '
                    f'a `Default:` line that ends with a period: '
                    f'`Default: {doc_default}`.',
                    type='defaults',
                    subtype='trailing_period',
                )

    # Rule 8: ensure that type hint annotations of parameters whose signature
    # default is literally `None` explicitly admit `None`. This check is
    # independent of docstring text and only validates parameters with type
    # hints.
    for param_name in sorted(none_defaulted):
        annotation = annotations.get(param_name)
        if annotation is None or _annotation_admits_none(annotation):
            continue
        logger.warning(
            f'[defaults] ({location}) '
            f'{name}: parameter `{param_name}` has the '
            f'signature default `None` but its annotation '
            f'`{ast.unparse(annotation)}` does not admit `None`. To fix this,'
            f' write `Optional[...]` or `... | None` as type hint.',
            type='defaults',
            subtype='none_default_not_optional',
        )


def setup(app):
    """
    Register the extension with Sphinx.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application to register with.

    Returns
    -------
    dict
        The extension metadata, declaring the version and the parallel
        read and write safety.
    """
    # Objects excluded from every check, as fully-qualified names.
    app.add_config_value('defaults_ignore', [], 'env', types=[list])

    # Priority 200 runs the handler before numpydoc, whose default
    # priority is 500, so that it sees the raw numpy-format lines.
    app.connect('autodoc-process-docstring', _validate_defaults, priority=200)
    return {
        'version': '0.3',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
