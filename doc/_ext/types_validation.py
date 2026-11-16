"""
Sphinx extension that validates the type fields documented for parameters,
returns, yields and attributes.

The type field of every numpydoc entry is checked against one grammar and,
where the signature carries a type hint annotation, against that
annotation. A class is checked through the signature of its own `__init__`
against the class docstring. Classes that inherit their constructor are
skipped. A property is checked through the signature of its `fget`
function, which carries no documented parameter, so only the format rules
and the return comparison reach it.

The extension compares the text as written. It resolves no aliases and
treats no two spellings of a name as equivalent, so `neo.core.SpikeTrain`
in a signature does not satisfy `neo.SpikeTrain` in a docstring, and
`List[int]` does not satisfy `list[int]`. Such a pair is reported, and the
fix belongs in the source. The only transformations applied are structural,
where the annotation grammar and the docstring grammar express the same
thing in different syntax:

- `Optional[X]` and `Union[A, B]` are flattened to the union members `X`,
  `None`, `A` and `B`, because rule 1 requires the docstring to spell a
  union with ` | `.
- `Literal['a', 'b']` yields the set member `{'a', 'b'}`, which is rule 6.
- `Annotated[X, ...]` yields the members of `X`.
- A string annotation is a forward reference and contributes its text, so
  `'Trials'` is the written name `Trials`.

Union members are compared without regard to their order.

Non-goal
--------
Every check is decidable from the written text alone. Validating the classes
and prefixes is out of scope and is left to review: whether `integer` should
read `int`, whether `ints` should read `int`, whether a module prefix such as
`numpy.` should read `np.`, and whether a bare `BinnedSpikeTrain` should carry
its `elephant.conversion.` prefix. This affects cross-references, and checking
for invalid classes will be the responsibility of another extension.

Grammar
-------
A type field is `TYPELIST[, optional]`, where `TYPELIST` is one or more
members separated by ` | `. A member is one of:

- an optional shape group, one parenthesized group of dimensions followed
  by a single space, then a qualified name, as in `(N, M) np.ndarray` or
  `(N,) elephant.conversion.BinnedSpikeTrain`;
- a qualified name, optionally subscripted, as in `int`, `pq.Quantity`,
  `neo.SpikeTrain`, `list[int]`, `dict[str, int]` or
  `list[list[neo.SpikeTrain]]`;
- a set of literals, as in `{'a', 'b', 'c'}` or `{1, 2}`;
- `None`.

The `, optional` marker is parsed and stripped, but its presence is not
enforced here. That is a rule of the `defaults` validation, and enforcing
it in both places would report every omission twice.

Rules
-----
1. Union separator. Members of a union are separated by ` | `, one space
   on each side of the bar. A separating `or`, a comma that is neither the
   `, optional` marker nor part of a subscript, a set or a shape group, and
   a bar that does not carry exactly one space on each side, are reported
   together with the corrected type list.
   Subtype: `types.or_separator`.

2. Element type by subscript. The element type of a container is written as
   a subscript, so `list of int` is written `list[int]`. The array form
   `np.ndarray of shape (N, M)` is written with the shape as a prefix,
   `(N, M) np.ndarray`.
   Subtype: `types.container_of`.

3. No `typing` spelling. The type field does not carry `List`, `Dict`,
   `Tuple`, `Set`, `FrozenSet`, `Type`, `Text`, `Optional`, `Union` or
   `Literal`. The required spelling is the builtin for the container
   generics, ` | None` for `Optional`, ` | ` for `Union` and a `{...}` set
   for `Literal`.
   Subtype: `types.typing_spelling`.

4. No Sphinx markup. The type field is plain text and carries no reST role
   such as `:class:` and no backticks. The cross-reference links are
   applied by the documentation build, so the source docstring stays
   readable.
   Subtype: `types.sphinx_markup`.

5. Annotation agreement. Where a parameter carries a type hint annotation,
   the documented type list holds the same members as the annotation, and a
   parameter that carries an annotation documents a type field at all. The
   same comparison runs in the return position: where a function, a method
   or a property carries a return annotation, the single entry of its
   `Returns` section holds the same members, and an object that carries a
   return annotation documents a return type field at all. A `Returns`
   section of several entries is left alone, because a tuple documented
   entry by entry has no single type field to compare the one annotation
   against. The message prints both sides as written.
   Subtype: `types.annotation_mismatch`.

6. Literal as a set. Where the annotation is `Literal[...]`, the documented
   type list holds the corresponding `{...}` set. This is the specific form
   of rule 5 and is reported in its place.
   Subtype: `types.literal_not_set`.

7. Well-formed members. Every member parses under the grammar above. A
   field that does not parse, and that none of rules 1 to 4 already
   explains, is reported here.
   Subtype: `types.malformed`.

8. Shape group. The dimensions of a shape group are separated by a comma
   followed by exactly one space, and exactly one space separates the
   closing parenthesis from the type name. A dimension is a single word or
   an integer, or several of those joined by `*` without surrounding
   spaces. A group of one dimension is the Python tuple spelling `(N,)`,
   the one place a comma is not followed by a space.
   Subtype: `types.shape_group`.

9. Set spelling. A set of literals is written the way Python writes it:
   single quotes around a string member and `, ` between the members, with
   no padding inside the braces. The written set is reported together with
   its corrected spelling. A set whose members are separated by a bare
   comma does not parse under the grammar and is reported by rule 7
   instead, so that a field is not explained twice.
   Subtype: `types.set_quotes`.

Rules 5 and 6 apply to the `Parameters` and `Other Parameters` sections,
matched against the signature. Rule 5 also applies to `Returns`, matched
against the return annotation. Rules 1, 2, 3, 4, 7, 8 and 9 apply to the
parameter sections and to `Returns`, `Yields` and `Attributes`.

Rules 1 to 4 are spelling rules, and so are rules 8 and 9. When any of them
fires on an entry, or when rule 7 fires, the annotation comparison for that
entry is skipped, so a field such as `tuple of :class:`pq.Quantity`` yields
two precise warnings instead of those two plus a mismatch that repeats
them. The comparison runs again once the spelling is fixed.

Multi-name headers
------------------
Several parameters might be grouped at once, as in
`t_start, t_stop : pq.Quantity`. The group shares one type field, which is
checked once against the spelling rules and then separately against the
annotation of each member of the group. A group whose members carry
different annotations is therefore reported against the single type field,
which shows that the grouped parameters should be documented separately.

Validation output
-----------------

Specific checks may be suppressed through `suppress_warnings` in `conf.py`::

    suppress_warnings = [
        'types.or_separator',
        'types.container_of',
        'types.typing_spelling',
        'types.sphinx_markup',
        'types.annotation_mismatch',
        'types.literal_not_set',
        'types.malformed',
        'types.shape_group',
        'types.set_quotes',
    ]

Objects can be excluded from every check through `types_ignore` in
`conf.py`::

    types_ignore = [
        'elephant.statistics.cv',
        'elephant.trials',
    ]

An entry is the fully-qualified name reported by autodoc, matched exactly
or as a dotted prefix, so a module or a class name excludes everything
below it. This differs from `suppress_warnings`, which disables one rule
across all objects. There is no alias configuration, because the extension
resolves no equivalence between two spellings of a name.

This extension requires Python 3.9 or newer, which provides `ast.unparse`.
"""

import ast
import inspect
import re
import textwrap
import warnings

import sphinx.util.logging
from numpydoc.docscrape import NumpyDocString, ParseError
from numpydoc.xref import QUALIFIED_NAME_RE, ROLE_SPLIT_RE

logger = sphinx.util.logging.getLogger(__name__)

# Matches a member separator at the top level of a type list: the required
# ` | `, and the `or` and comma spellings that rule 1 reports.
_MEMBER_SEP_RE = re.compile(r'\s*\|\s*|\s*,\s*|\s+or\s+')

# Matches the `of` element-type separator that rule 2 reports.
_OF_SEP_RE = re.compile(r'\s+of\s+')

# Matches the shape prefix of a member: one parenthesized group followed by
# whitespace. The content of the group is not inspected.
_SHAPE_PREFIX_RE = re.compile(r'^\([^()]*\)\s+')

# Matches the same shape prefix, capturing the content of the group and the
# whitespace that separates it from the type name, which rule 8 inspects.
_SHAPE_GROUP_RE = re.compile(r'^\(([^()]*)\)(\s+)')

# Matches the `shape (N, M)` tail of the `np.ndarray of shape (N, M)` form,
# capturing the parenthesized group that becomes the shape prefix.
_SHAPE_TAIL_RE = re.compile(r'^shape\s+(\([^()]*\))$')

# Matches a comma that is not written as `, `: one preceded by whitespace,
# or one not followed by a single space. The grammar spells the elements of
# a subscript or a set with `, `, which is also what `ast.unparse` emits.
_TIGHT_COMMA_RE = re.compile(r'\s,|,(?![ ]|$)')

# Matches the `typing` spellings rejected by rule 3.
_TYPING_RE = re.compile(
    r'\b(List|Dict|Tuple|Set|FrozenSet|Type|Text|Optional|Union|Literal)\b')

# The required spelling per rejected `typing` name.
_TYPING_REPLACEMENTS = {
    'List': '`list`',
    'Dict': '`dict`',
    'Tuple': '`tuple`',
    'Set': '`set`',
    'FrozenSet': '`frozenset`',
    'Type': '`type`',
    'Text': '`str`',
    'Optional': '` | None`',
    'Union': '` | `',
    'Literal': 'a `{...}` set',
}

_PARAM_SECTIONS = ('Parameters', 'Other Parameters')
_TYPE_SECTIONS = _PARAM_SECTIONS + ('Returns', 'Yields', 'Attributes')

_OPTIONAL_SUFFIX = ', optional'


def _split_top_level(text, pattern):
    """
    Split `text` at every match of `pattern` that lies outside brackets,
    braces, parentheses and quotes.

    Parameters
    ----------
    text : str
        The text to split, such as a type list or one of its members.
    pattern : re.Pattern
        The separator pattern, matched at each candidate position with
        `re.Pattern.match`.

    Returns
    -------
    parts : list[str]
        The text between the separators. It holds one element more than
        `separators` and is never empty.
    separators : list[str]
        The matched separator text, in the order it occurs.
    """
    parts = []
    separators = []
    depth = 0
    quote = None
    start = 0
    index = 0

    while index < len(text):
        character = text[index]

        # Inside a quoted literal, only the closing quote is meaningful.
        if quote is not None:
            if character == quote:
                quote = None
            index += 1
            continue

        if character in '\'"':
            quote = character
        elif character in '([{':
            depth += 1
        elif character in ')]}':
            depth -= 1
        elif depth == 0:
            match = pattern.match(text, index)
            if match is not None:
                parts.append(text[start:index])
                separators.append(match.group())
                index = match.end()
                start = index
                continue

        index += 1

    parts.append(text[start:])
    return parts, separators


def _is_member(text, allow_shape=True):
    """
    Returns `True` when `text` is one member of the type list grammar.

    A member is `None`, a set of literals, a qualified name, or a qualified
    name subscripted with further members. A shape prefix is accepted only
    at the outer level, since an element of a subscript carries none.

    Parameters
    ----------
    text : str
        The member text, as written in the docstring.
    allow_shape : bool
        Whether a leading parenthesized shape group is accepted.
        Default: True

    Returns
    -------
    bool
        `True` when `text` parses under the grammar.
    """
    text = text.strip()
    if allow_shape:
        text = _SHAPE_PREFIX_RE.sub('', text, count=1).strip()
    if not text or _TIGHT_COMMA_RE.search(text):
        return False

    try:
        node = ast.parse(text, mode='eval').body
    except SyntaxError:
        return False

    # The only bare constant the grammar admits is `None`.
    if isinstance(node, ast.Constant):
        return node.value is None

    # A set display holds the literals of rule 6.
    if isinstance(node, ast.Set):
        return all(isinstance(element, ast.Constant)
                   for element in node.elts)

    # A subscript carries a qualified name as its base and members as its
    # elements. The slice is a tuple for several elements and a single node
    # for one.
    if isinstance(node, ast.Subscript):
        if QUALIFIED_NAME_RE.match(ast.unparse(node.value)) is None:
            return False
        elements = (node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice])
        return all(_is_member(ast.unparse(element), allow_shape=False)
                   for element in elements)

    return QUALIFIED_NAME_RE.match(text) is not None


def _is_dimension(text):
    """
    Returns `True` when `text` is one dimension of a shape group.

    A dimension is a single word or an integer literal, or several of those
    joined by `*` without surrounding spaces, as in `N`, `3` and `xDim*T`.
    The absence of whitespace is checked directly, because `ast` accepts
    `N * T` as readily as `N*T`.

    Parameters
    ----------
    text : str
        The dimension text, as written in the docstring.

    Returns
    -------
    bool
        `True` when `text` is a dimension.
    """
    if not text or re.search(r'\s', text):
        return False

    try:
        node = ast.parse(text, mode='eval').body
    except SyntaxError:
        return False

    # The nodes a product of words and integers is built from. `ast.Load`
    # is the context every name carries and `ast.Mult` the operator.
    return all(
        isinstance(child, (ast.Name, ast.Mult, ast.Load))
        or (isinstance(child, ast.BinOp)
            and isinstance(child.op, ast.Mult))
        or (isinstance(child, ast.Constant)
            and isinstance(child.value, int))
        for child in ast.walk(node))


def _shape_group_problem(text):
    """
    Returns the defect of the shape group of one type list member.

    A well-formed group separates its dimensions by a comma followed by
    exactly one space and is separated from the type name by exactly one
    space. A group of one dimension carries the trailing comma of the
    Python tuple spelling, `(N,)`, which is the one comma not followed by
    a space.

    Parameters
    ----------
    text : str
        The member text, as written in the docstring.

    Returns
    -------
    str or None
        A sentence naming the defect and the correction, or `None` when
        `text` carries no shape group or a well-formed one.
    """
    text = text.strip()
    match = _SHAPE_GROUP_RE.match(text)
    if match is None:
        return None

    content = match.group(1)
    spacing = match.group(2)
    type_name = text[match.end():]

    # A trailing comma is the one-dimensional tuple spelling `(N,)` and
    # contributes no dimension of its own.
    elements = content.split(',')
    if len(elements) > 1 and not elements[-1].strip():
        elements = elements[:-1]
    dimensions = [element.strip() for element in elements]

    for dimension in dimensions:
        if not _is_dimension(dimension):
            return (f'the shape group of `{text}` holds `{dimension}`, '
                    f'which is not a dimension. A dimension is a single '
                    f'word or an integer, or several of those joined by '
                    f'`*`.')

    # One dimension keeps the trailing comma of the tuple spelling; several
    # are separated by a comma and exactly one space.
    corrected_content = (f'{dimensions[0]},' if len(dimensions) == 1
                         else ', '.join(dimensions))
    if corrected_content != content or spacing != ' ':
        return (f'the shape group of `{text}` is written '
                f'`({corrected_content}) {type_name}`.')

    return None


def _member_text(text):
    """
    Returns the comparable rendering of one member of a type list.

    The shape prefix is dropped, because an annotation cannot express it,
    and the remainder is rendered with `ast.unparse` so that the spacing
    inside a subscript or a set matches the spacing of the annotation side.
    Nothing else is changed: the names keep the spelling of the docstring.

    Parameters
    ----------
    text : str
        The member text, as written in the docstring.

    Returns
    -------
    str
        The member rendered for comparison, or the stripped text when it
        does not parse as a Python expression.
    """
    text = _SHAPE_PREFIX_RE.sub('', text.strip(), count=1).strip()
    try:
        return ast.unparse(ast.parse(text, mode='eval').body)
    except SyntaxError:
        return text


def _annotation_members(node):
    """
    Returns the members of the union expressed by a type hint annotation.

    The four structural spellings that a docstring type list cannot
    reproduce are expanded: `Optional[X]` and `Union[A, B]` become their
    union members, `Literal[...]` becomes the equivalent `{...}` set,
    `Annotated[X, ...]` becomes the members of `X`, and a string forward
    reference contributes its text. Every other node is rendered with
    `ast.unparse` and keeps the spelling of the signature.

    Parameters
    ----------
    node : ast.expr
        The annotation node to expand.

    Returns
    -------
    set[str]
        The member strings of the annotated value domain.
    """
    # Handle subscription-based annotations such as `Optional[X]` or
    # `Union[X, Y]`, which appear as `ast.Subscript` nodes.
    if isinstance(node, ast.Subscript):
        # Consider only the trailing identifier of the base, so that
        # `typing.Optional` is recognized as `Optional`.
        base = ast.unparse(node.value).split('.')[-1]

        # The slice is a tuple for several items and a single node for one.
        elements = (node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice])

        if base == 'Optional':
            return _annotation_members(elements[0]) | {'None'}

        if base == 'Union':
            return set().union(*(_annotation_members(element)
                                 for element in elements))

        # `Literal[...]` is documented as the set of its values.
        if base == 'Literal':
            values = ', '.join(ast.unparse(element) for element in elements)
            return {f'{{{values}}}'}

        # `Annotated[X, ...]` carries the annotation as its first element
        # and metadata after it.
        if base == 'Annotated':
            return _annotation_members(elements[0])

        return {ast.unparse(node)}

    # Handle the PEP 604 union syntax `X | None`, represented as a binary
    # operation with the `BitOr` operator.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (_annotation_members(node.left)
                | _annotation_members(node.right))

    # A string annotation is a forward reference and contributes its text.
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value.strip()}

    return {ast.unparse(node)}


def _read_signature(obj):
    """
    Read the type hint annotations from the signature of a callable.

    The AST is used rather than `inspect.signature` or
    `typing.get_type_hints`, because the source spelling is what has to be
    compared and because a forward reference would fail to resolve.

    Parameters
    ----------
    obj : callable
        The function or method whose source is parsed.

    Returns
    -------
    annotations : dict[str, ast.expr]
        The AST node of the type hint annotation per parameter name.
        Parameters without an annotation are absent.
    returns : ast.expr or None
        The AST node of the return annotation, or `None` when the
        signature carries none.

    Notes
    -----
    The empty result `({}, None)` is returned on any failure, such as
    unavailable source code for `obj`, a syntax error, or `obj` not being a
    Python function or method.
    """
    # Parse the source of `obj` into an AST. If source retrieval or parsing
    # fails (builtins, C extensions, or dynamic objects), return the empty
    # result.
    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        logger.debug(f'types: no source for '
                     f'{getattr(obj, "__qualname__", obj)} ({error})')
        return {}, None

    # Stop at the first function node, so that the code works when the
    # source carries decorators or wrapper code.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        arguments = args.posonlyargs + args.args + args.kwonlyargs
        return ({argument.arg: argument.annotation
                 for argument in arguments if argument.annotation},
                node.returns)

    # No suitable function node found in the source.
    return {}, None


def _validate_types(app, what, name, obj, options, lines):
    """
    Check the type fields of a docstring against the grammar and against
    the type hint annotations of the signature.

    A warning is emitted per violated rule, carrying the `types` type and
    the rule-specific subtype.

    Parameters
    ----------
    app : sphinx.application.Sphinx or None
        The Sphinx application emitting the `autodoc-process-docstring`
        event, which supplies the `types_ignore` configuration. It is None
        when the handler is called directly, outside a build.
    what : str
        The kind of object being documented. Only `'function'`,
        `'method'`, `'property'`, `'class'` and `'exception'` are
        processed. The latter two are processed through their own
        `__init__` methods to validate the parameters of their
        constructor, and a property through its `fget` function.
    name : str
        The fully-qualified name of the documented object.
    obj : object
        The documented object itself.
    options : dict
        The options given to the autodoc directive.
    lines : list[str]
        The raw docstring lines, left unmodified by this handler.
    """
    # Skip objects listed in `types_ignore`. An entry matches the
    # fully-qualified name reported by autodoc, either exactly or as a
    # dotted prefix, so a module or a class name covers everything below
    # it. `app` is None when the handler is called outside a build.
    ignored_names = app.config.types_ignore if app is not None else ()
    for ignored in ignored_names:
        if name == ignored or name.startswith(f'{ignored}.'):
            return

    # Only validate functions, methods, properties and classes. A class
    # documents the parameters of its constructor in the class docstring,
    # so the annotations are read from `__init__`. A class that inherits
    # its constructor is skipped, as those parameters belong to the parent
    # class docstring. A property carries its docstring and its return
    # annotation on `fget`, which is also where its source location is.
    location_obj = obj
    if what in ('class', 'exception'):
        if '__init__' not in vars(obj):
            return
        signature_obj = obj.__init__
    elif what == 'property':
        signature_obj = obj.fget
        location_obj = obj.fget
    elif what in ('function', 'method'):
        signature_obj = obj
    else:
        return

    annotations, return_annotation = _read_signature(signature_obj)

    # A class documents the parameters of its constructor, never a return
    # value, so the return comparison is limited to the callables.
    if what in ('class', 'exception'):
        return_annotation = None

    # numpydoc warns about section underline lengths while parsing, and
    # reports an unparseable docstring through its own validation, so this
    # handler stays silent on both.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            docstring = NumpyDocString('\n'.join(lines))
        except ParseError:
            return

    # Get the source code information for more informative warnings. The
    # location describes the object whose docstring is validated, not the
    # constructor whose signature was read. The object is unwrapped first
    # because `inspect.getsourcelines` follows `__wrapped__` while
    # `inspect.getfile` does not, so a decorated function would report the
    # line of its definition in the file of its decorator.
    try:
        unwrapped = inspect.unwrap(location_obj)
        source_file = inspect.getfile(unwrapped)
        source_line = inspect.getsourcelines(unwrapped)[1]
        location = f'{source_file}:{source_line}'
    except (OSError, TypeError):
        location = name

    # Whether the spelling of the single `Returns` entry is clean enough
    # for the return comparison to add information beyond the spelling
    # warnings already emitted.
    return_comparable = True

    for section in _TYPE_SECTIONS:
        for entry in docstring[section]:
            # A grouped header such as `t_start, t_stop : pq.Quantity`
            # arrives as one entry whose name holds every parameter. A
            # variadic parameter is looked up without its stars.
            entry_names = [part.strip().lstrip('*')
                           for part in entry.name.split(',')
                           if part.strip()]
            type_field = entry.type.strip()

            if section in _PARAM_SECTIONS:
                subject = (f'parameter `{entry.name}`' if entry.name
                           else 'an unnamed parameter')
            else:
                subject = (f'the `{section}` entry `{entry.name}`'
                           if entry.name else f'a `{section}` entry')

            # Rule 5, in the form of a parameter that carries an annotation
            # but documents no type at all.
            if not type_field:
                if section in _PARAM_SECTIONS:
                    for entry_name in entry_names:
                        annotation = annotations.get(entry_name)
                        if annotation is None:
                            continue
                        logger.warning(
                            f'[types] ({location}) '
                            f'{name}: parameter `{entry_name}` carries the '
                            f'annotation `{ast.unparse(annotation)}` but '
                            f'documents no type field.',
                            type='types',
                            subtype='annotation_mismatch',
                        )
                continue

            # `comparable` records whether the spelling of the type field
            # is clean enough for the annotation comparison to add
            # information beyond the spelling warnings already emitted.
            comparable = True

            # Rule 4: the type field is plain text.
            if ROLE_SPLIT_RE.search(type_field) or '`' in type_field:
                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: {subject} carries Sphinx markup in its type '
                    f'field `{type_field}`. The type field is plain text; '
                    f'the documentation build applies the cross-reference '
                    f'links.',
                    type='types',
                    subtype='sphinx_markup',
                )
                comparable = False

            # Rule 3: no `typing` spelling.
            typing_names = [spelling
                            for spelling in _TYPING_REPLACEMENTS
                            if spelling in set(_TYPING_RE.findall(
                                type_field))]
            if typing_names:
                wanted = ', '.join(
                    f'`{spelling}` (write {_TYPING_REPLACEMENTS[spelling]})'
                    for spelling in typing_names)
                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: {subject} uses the `typing` spelling '
                    f'{wanted} in its type field `{type_field}`.',
                    type='types',
                    subtype='typing_spelling',
                )
                comparable = False

            # Strip the optional marker before splitting the type list. Its
            # presence is a rule of the `defaults` validation.
            type_list = type_field
            is_optional = type_list.endswith(_OPTIONAL_SUFFIX)
            if is_optional:
                type_list = type_list[:-len(_OPTIONAL_SUFFIX)]

            members, separators = _split_top_level(type_list, _MEMBER_SEP_RE)

            # Rule 1: the members of a union are separated by ` | `, one
            # space on each side of the bar.
            if any(separator != ' | ' for separator in separators):
                corrected = ' | '.join(member.strip() for member in members)
                if is_optional:
                    corrected += _OPTIONAL_SUFFIX
                if all(separator.strip() == '|'
                       for separator in separators):
                    defect = (f'does not put exactly one space on each '
                              f'side of the `|` in its type field '
                              f'`{type_field}`')
                else:
                    defect = (f'separates the members of its type field '
                              f'`{type_field}` with `or` or a comma')
                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: {subject} {defect}. Write the union as '
                    f'`{corrected}`.',
                    type='types',
                    subtype='or_separator',
                )
                comparable = False

            # Rule 2: the element type of a container is a subscript, and
            # the shape of an array is a prefix.
            rewrites = []
            for member in members:
                parts, of_separators = _split_top_level(member.strip(),
                                                        _OF_SEP_RE)
                if not of_separators:
                    continue
                tail = parts[-1].strip()
                shape = _SHAPE_TAIL_RE.match(tail)
                if shape is not None and len(parts) == 2:
                    corrected_member = f'{shape.group(1)} {parts[0].strip()}'
                else:
                    # Nest the subscripts from the innermost element out,
                    # so that `list of list of int` becomes
                    # `list[list[int]]`.
                    corrected_member = tail
                    for outer in reversed(parts[:-1]):
                        corrected_member = (f'{outer.strip()}'
                                            f'[{corrected_member}]')
                rewrites.append((member.strip(), corrected_member))

            if rewrites:
                wanted = ', '.join(f'`{written}` as `{corrected_member}`'
                                   for written, corrected_member in rewrites)
                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: {subject} writes the element type with `of` '
                    f'in its type field `{type_field}`. Write {wanted}.',
                    type='types',
                    subtype='container_of',
                )
                comparable = False

            # Rule 8: the shape group separates its dimensions by `, ` and
            # the type name by a single space.
            problems = [problem for problem in
                        (_shape_group_problem(member)
                         for member in members)
                        if problem is not None]
            if problems:
                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: {subject} has the type field '
                    f'`{type_field}`, in which {" ".join(problems)}',
                    type='types',
                    subtype='shape_group',
                )
                comparable = False

            # Rule 7: every member parses under the grammar. A field that
            # one of the spelling rules already explains is not reported
            # twice.
            if comparable:
                malformed = [member.strip() for member in members
                             if not _is_member(member)]
                if malformed:
                    listed = ', '.join(f'`{member}`' for member in malformed)
                    logger.warning(
                        f'[types] ({location}) '
                        f'{name}: {subject} has the type field '
                        f'`{type_field}`, in which {listed} is not a type. '
                        f'A member is a qualified name with an optional '
                        f'subscript, a `{{...}}` set of literals, or '
                        f'`None`, with an optional `(N, M)` shape prefix.',
                        type='types',
                        subtype='malformed',
                    )
                    comparable = False

            # Rule 9: a set of literals carries single quotes and the
            # spacing of the Python set display, which is what
            # `ast.unparse` writes.
            if comparable:
                set_rewrites = []
                for member in members:
                    written = member.strip()
                    try:
                        node = ast.parse(written, mode='eval').body
                    except SyntaxError:
                        continue
                    if not isinstance(node, ast.Set):
                        continue
                    corrected_member = ast.unparse(node)
                    if corrected_member != written:
                        set_rewrites.append((written, corrected_member))

                if set_rewrites:
                    wanted = ', '.join(
                        f'`{written}` as `{corrected_member}`'
                        for written, corrected_member in set_rewrites)
                    logger.warning(
                        f'[types] ({location}) '
                        f'{name}: {subject} has the type field '
                        f'`{type_field}`, in which a set of literals is '
                        f'not written with single quotes and `, ` between '
                        f'its members. Write {wanted}.',
                        type='types',
                        subtype='set_quotes',
                    )
                    comparable = False

            # A spelling rule that fired on the `Returns` entry suppresses
            # the return comparison, exactly as it does on the parameter
            # side.
            if section == 'Returns' and not comparable:
                return_comparable = False

            # Rules 5 and 6 compare the documented members against the
            # annotation, which only the parameter sections carry.
            if not comparable or section not in _PARAM_SECTIONS:
                continue

            documented = frozenset(_member_text(member)
                                   for member in members)
            for entry_name in entry_names:
                annotation = annotations.get(entry_name)
                if annotation is None:
                    continue
                annotated = frozenset(_annotation_members(annotation))
                if documented == annotated:
                    continue

                # Rule 6: a `Literal[...]` annotation that the docstring
                # does not spell as a set is reported on its own, in place
                # of the generic mismatch of rule 5.
                missing_sets = sorted(member for member in annotated
                                      if member.startswith('{')
                                      and member not in documented)
                if missing_sets:
                    listed = ', '.join(f'`{member}`'
                                       for member in missing_sets)
                    logger.warning(
                        f'[types] ({location}) '
                        f'{name}: parameter `{entry_name}` has the '
                        f'annotation `{ast.unparse(annotation)}` but its '
                        f'type field `{type_field}` does not document the '
                        f'set {listed}.',
                        type='types',
                        subtype='literal_not_set',
                    )
                    continue

                logger.warning(
                    f'[types] ({location}) '
                    f'{name}: parameter `{entry_name}` documents the type '
                    f'field `{type_field}` but its annotation is '
                    f'`{ast.unparse(annotation)}`.',
                    type='types',
                    subtype='annotation_mismatch',
                )

    if return_annotation is None:
        return

    # Rule 5 in the return position. A `Returns` section of several entries
    # documents the elements of a tuple one by one and carries no single
    # type field to compare the one annotation against, so it is left
    # alone.
    returns = docstring['Returns']
    if len(returns) > 1:
        return

    return_field = returns[0].type.strip() if returns else ''
    if not return_field:
        logger.warning(
            f'[types] ({location}) '
            f'{name}: carries the return annotation '
            f'`{ast.unparse(return_annotation)}` but documents no return '
            f'type field.',
            type='types',
            subtype='annotation_mismatch',
        )
        return

    if not return_comparable:
        return

    return_list = return_field
    if return_list.endswith(_OPTIONAL_SUFFIX):
        return_list = return_list[:-len(_OPTIONAL_SUFFIX)]
    documented = frozenset(
        _member_text(member)
        for member in _split_top_level(return_list, _MEMBER_SEP_RE)[0])

    if documented != frozenset(_annotation_members(return_annotation)):
        logger.warning(
            f'[types] ({location}) '
            f'{name}: documents the return type field `{return_field}` '
            f'but its return annotation is '
            f'`{ast.unparse(return_annotation)}`.',
            type='types',
            subtype='annotation_mismatch',
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
    app.add_config_value('types_ignore', [], 'env', types=[list])

    # Priority 200 runs the handler before numpydoc, whose default
    # priority is 500, so that it sees the raw numpy-format lines.
    app.connect('autodoc-process-docstring', _validate_types, priority=200)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
