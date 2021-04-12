"""
This module implements functions to activate provenance capture when using
IPython (e.g., when running Jupyter Notebooks).

:copyright: Copyright 2014-2021 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

PROV_LINE = "buffalo.activate()\n"


def _add_provenance_to_cell(lines):
    if lines:
        if lines[0] != PROV_LINE:
            return [PROV_LINE] + lines
    return lines


def activate_ipython():
    """
    Activates provenance tracking within Elephant, when using IPython.
    """
    try:
        ip = get_ipython()
        ip.input_transformers_cleanup.append(_add_provenance_to_cell)
    except NameError:
        print("You are running outside IPython. 'buffalo.activate()' should"
              "be used.")
