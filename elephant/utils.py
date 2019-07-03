# -*- coding: utf-8 -*-
"""
Elephant-wide utility functions.

:copyright: Copyright 2015-2019 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""


import quantities as pq


def check_quantities(var, var_name):
    """
    Checks input `var` data for having the type of `pq.Quantity`.

    Parameters
    ----------
    var
        input argument
    var_name : str
        variable name

    Raises
    -------
    AssertionError
        If the input `var` is not a `pq.Quantity`
    """
    assert isinstance(var, pq.Quantity), \
        "{0} must be of type pq.Quantity".format(var_name)
