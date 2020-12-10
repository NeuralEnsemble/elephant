# -*- coding: utf-8 -*-
"""
Tools to manipulate Neo objects.

.. autosummary::
    :toctree: _toctree/neo_tools

    extract_neo_attributes
    get_all_spiketrains
    get_all_events
    get_all_epochs

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals
import warnings

from itertools import chain

from neo.core.container import unique_objs
from elephant.utils import deprecated_alias

__all__ = [
    "extract_neo_attributes",
    "get_all_spiketrains",
    "get_all_events",
    "get_all_epochs"
]


@deprecated_alias(obj='neo_object')
def extract_neo_attributes(neo_object, parents=True, child_first=True,
                           skip_array=False, skip_none=False):
    """
    Given a Neo object, return a dictionary of attributes and annotations.

    Parameters
    ----------
    neo_object : neo.BaseNeo
        Object to get attributes and annotations.
    parents : bool, optional
        If True, also include attributes and annotations from parent Neo
        objects (if any).
        Default: True
    child_first : bool, optional
        If True, values of child attributes are used over parent attributes in
        the event of a name conflict.
        If False, parent attributes are used.
        This parameter does nothing if `parents` is False.
        Default: True
    skip_array : bool, optional
        If True, skip attributes that store non-scalar array values.
        Default: False
    skip_none : bool, optional
        If True, skip annotations and attributes that have a value of None.
        Default: False

    Returns
    -------
    dict
        A dictionary where the keys are annotations or attribute names and
        the values are the corresponding annotation or attribute value.

    """
    attrs = neo_object.annotations.copy()
    if not skip_array and hasattr(neo_object, "array_annotations"):
        # Exclude labels and durations, and any other fields that should not
        # be a part of array_annotation.
        required_keys = set(neo_object.array_annotations).difference(
            dir(neo_object))
        for a in required_keys:
            if "array_annotations" not in attrs:
                attrs["array_annotations"] = {}
            attrs["array_annotations"][a] = \
                neo_object.array_annotations[a].copy()
    for attr in neo_object._necessary_attrs + neo_object._recommended_attrs:
        if skip_array and len(attr) >= 3 and attr[2]:
            continue
        attr = attr[0]
        if attr == getattr(neo_object, '_quantity_attr', None):
            continue
        attrs[attr] = getattr(neo_object, attr, None)

    if skip_none:
        for attr, value in attrs.copy().items():
            if value is None:
                del attrs[attr]

    if not parents:
        return attrs

    for parent in getattr(neo_object, 'parents', []):
        if parent is None:
            continue
        newattr = extract_neo_attributes(parent, parents=True,
                                         child_first=child_first,
                                         skip_array=skip_array,
                                         skip_none=skip_none)
        if child_first:
            newattr.update(attrs)
            attrs = newattr
        else:
            attrs.update(newattr)

    return attrs


def extract_neo_attrs(*args, **kwargs):
    warnings.warn("'extract_neo_attrs' function is deprecated; "
                  "use 'extract_neo_attributes'", DeprecationWarning)
    return extract_neo_attributes(*args, **kwargs)


def _get_all_objs(container, class_name):
    """
    Get all Neo objects of a given type from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    Neo objects of a particular class, as well as any Neo object that can hold
    the object.
    Objects are searched recursively, so the objects can be nested (such as a
    list of blocks).

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo.Container
                The container for the Neo objects.
    class_name : str
                The name of the class, with proper capitalization
                (i.e., 'SpikeTrain', not 'Spiketrain' or 'spiketrain').

    Returns
    -------
    list
        A list of unique Neo objects.

    Raises
    ------
    ValueError
        If can not handle containers of the type passed in `container`.

    """
    if container.__class__.__name__ == class_name:
        return [container]
    classholder = class_name.lower() + 's'
    if hasattr(container, classholder):
        vals = getattr(container, classholder)
    elif hasattr(container, 'list_children_by_class'):
        vals = container.list_children_by_class(class_name)
    elif hasattr(container, 'values') and not hasattr(container, 'ndim'):
        vals = container.values()
    elif hasattr(container, '__iter__') and not hasattr(container, 'ndim'):
        vals = container
    else:
        raise ValueError('Cannot handle object of type %s' % type(container))
    res = list(chain.from_iterable(_get_all_objs(obj, class_name)
                                   for obj in vals))
    return unique_objs(res)


def get_all_spiketrains(container):
    """
    Get all `neo.Spiketrain` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    spiketrains, as well as any Neo object that can hold spiketrains:
    `neo.Block`, `neo.ChannelIndex`, `neo.Unit`, and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo.Block, neo.Segment, neo.Unit,
        neo.ChannelIndex
        The container for the spiketrains.

    Returns
    -------
    list
        A list of the unique `neo.SpikeTrain` objects in `container`.

    """
    return _get_all_objs(container, 'SpikeTrain')


def get_all_events(container):
    """
    Get all `neo.Event` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    events, as well as any neo object that can hold events:
    `neo.Block` and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo.Block, neo.Segment
                The container for the events.

    Returns
    -------
    list
        A list of the unique `neo.Event` objects in `container`.

    """
    return _get_all_objs(container, 'Event')


def get_all_epochs(container):
    """
    Get all `neo.Epoch` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    epochs, as well as any neo object that can hold epochs:
    `neo.Block` and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo.Block, neo.Segment
                The container for the epochs.

    Returns
    -------
    list
        A list of the unique `neo.Epoch` objects in `container`.

    """
    return _get_all_objs(container, 'Epoch')
