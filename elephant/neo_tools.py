# -*- coding: utf-8 -*-
"""
Tools to manipulate Neo objects.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

from itertools import chain

from neo.core.container import unique_objs


def extract_neo_attrs(obj, parents=True, child_first=True,
                      skip_array=False, skip_none=False):
    """Given a neo object, return a dictionary of attributes and annotations.

    Parameters
    ----------

    obj : neo object
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.
    skip_array : bool, optional
                 If True (default False), skip attributes that store non-scalar
                 array values.
    skip_none : bool, optional
                If True (default False), skip annotations and attributes that
                have a value of `None`.

    Returns
    -------

    dict
        A dictionary where the keys are annotations or attribute names and
        the values are the corresponding annotation or attribute value.

    """
    attrs = obj.annotations.copy()
    for attr in obj._necessary_attrs + obj._recommended_attrs:
        if skip_array and len(attr) >= 3 and attr[2]:
            continue
        attr = attr[0]
        if attr == getattr(obj, '_quantity_attr', None):
            continue
        attrs[attr] = getattr(obj, attr, None)

    if skip_none:
        for attr, value in attrs.copy().items():
            if value is None:
                del attrs[attr]

    if not parents:
        return attrs

    for parent in getattr(obj, 'parents', []):
        if parent is None:
            continue
        newattr = extract_neo_attrs(parent, parents=True,
                                    child_first=child_first,
                                    skip_array=skip_array,
                                    skip_none=skip_none)
        if child_first:
            newattr.update(attrs)
            attrs = newattr
        else:
            attrs.update(newattr)

    return attrs


def _get_all_objs(container, classname):
    """Get all `neo` objects of a given type from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    neo objects of a particular class, as well as any neo object that can hold
    the object.
    Objects are searched recursively, so the objects can be nested (such as a
    list of blocks).

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo container
                The container for the neo objects.
    classname : str
                The name of the class, with proper capitalization
                (so `SpikeTrain`, not `Spiketrain` or `spiketrain`)

    Returns
    -------

    list
        A list of unique `neo` objects

    """
    if container.__class__.__name__ == classname:
        return [container]
    classholder = classname.lower() + 's'
    if hasattr(container, classholder):
        vals = getattr(container, classholder)
    elif hasattr(container, 'list_children_by_class'):
        vals = container.list_children_by_class(classname)
    elif hasattr(container, 'values') and not hasattr(container, 'ndim'):
        vals = container.values()
    elif hasattr(container, '__iter__') and not hasattr(container, 'ndim'):
        vals = container
    else:
        raise ValueError('Cannot handle object of type %s' % type(container))
    res = list(chain.from_iterable(_get_all_objs(obj, classname)
                                   for obj in vals))
    return unique_objs(res)


def get_all_spiketrains(container):
    """Get all `neo.Spiketrain` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    spiketrains, as well as any neo object that can hold spiketrains:
    `neo.Block`, `neo.ChannelIndex`, `neo.Unit`, and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------

    container : list, tuple, iterable, dict,
                neo Block, neo Segment, neo Unit, neo ChannelIndex
                The container for the spiketrains.

    Returns
    -------

    list
        A list of the unique `neo.SpikeTrain` objects in `container`.

    """
    return _get_all_objs(container, 'SpikeTrain')


def get_all_events(container):
    """Get all `neo.Event` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    events, as well as any neo object that can hold events:
    `neo.Block` and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo Block, neo Segment
                The container for the events.

    Returns
    -------

    list
        A list of the unique `neo.Event` objects in `container`.

    """
    return _get_all_objs(container, 'Event')


def get_all_epochs(container):
    """Get all `neo.Epoch` objects from a container.

    The objects can be any list, dict, or other iterable or mapping containing
    epochs, as well as any neo object that can hold epochs:
    `neo.Block` and `neo.Segment`.

    Containers are searched recursively, so the objects can be nested
    (such as a list of blocks).

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo Block, neo Segment
                The container for the epochs.

    Returns
    -------

    list
        A list of the unique `neo.Epoch` objects in `container`.

    """
    return _get_all_objs(container, 'Epoch')
