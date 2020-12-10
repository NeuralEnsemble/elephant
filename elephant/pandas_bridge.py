# -*- coding: utf-8 -*-
"""
Bridge to the pandas library.

.. autosummary::
    :toctree: _toctree/pandas_bridge

    spiketrain_to_dataframe
    event_to_dataframe
    epoch_to_dataframe
    multi_spiketrains_to_dataframe
    multi_events_to_dataframe
    multi_epochs_to_dataframe
    slice_spiketrain

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import warnings
import quantities as pq

from elephant.neo_tools import (extract_neo_attributes, get_all_epochs,
                                get_all_events, get_all_spiketrains)


warnings.simplefilter('once', DeprecationWarning)
warnings.warn("pandas_bridge module will be removed in Elephant v0.8.x",
              DeprecationWarning)


def _multiindex_from_dict(inds):
    """Given a dictionary, return a `pandas.MultiIndex`.

    Parameters
    ----------
    inds : dict
           A dictionary where the keys are annotations or attribute names and
           the values are the corresponding annotation or attribute value.

    Returns
    -------
    pandas MultiIndex
    """
    names, indexes = zip(*sorted(inds.items()))
    return pd.MultiIndex.from_tuples([indexes], names=names)


def _sort_inds(obj, axis=0):
    """Put the indexes and index levels of a pandas object in sorted order.

    Paramters
    ---------
    obj : pandas Series, DataFrame, Panel, or Panel4D
          The object whose indexes should be sorted.
    axis : int, list, optional, 'all'
           The axis whose indexes should be sorted.  Default is 0.
           Can also be a list of indexes, in which case all of those axes
           are sorted.  If 'all', sort all indexes.

    Returns
    -------
    pandas Series, DataFrame, Panel, or Panel4D
        A copy of the object with indexes sorted.
        Indexes are sorted in-place.
    """
    if axis == 'all':
        return _sort_inds(obj, axis=range(obj.ndim))

    if hasattr(axis, '__iter__'):
        for iax in axis:
            obj = _sort_inds(obj, iax)
        return obj

    obj = obj.reorder_levels(sorted(obj.axes[axis].names), axis=axis)
    return obj.sort_index(level=0, axis=axis, sort_remaining=True)


def _extract_neo_attrs_safe(obj, parents=True, child_first=True):
    """Given a neo object, return a dictionary of attributes and annotations.

    This is done in a manner that is safe for `pandas` indexes.

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

    Returns
    -------

    dict
        A dictionary where the keys are annotations or attribute names and
        the values are the corresponding annotation or attribute value.

    """
    res = extract_neo_attributes(obj, skip_array=True, skip_none=True,
                                 parents=parents, child_first=child_first)
    for key, value in res.items():
        res[key] = _convert_value_safe(value)
        key2 = _convert_value_safe(key)
        if key2 is not key:
            res[key2] = res.pop(key)

    return res


def _convert_value_safe(value):
    """Convert `neo` values to a value compatible with `pandas`.

    Some types and dtypes used with neo are not safe to use with pandas in some
    or all situations.

    `quantities.Quantity` don't follow the normal python rule that values
    with that are equal should have the same hash, making it fundamentally
    incompatible with `pandas`.

    On python 3, `pandas` coerces `S` dtypes to bytes, which are not always
    safe to use.

    Parameters
    ----------

    value : any
            Value to convert (if it has any known issues).

    Returns
    -------

    any
        `value` or a version of value with potential problems fixed.

    """
    if hasattr(value, 'dimensionality'):
        return (value.magnitude.tolist(), str(value.dimensionality))
    if hasattr(value, 'dtype') and value.dtype.kind == 'S':
        return value.astype('U').tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'decode') and not hasattr(value, 'encode'):
        return value.decode('UTF8')
    return value


def spiketrain_to_dataframe(spiketrain, parents=True, child_first=True):
    """Convert a `neo.SpikeTrain` to a `pandas.DataFrame`.

    The `pandas.DataFrame` object has a single column, with each element
    being the spike time converted to a `float` value in seconds.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations.  The `index`
    is the spike number.

    Parameters
    ----------

    spiketrain : neo SpikeTrain
                 The SpikeTrain to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the spike times from `spiketrain`.

    Notes
    -----

    The index name is `spike_number`.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    attrs = _extract_neo_attrs_safe(spiketrain,
                                    parents=parents, child_first=child_first)
    columns = _multiindex_from_dict(attrs)

    times = spiketrain.magnitude
    times = pq.Quantity(times, spiketrain.units).rescale('s').magnitude
    times = times[np.newaxis].T

    index = pd.Index(np.arange(len(spiketrain)), name='spike_number')

    pdobj = pd.DataFrame(times, index=index, columns=columns)
    return _sort_inds(pdobj, axis=1)


def event_to_dataframe(event, parents=True, child_first=True):
    """Convert a `neo.core.Event` to a `pandas.DataFrame`.

    The `pandas.DataFrame` object has a single column, with each element
    being the event label from the `event.label` attribute.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations.  The `index`
    is the time stamp from the `event.times` attribute.

    Parameters
    ----------

    event : neo Event
            The Event to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the labels from `event`.

    Notes
    -----

    If the length of event.times and event.labels are not the same,
    the longer will be truncated to the length of the shorter.

    The index name is `times`.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    attrs = _extract_neo_attrs_safe(event,
                                    parents=parents, child_first=child_first)
    columns = _multiindex_from_dict(attrs)

    times = event.times.rescale('s').magnitude
    labels = event.labels.astype('U')

    times = times[:len(labels)]
    labels = labels[:len(times)]

    index = pd.Index(times, name='times')

    pdobj = pd.DataFrame(labels[np.newaxis].T, index=index, columns=columns)
    return _sort_inds(pdobj, axis=1)


def epoch_to_dataframe(epoch, parents=True, child_first=True):
    """Convert a `neo.core.Epoch` to a `pandas.DataFrame`.

    The `pandas.DataFrame` object has a single column, with each element
    being the epoch label from the `epoch.label` attribute.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations.  The `index`
    is a `pandas.MultiIndex`, with the first index being the time stamp from
    the `epoch.times` attribute and the second being the duration from
    the `epoch.durations` attribute.

    Parameters
    ----------

    epoch : neo Epoch
            The Epoch to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the labels from `epoch`.

    Notes
    -----

    If the length of `epoch.times`, `epoch.duration`, and `epoch.labels` are
    not the same, the longer will be truncated to the length of the shortest.

    The index names for `epoch.times` and `epoch.durations` are `times` and
    `durations`, respectively.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    attrs = _extract_neo_attrs_safe(epoch,
                                    parents=parents, child_first=child_first)
    columns = _multiindex_from_dict(attrs)

    times = epoch.times.rescale('s').magnitude
    durs = epoch.durations.rescale('s').magnitude
    labels = epoch.labels.astype('U')

    minlen = min([len(durs), len(times), len(labels)])
    index = pd.MultiIndex.from_arrays([times[:minlen], durs[:minlen]],
                                      names=['times', 'durations'])

    pdobj = pd.DataFrame(labels[:minlen][np.newaxis].T,
                         index=index, columns=columns)
    return _sort_inds(pdobj, axis='all')


def _multi_objs_to_dataframe(container, conv_func, get_func,
                             parents=True, child_first=True):
    """Convert one or more of a given `neo` object to a `pandas.DataFrame`.

    The objects can be any list, dict, or other iterable or mapping containing
    the object, as well as any neo object that can hold the object.
    Objects are searched recursively, so the objects can be nested (such as a
    list of blocks).

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations of the respective
    object.

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo container object
                The container for the objects to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the converted objects.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    res = pd.concat([conv_func(obj, parents=parents, child_first=child_first)
                     for obj in get_func(container)], axis=1)
    return _sort_inds(res, axis=1)


def multi_spiketrains_to_dataframe(container,
                                   parents=True, child_first=True):
    """Convert one or more `neo.SpikeTrain` objects to a `pandas.DataFrame`.

    The objects can be any list, dict, or other iterable or mapping containing
    spiketrains, as well as any neo object that can hold spiketrains:
    `neo.Block`, `neo.ChannelIndex`, `neo.Unit`, and `neo.Segment`.
    Objects are searched recursively, so the objects can be nested (such as a
    list of blocks).

    The `pandas.DataFrame` object has one column for each spiketrain, with each
    element being the spike time converted to a `float` value in seconds.
    columns are padded to the same length with `NaN` values.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations of the respective
    spiketrain.  The `index` is the spike number.

    Parameters
    ----------

    container : list, tuple, iterable, dict,
                neo Block, neo Segment, neo Unit, neo ChannelIndex
                The container for the spiketrains to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the spike times from `container`.

    Notes
    -----

    The index name is `spike_number`.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    return _multi_objs_to_dataframe(container,
                                    spiketrain_to_dataframe,
                                    get_all_spiketrains,
                                    parents=parents, child_first=child_first)


def multi_events_to_dataframe(container, parents=True, child_first=True):
    """Convert one or more `neo.Event` objects to a `pandas.DataFrame`.

    The objects can be any list, dict, or other iterable or mapping containing
    events, as well as any neo object that can hold events:
    `neo.Block` and `neo.Segment`.  Objects are searched recursively, so the
    objects can be nested (such as a list of blocks).

    The `pandas.DataFrame` object has one column for each event, with each
    element being the event label. columns are padded to the same length with
    `NaN` values.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations of the respective
    event.  The `index` is the time stamp from the `event.times` attribute.

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo Block, neo Segment
                The container for the events to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the labels from `container`.

    Notes
    -----

    If the length of event.times and event.labels are not the same for any
    individual event, the longer will be truncated to the length of the
    shorter for that event.  Between events, lengths can differ.

    The index name is `times`.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    return _multi_objs_to_dataframe(container,
                                    event_to_dataframe, get_all_events,
                                    parents=parents, child_first=child_first)


def multi_epochs_to_dataframe(container, parents=True, child_first=True):
    """Convert one or more `neo.Epoch` objects to a `pandas.DataFrame`.

    The objects can be any list, dict, or other iterable or mapping containing
    epochs, as well as any neo object that can hold epochs:
    `neo.Block` and `neo.Segment`.  Objects are searched recursively, so the
    objects can be nested (such as a list of blocks).

    The `pandas.DataFrame` object has one column for each epoch, with each
    element being the epoch label. columns are padded to the same length with
    `NaN` values.

    The column heading is a `pandas.MultiIndex` with one index
    for each of the scalar attributes and annotations of the respective
    epoch.  The `index` is a `pandas.MultiIndex`, with the first index being
    the time stamp from the `epoch.times` attribute and the second being the
    duration from the `epoch.durations` attribute.

    Parameters
    ----------

    container : list, tuple, iterable, dict, neo Block, neo Segment
                The container for the epochs to convert.
    parents : bool, optional
              Also include attributes and annotations from parent neo
              objects (if any).
    child_first : bool, optional
                  If True (default True), values of child attributes are used
                  over parent attributes in the event of a name conflict.
                  If False, parent attributes are used.
                  This parameter does nothing if `parents` is False.

    Returns
    -------

    pandas DataFrame
        A DataFrame containing the labels from `container`.

    Notes
    -----

    If the length of `epoch.times`, `epoch.duration`, and `epoch.labels` are
    not the same for any individual epoch, the longer will be truncated to the
    length of the shorter for that epoch.  Between epochs, lengths can differ.

    The index level names for `epoch.times` and `epoch.durations` are
    `times` and `durations`, respectively.

    Attributes that contain non-scalar values are skipped.  So are
    annotations or attributes containing a value of `None`.

    `quantity.Quantities` types are incompatible with `pandas`, so attributes
    and annotations of that type are converted to a tuple where the first
    element is the scalar value and the second is the string representation of
    the units.

    """
    return _multi_objs_to_dataframe(container,
                                    epoch_to_dataframe, get_all_epochs,
                                    parents=parents, child_first=child_first)


def slice_spiketrain(pdobj, t_start=None, t_stop=None):
    """Slice a `pandas.DataFrame`, changing indices appropriately.

    Values outside the sliced range are converted to `NaN` values.

    Slicing happens over columns.

    This sets the `t_start` and `t_stop` column indexes to be the new values.
    Otherwise it is the same as setting values outside the range to `NaN`.

    Parameters
    ----------
    pdobj : pandas DataFrame
            The DataFrame to slice.
    t_start : float, optional.
              If specified, the returned DataFrame values less than this set
              to `NaN`.
              Default is `None` (do not use this argument).
    t_stop : float, optional.
             If specified, the returned DataFrame values greater than this set
             to `NaN`.
             Default is `None` (do not use this argument).

    Returns
    -------

    pdobj : scalar, pandas Series, DataFrame, or Panel
            The returned data type is the same as the type of `pdobj`

    Notes
    -----

    The order of the index and/or column levels of the returned object may
    differ  from the order of the original.

    If `t_start` or `t_stop` is specified, all columns indexes will be changed
    to  the respective values, including those already within the new range.
    If `t_start` or `t_stop` is not specified, those column indexes will not
    be changed.

    Returns a copy, even if `t_start` and `t_stop` are both `None`.

    """
    if t_start is None and t_stop is None:
        return pdobj.copy()

    if t_stop is not None:
        pdobj[pdobj > t_stop] = np.nan

        pdobj = pdobj.T.reset_index(level='t_stop')
        pdobj['t_stop'] = t_stop
        pdobj = pdobj.set_index('t_stop', append=True).T
        pdobj = _sort_inds(pdobj, axis=1)

    if t_start is not None:
        pdobj[pdobj < t_start] = np.nan

        pdobj = pdobj.T.reset_index(level='t_start')
        pdobj['t_start'] = t_start
        pdobj = pdobj.set_index('t_start', append=True).T
        pdobj = _sort_inds(pdobj, axis=1)

    return pdobj
