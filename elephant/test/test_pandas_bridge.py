# -*- coding: utf-8 -*-
"""
Unit tests for the pandas bridge module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import unittest
import warnings
from distutils.version import StrictVersion
from itertools import chain

import numpy as np
import quantities as pq
from neo.test.generate_datasets import fake_neo
from numpy.testing import assert_array_equal

try:
    import pandas as pd
    from pandas.util.testing import assert_frame_equal, assert_index_equal
except ImportError:
    HAVE_PANDAS = False
    pandas_version = StrictVersion('0.0.0')
else:
    import elephant.pandas_bridge as ep
    HAVE_PANDAS = True
    pandas_version = StrictVersion(pd.__version__)

if HAVE_PANDAS:
    # Currying, otherwise the unittest will break with pandas>=0.16.0
    # parameter check_names is introduced in a newer versions than 0.14.0
    # this test is written for pandas 0.14.0
    def assert_index_equal(left, right):
        try:
            # pandas>=0.16.0
            return pd.util.testing.assert_index_equal(left, right,
                                                      check_names=False)
        except TypeError:
            # pandas older version
            return pd.util.testing.assert_index_equal(left, right)


@unittest.skipUnless(pandas_version >= '0.24.0', 'requires pandas v0.24.0')
class MultiindexFromDictTestCase(unittest.TestCase):
    def test__multiindex_from_dict(self):
        inds = {'test1': 6.5,
                'test2': 5,
                'test3': 'test'}
        targ = pd.MultiIndex(levels=[[6.5], [5], ['test']],
                             codes=[[0], [0], [0]],
                             names=['test1', 'test2', 'test3'])
        res0 = ep._multiindex_from_dict(inds)
        self.assertEqual(targ.levels, res0.levels)
        self.assertEqual(targ.names, res0.names)
        self.assertEqual(targ.codes, res0.codes)


def _convert_levels(levels):
    """Convert a list of levels to the format pandas returns for a MultiIndex.

    Parameters
    ----------

    levels : list
             The list of levels to convert.

    Returns
    -------

    list
        The the level in `list` converted to values like what pandas will give.

    """
    levels = list(levels)
    for i, level in enumerate(levels):
        if hasattr(level, 'lower'):
            try:
                level = unicode(level)
            except NameError:
                pass
        elif hasattr(level, 'date'):
            levels[i] = pd.DatetimeIndex(data=[level])
            continue
        elif level is None:
            levels[i] = pd.Index([])
            continue

        # pd.Index around pd.Index to convert to Index structure if MultiIndex
        levels[i] = pd.Index(pd.Index([level]))
    return levels


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class ConvertValueSafeTestCase(unittest.TestCase):
    def test__convert_value_safe__float(self):
        targ = 5.5
        value = targ

        res = ep._convert_value_safe(value)

        self.assertIs(res, targ)

    def test__convert_value_safe__str(self):
        targ = 'test'
        value = targ

        res = ep._convert_value_safe(value)

        self.assertIs(res, targ)

    def test__convert_value_safe__bytes(self):
        targ = 'test'
        value = b'test'

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)

    def test__convert_value_safe__numpy_int_scalar(self):
        targ = 5
        value = np.array(5)

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)
        self.assertFalse(hasattr(res, 'dtype'))

    def test__convert_value_safe__numpy_float_scalar(self):
        targ = 5.
        value = np.array(5.)

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)
        self.assertFalse(hasattr(res, 'dtype'))

    def test__convert_value_safe__numpy_unicode_scalar(self):
        targ = u'test'
        value = np.array('test', dtype='U')

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)
        self.assertFalse(hasattr(res, 'dtype'))

    def test__convert_value_safe__numpy_str_scalar(self):
        targ = u'test'
        value = np.array('test', dtype='S')

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)
        self.assertFalse(hasattr(res, 'dtype'))

    def test__convert_value_safe__quantity_scalar(self):
        targ = (10., 'ms')
        value = 10. * pq.ms

        res = ep._convert_value_safe(value)

        self.assertEqual(res, targ)
        self.assertFalse(hasattr(res[0], 'dtype'))
        self.assertFalse(hasattr(res[0], 'units'))


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class SpiketrainToDataframeTestCase(unittest.TestCase):
    def test__spiketrain_to_dataframe__parents_empty(self):
        obj = fake_neo('SpikeTrain', seed=0)

        res0 = ep.spiketrain_to_dataframe(obj)
        res1 = ep.spiketrain_to_dataframe(obj, child_first=True)
        res2 = ep.spiketrain_to_dataframe(obj, child_first=False)
        res3 = ep.spiketrain_to_dataframe(obj, parents=True)
        res4 = ep.spiketrain_to_dataframe(obj, parents=True,
                                          child_first=True)
        res5 = ep.spiketrain_to_dataframe(obj, parents=True,
                                          child_first=False)
        res6 = ep.spiketrain_to_dataframe(obj, parents=False)
        res7 = ep.spiketrain_to_dataframe(obj, parents=False, child_first=True)
        res8 = ep.spiketrain_to_dataframe(obj, parents=False,
                                          child_first=False)

        targvalues = pq.Quantity(obj.magnitude, units=obj.units)
        targvalues = targvalues.rescale('s').magnitude[np.newaxis].T
        targindex = np.arange(len(targvalues))

        attrs = ep._extract_neo_attrs_safe(obj, parents=True, child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))
        self.assertEqual(1, len(res4.columns))
        self.assertEqual(1, len(res5.columns))
        self.assertEqual(1, len(res6.columns))
        self.assertEqual(1, len(res7.columns))
        self.assertEqual(1, len(res8.columns))

        self.assertEqual(len(obj), len(res0.index))
        self.assertEqual(len(obj), len(res1.index))
        self.assertEqual(len(obj), len(res2.index))
        self.assertEqual(len(obj), len(res3.index))
        self.assertEqual(len(obj), len(res4.index))
        self.assertEqual(len(obj), len(res5.index))
        self.assertEqual(len(obj), len(res6.index))
        self.assertEqual(len(obj), len(res7.index))
        self.assertEqual(len(obj), len(res8.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)
        assert_array_equal(targvalues, res4.values)
        assert_array_equal(targvalues, res5.values)
        assert_array_equal(targvalues, res6.values)
        assert_array_equal(targvalues, res7.values)
        assert_array_equal(targvalues, res8.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)
        assert_array_equal(targindex, res3.index)
        assert_array_equal(targindex, res4.index)
        assert_array_equal(targindex, res5.index)
        assert_array_equal(targindex, res6.index)
        assert_array_equal(targindex, res7.index)
        assert_array_equal(targindex, res8.index)

        self.assertEqual(['spike_number'], res0.index.names)
        self.assertEqual(['spike_number'], res1.index.names)
        self.assertEqual(['spike_number'], res2.index.names)
        self.assertEqual(['spike_number'], res3.index.names)
        self.assertEqual(['spike_number'], res4.index.names)
        self.assertEqual(['spike_number'], res5.index.names)
        self.assertEqual(['spike_number'], res6.index.names)
        self.assertEqual(['spike_number'], res7.index.names)
        self.assertEqual(['spike_number'], res8.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)
        self.assertEqual(keys, res4.columns.names)
        self.assertEqual(keys, res5.columns.names)
        self.assertEqual(keys, res6.columns.names)
        self.assertEqual(keys, res7.columns.names)
        self.assertEqual(keys, res8.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res4.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res5.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res6.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res7.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res8.columns.levels):
            assert_index_equal(value, level)

    def test__spiketrain_to_dataframe__noparents(self):
        blk = fake_neo('Block', seed=0)
        obj = blk.list_children_by_class('SpikeTrain')[0]

        res0 = ep.spiketrain_to_dataframe(obj, parents=False)
        res1 = ep.spiketrain_to_dataframe(obj, parents=False,
                                          child_first=True)
        res2 = ep.spiketrain_to_dataframe(obj, parents=False,
                                          child_first=False)

        targvalues = pq.Quantity(obj.magnitude, units=obj.units)
        targvalues = targvalues.rescale('s').magnitude[np.newaxis].T
        targindex = np.arange(len(targvalues))

        attrs = ep._extract_neo_attrs_safe(obj, parents=False,
                                           child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))

        self.assertEqual(len(obj), len(res0.index))
        self.assertEqual(len(obj), len(res1.index))
        self.assertEqual(len(obj), len(res2.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)

        self.assertEqual(['spike_number'], res0.index.names)
        self.assertEqual(['spike_number'], res1.index.names)
        self.assertEqual(['spike_number'], res2.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)

    def test__spiketrain_to_dataframe__parents_childfirst(self):
        blk = fake_neo('Block', seed=0)
        obj = blk.list_children_by_class('SpikeTrain')[0]
        res0 = ep.spiketrain_to_dataframe(obj)
        res1 = ep.spiketrain_to_dataframe(obj, child_first=True)
        res2 = ep.spiketrain_to_dataframe(obj, parents=True)
        res3 = ep.spiketrain_to_dataframe(obj, parents=True, child_first=True)

        targvalues = pq.Quantity(obj.magnitude, units=obj.units)
        targvalues = targvalues.rescale('s').magnitude[np.newaxis].T
        targindex = np.arange(len(targvalues))

        attrs = ep._extract_neo_attrs_safe(obj, parents=True, child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))

        self.assertEqual(len(obj), len(res0.index))
        self.assertEqual(len(obj), len(res1.index))
        self.assertEqual(len(obj), len(res2.index))
        self.assertEqual(len(obj), len(res3.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)
        assert_array_equal(targindex, res3.index)

        self.assertEqual(['spike_number'], res0.index.names)
        self.assertEqual(['spike_number'], res1.index.names)
        self.assertEqual(['spike_number'], res2.index.names)
        self.assertEqual(['spike_number'], res3.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)

    def test__spiketrain_to_dataframe__parents_parentfirst(self):
        blk = fake_neo('Block', seed=0)
        obj = blk.list_children_by_class('SpikeTrain')[0]
        res0 = ep.spiketrain_to_dataframe(obj, child_first=False)
        res1 = ep.spiketrain_to_dataframe(obj, parents=True, child_first=False)

        targvalues = pq.Quantity(obj.magnitude, units=obj.units)
        targvalues = targvalues.rescale('s').magnitude[np.newaxis].T
        targindex = np.arange(len(targvalues))

        attrs = ep._extract_neo_attrs_safe(obj, parents=True,
                                           child_first=False)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))

        self.assertEqual(len(obj), len(res0.index))
        self.assertEqual(len(obj), len(res1.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)

        self.assertEqual(['spike_number'], res0.index.names)
        self.assertEqual(['spike_number'], res1.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class EventToDataframeTestCase(unittest.TestCase):
    def test__event_to_dataframe__parents_empty(self):
        obj = fake_neo('Event', seed=42)

        res0 = ep.event_to_dataframe(obj)
        res1 = ep.event_to_dataframe(obj, child_first=True)
        res2 = ep.event_to_dataframe(obj, child_first=False)
        res3 = ep.event_to_dataframe(obj, parents=True)
        res4 = ep.event_to_dataframe(obj, parents=True, child_first=True)
        res5 = ep.event_to_dataframe(obj, parents=True, child_first=False)
        res6 = ep.event_to_dataframe(obj, parents=False)
        res7 = ep.event_to_dataframe(obj, parents=False, child_first=True)
        res8 = ep.event_to_dataframe(obj, parents=False, child_first=False)

        targvalues = obj.labels[:len(obj.times)][np.newaxis].T.astype('U')
        targindex = obj.times[:len(obj.labels)].rescale('s').magnitude

        attrs = ep._extract_neo_attrs_safe(obj, parents=True, child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))
        self.assertEqual(1, len(res4.columns))
        self.assertEqual(1, len(res5.columns))
        self.assertEqual(1, len(res6.columns))
        self.assertEqual(1, len(res7.columns))
        self.assertEqual(1, len(res8.columns))

        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res2.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res3.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res4.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res5.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res6.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res7.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res8.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)
        assert_array_equal(targvalues, res4.values)
        assert_array_equal(targvalues, res5.values)
        assert_array_equal(targvalues, res6.values)
        assert_array_equal(targvalues, res7.values)
        assert_array_equal(targvalues, res8.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)
        assert_array_equal(targindex, res3.index)
        assert_array_equal(targindex, res4.index)
        assert_array_equal(targindex, res5.index)
        assert_array_equal(targindex, res6.index)
        assert_array_equal(targindex, res7.index)
        assert_array_equal(targindex, res8.index)

        self.assertEqual(['times'], res0.index.names)
        self.assertEqual(['times'], res1.index.names)
        self.assertEqual(['times'], res2.index.names)
        self.assertEqual(['times'], res3.index.names)
        self.assertEqual(['times'], res4.index.names)
        self.assertEqual(['times'], res5.index.names)
        self.assertEqual(['times'], res6.index.names)
        self.assertEqual(['times'], res7.index.names)
        self.assertEqual(['times'], res8.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)
        self.assertEqual(keys, res4.columns.names)
        self.assertEqual(keys, res5.columns.names)
        self.assertEqual(keys, res6.columns.names)
        self.assertEqual(keys, res7.columns.names)
        self.assertEqual(keys, res8.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res4.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res5.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res6.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res7.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res8.columns.levels):
            assert_index_equal(value, level)

    def test__event_to_dataframe__noparents(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Event')[0]

        res0 = ep.event_to_dataframe(obj, parents=False)
        res1 = ep.event_to_dataframe(obj, parents=False, child_first=False)
        res2 = ep.event_to_dataframe(obj, parents=False, child_first=True)

        targvalues = obj.labels[:len(obj.times)][np.newaxis].T.astype('U')
        targindex = obj.times[:len(obj.labels)].rescale('s').magnitude

        attrs = ep._extract_neo_attrs_safe(obj, parents=False,
                                           child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))

        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res2.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)

        self.assertEqual(['times'], res0.index.names)
        self.assertEqual(['times'], res1.index.names)
        self.assertEqual(['times'], res2.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)

    def test__event_to_dataframe__parents_childfirst(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Event')[0]

        res0 = ep.event_to_dataframe(obj)
        res1 = ep.event_to_dataframe(obj, child_first=True)
        res2 = ep.event_to_dataframe(obj, parents=True)
        res3 = ep.event_to_dataframe(obj, parents=True, child_first=True)

        targvalues = obj.labels[:len(obj.times)][np.newaxis].T.astype('U')
        targindex = obj.times[:len(obj.labels)].rescale('s').magnitude

        attrs = ep._extract_neo_attrs_safe(obj, parents=True, child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))

        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res2.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res3.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)
        assert_array_equal(targindex, res2.index)
        assert_array_equal(targindex, res3.index)

        self.assertEqual(['times'], res0.index.names)
        self.assertEqual(['times'], res1.index.names)
        self.assertEqual(['times'], res2.index.names)
        self.assertEqual(['times'], res3.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)

    def test__event_to_dataframe__parents_parentfirst(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Event')[0]
        res0 = ep.event_to_dataframe(obj, child_first=False)
        res1 = ep.event_to_dataframe(obj, parents=True, child_first=False)

        targvalues = obj.labels[:len(obj.times)][np.newaxis].T.astype('U')
        targindex = obj.times[:len(obj.labels)].rescale('s').magnitude

        attrs = ep._extract_neo_attrs_safe(obj, parents=True,
                                           child_first=False)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))

        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.labels)),
                         len(res1.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)

        assert_array_equal(targindex, res0.index)
        assert_array_equal(targindex, res1.index)

        self.assertEqual(['times'], res0.index.names)
        self.assertEqual(['times'], res1.index.names)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class EpochToDataframeTestCase(unittest.TestCase):
    def test__epoch_to_dataframe__parents_empty(self):
        obj = fake_neo('Epoch', seed=42)

        res0 = ep.epoch_to_dataframe(obj)
        res1 = ep.epoch_to_dataframe(obj, child_first=True)
        res2 = ep.epoch_to_dataframe(obj, child_first=False)
        res3 = ep.epoch_to_dataframe(obj, parents=True)
        res4 = ep.epoch_to_dataframe(obj, parents=True, child_first=True)
        res5 = ep.epoch_to_dataframe(obj, parents=True, child_first=False)
        res6 = ep.epoch_to_dataframe(obj, parents=False)
        res7 = ep.epoch_to_dataframe(obj, parents=False, child_first=True)
        res8 = ep.epoch_to_dataframe(obj, parents=False, child_first=False)

        minlen = min([len(obj.times), len(obj.durations), len(obj.labels)])
        targvalues = obj.labels[:minlen][np.newaxis].T.astype('U')
        targindex = np.vstack([obj.durations[:minlen].rescale('s').magnitude,
                               obj.times[:minlen].rescale('s').magnitude])
        targvalues = targvalues[targindex.argsort()[0], :]
        targindex.sort()

        attrs = ep._extract_neo_attrs_safe(obj, parents=True,
                                           child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))
        self.assertEqual(1, len(res4.columns))
        self.assertEqual(1, len(res5.columns))
        self.assertEqual(1, len(res6.columns))
        self.assertEqual(1, len(res7.columns))
        self.assertEqual(1, len(res8.columns))

        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res2.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res3.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res4.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res5.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res6.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res7.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res8.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)
        assert_array_equal(targvalues, res4.values)
        assert_array_equal(targvalues, res5.values)
        assert_array_equal(targvalues, res6.values)
        assert_array_equal(targvalues, res7.values)
        assert_array_equal(targvalues, res8.values)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)
        self.assertEqual(keys, res4.columns.names)
        self.assertEqual(keys, res5.columns.names)
        self.assertEqual(keys, res6.columns.names)
        self.assertEqual(keys, res7.columns.names)
        self.assertEqual(keys, res8.columns.names)

        self.assertEqual([u'durations', u'times'], res0.index.names)
        self.assertEqual([u'durations', u'times'], res1.index.names)
        self.assertEqual([u'durations', u'times'], res2.index.names)
        self.assertEqual([u'durations', u'times'], res3.index.names)
        self.assertEqual([u'durations', u'times'], res4.index.names)
        self.assertEqual([u'durations', u'times'], res5.index.names)
        self.assertEqual([u'durations', u'times'], res6.index.names)
        self.assertEqual([u'durations', u'times'], res7.index.names)
        self.assertEqual([u'durations', u'times'], res8.index.names)

        self.assertEqual(2, len(res0.index.levels))
        self.assertEqual(2, len(res1.index.levels))
        self.assertEqual(2, len(res2.index.levels))
        self.assertEqual(2, len(res3.index.levels))
        self.assertEqual(2, len(res4.index.levels))
        self.assertEqual(2, len(res5.index.levels))
        self.assertEqual(2, len(res6.index.levels))
        self.assertEqual(2, len(res7.index.levels))
        self.assertEqual(2, len(res8.index.levels))

        assert_array_equal(targindex, res0.index.levels)
        assert_array_equal(targindex, res1.index.levels)
        assert_array_equal(targindex, res2.index.levels)
        assert_array_equal(targindex, res3.index.levels)
        assert_array_equal(targindex, res4.index.levels)
        assert_array_equal(targindex, res5.index.levels)
        assert_array_equal(targindex, res6.index.levels)
        assert_array_equal(targindex, res7.index.levels)
        assert_array_equal(targindex, res8.index.levels)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res4.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res5.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res6.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res7.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res8.columns.levels):
            assert_index_equal(value, level)

    def test__epoch_to_dataframe__noparents(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Epoch')[0]

        res0 = ep.epoch_to_dataframe(obj, parents=False)
        res1 = ep.epoch_to_dataframe(obj, parents=False, child_first=True)
        res2 = ep.epoch_to_dataframe(obj, parents=False, child_first=False)

        minlen = min([len(obj.times), len(obj.durations), len(obj.labels)])
        targvalues = obj.labels[:minlen][np.newaxis].T.astype('U')
        targindex = np.vstack([obj.durations[:minlen].rescale('s').magnitude,
                               obj.times[:minlen].rescale('s').magnitude])
        targvalues = targvalues[targindex.argsort()[0], :]
        targindex.sort()

        attrs = ep._extract_neo_attrs_safe(obj, parents=False,
                                           child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))

        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res2.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)

        self.assertEqual([u'durations', u'times'], res0.index.names)
        self.assertEqual([u'durations', u'times'], res1.index.names)
        self.assertEqual([u'durations', u'times'], res2.index.names)

        self.assertEqual(2, len(res0.index.levels))
        self.assertEqual(2, len(res1.index.levels))
        self.assertEqual(2, len(res2.index.levels))

        assert_array_equal(targindex, res0.index.levels)
        assert_array_equal(targindex, res1.index.levels)
        assert_array_equal(targindex, res2.index.levels)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)

    def test__epoch_to_dataframe__parents_childfirst(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Epoch')[0]

        res0 = ep.epoch_to_dataframe(obj)
        res1 = ep.epoch_to_dataframe(obj, child_first=True)
        res2 = ep.epoch_to_dataframe(obj, parents=True)
        res3 = ep.epoch_to_dataframe(obj, parents=True, child_first=True)

        minlen = min([len(obj.times), len(obj.durations), len(obj.labels)])
        targvalues = obj.labels[:minlen][np.newaxis].T.astype('U')
        targindex = np.vstack([obj.durations[:minlen].rescale('s').magnitude,
                               obj.times[:minlen].rescale('s').magnitude])
        targvalues = targvalues[targindex.argsort()[0], :]
        targindex.sort()

        attrs = ep._extract_neo_attrs_safe(obj, parents=True, child_first=True)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))
        self.assertEqual(1, len(res2.columns))
        self.assertEqual(1, len(res3.columns))

        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res1.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res2.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res3.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)
        assert_array_equal(targvalues, res2.values)
        assert_array_equal(targvalues, res3.values)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)
        self.assertEqual(keys, res2.columns.names)
        self.assertEqual(keys, res3.columns.names)

        self.assertEqual([u'durations', u'times'], res0.index.names)
        self.assertEqual([u'durations', u'times'], res1.index.names)
        self.assertEqual([u'durations', u'times'], res2.index.names)
        self.assertEqual([u'durations', u'times'], res3.index.names)

        self.assertEqual(2, len(res0.index.levels))
        self.assertEqual(2, len(res1.index.levels))
        self.assertEqual(2, len(res2.index.levels))
        self.assertEqual(2, len(res3.index.levels))

        assert_array_equal(targindex, res0.index.levels)
        assert_array_equal(targindex, res1.index.levels)
        assert_array_equal(targindex, res2.index.levels)
        assert_array_equal(targindex, res3.index.levels)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res2.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res3.columns.levels):
            assert_index_equal(value, level)

    def test__epoch_to_dataframe__parents_parentfirst(self):
        blk = fake_neo('Block', seed=42)
        obj = blk.list_children_by_class('Epoch')[0]

        res0 = ep.epoch_to_dataframe(obj, child_first=False)
        res1 = ep.epoch_to_dataframe(obj, parents=True, child_first=False)

        minlen = min([len(obj.times), len(obj.durations), len(obj.labels)])
        targvalues = obj.labels[:minlen][np.newaxis].T.astype('U')
        targindex = np.vstack([obj.durations[:minlen].rescale('s').magnitude,
                               obj.times[:minlen].rescale('s').magnitude])
        targvalues = targvalues[targindex.argsort()[0], :]
        targindex.sort()

        attrs = ep._extract_neo_attrs_safe(obj, parents=True,
                                           child_first=False)
        keys, values = zip(*sorted(attrs.items()))
        values = _convert_levels(values)

        self.assertEqual(1, len(res0.columns))
        self.assertEqual(1, len(res1.columns))

        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res0.index))
        self.assertEqual(min(len(obj.times), len(obj.durations),
                             len(obj.labels)),
                         len(res1.index))

        assert_array_equal(targvalues, res0.values)
        assert_array_equal(targvalues, res1.values)

        self.assertEqual(keys, res0.columns.names)
        self.assertEqual(keys, res1.columns.names)

        self.assertEqual([u'durations', u'times'], res0.index.names)
        self.assertEqual([u'durations', u'times'], res1.index.names)

        self.assertEqual(2, len(res0.index.levels))
        self.assertEqual(2, len(res1.index.levels))

        assert_array_equal(targindex, res0.index.levels)
        assert_array_equal(targindex, res1.index.levels)

        for value, level in zip(values, res0.columns.levels):
            assert_index_equal(value, level)
        for value, level in zip(values, res1.columns.levels):
            assert_index_equal(value, level)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class MultiSpiketrainsToDataframeTestCase(unittest.TestCase):
    def setUp(self):
        if hasattr(self, 'assertItemsEqual'):
            self.assertCountEqual = self.assertItemsEqual

    def test__multi_spiketrains_to_dataframe__single(self):
        obj = fake_neo('SpikeTrain', seed=0, n=5)

        res0 = ep.multi_spiketrains_to_dataframe(obj)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=False)
        res2 = ep.multi_spiketrains_to_dataframe(obj, parents=True)
        res3 = ep.multi_spiketrains_to_dataframe(obj, child_first=True)
        res4 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=True)
        res5 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=True)
        res6 = ep.multi_spiketrains_to_dataframe(obj, child_first=False)
        res7 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=False)
        res8 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=False)

        targ = ep.spiketrain_to_dataframe(obj)

        keys = ep._extract_neo_attrs_safe(obj, parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = 1
        targlen = len(obj)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))
        self.assertEqual(targwidth, len(res4.columns))
        self.assertEqual(targwidth, len(res5.columns))
        self.assertEqual(targwidth, len(res6.columns))
        self.assertEqual(targwidth, len(res7.columns))
        self.assertEqual(targwidth, len(res8.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))
        self.assertEqual(targlen, len(res4.index))
        self.assertEqual(targlen, len(res5.index))
        self.assertEqual(targlen, len(res6.index))
        self.assertEqual(targlen, len(res7.index))
        self.assertEqual(targlen, len(res8.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)
        self.assertCountEqual(keys, res4.columns.names)
        self.assertCountEqual(keys, res5.columns.names)
        self.assertCountEqual(keys, res6.columns.names)
        self.assertCountEqual(keys, res7.columns.names)
        self.assertCountEqual(keys, res8.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)
        assert_array_equal(targ.values, res3.values)
        assert_array_equal(targ.values, res4.values)
        assert_array_equal(targ.values, res5.values)
        assert_array_equal(targ.values, res6.values)
        assert_array_equal(targ.values, res7.values)
        assert_array_equal(targ.values, res8.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)
        assert_frame_equal(targ, res4)
        assert_frame_equal(targ, res5)
        assert_frame_equal(targ, res6)
        assert_frame_equal(targ, res7)
        assert_frame_equal(targ, res8)

    def test__multi_spiketrains_to_dataframe__unit_default(self):
        obj = fake_neo('Unit', seed=0, n=5)

        res0 = ep.multi_spiketrains_to_dataframe(obj)

        objs = obj.spiketrains

        targ = [ep.spiketrain_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(targ.values, res0.values)

        assert_frame_equal(targ, res0)

    def test__multi_spiketrains_to_dataframe__segment_default(self):
        obj = fake_neo('Segment', seed=0, n=5)

        res0 = ep.multi_spiketrains_to_dataframe(obj)

        objs = obj.spiketrains

        targ = [ep.spiketrain_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)
        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(targ.values, res0.values)

        assert_frame_equal(targ, res0)

    def test__multi_spiketrains_to_dataframe__block_noparents(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_spiketrains_to_dataframe(obj, parents=False)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=True)
        res2 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=False)

        objs = obj.list_children_by_class('SpikeTrain')

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_spiketrains_to_dataframe__block_parents_childfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_spiketrains_to_dataframe(obj)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=True)
        res2 = ep.multi_spiketrains_to_dataframe(obj, child_first=True)
        res3 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=True)

        objs = obj.list_children_by_class('SpikeTrain')

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)
        assert_array_equal(targ.values, res3.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_spiketrains_to_dataframe__block_parents_parentfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_spiketrains_to_dataframe(obj, child_first=False)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=False)

        objs = obj.list_children_by_class('SpikeTrain')

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_spiketrains_to_dataframe__list_noparents(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_spiketrains_to_dataframe(obj, parents=False)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=True)
        res2 = ep.multi_spiketrains_to_dataframe(obj, parents=False,
                                                 child_first=False)

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_spiketrains_to_dataframe__list_parents_childfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_spiketrains_to_dataframe(obj)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=True)
        res2 = ep.multi_spiketrains_to_dataframe(obj, child_first=True)
        res3 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=True)

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)
        assert_array_equal(targ.values, res3.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_spiketrains_to_dataframe__list_parents_parentfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_spiketrains_to_dataframe(obj, child_first=False)
        res1 = ep.multi_spiketrains_to_dataframe(obj, parents=True,
                                                 child_first=False)

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.spiketrain_to_dataframe(iobj,
                                           parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_spiketrains_to_dataframe__tuple_default(self):
        obj = tuple(fake_neo('Block', seed=i, n=3) for i in range(3))

        res0 = ep.multi_spiketrains_to_dataframe(obj)

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.spiketrain_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(targ.values, res0.values)

        assert_frame_equal(targ, res0)

    def test__multi_spiketrains_to_dataframe__iter_default(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_spiketrains_to_dataframe(iter(obj))

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in obj)
        objs = list(chain.from_iterable(objs))
        targ = [ep.spiketrain_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(targ.values, res0.values)

        assert_frame_equal(targ, res0)

    def test__multi_spiketrains_to_dataframe__dict_default(self):
        obj = dict((i, fake_neo('Block', seed=i, n=3)) for i in range(3))

        res0 = ep.multi_spiketrains_to_dataframe(obj)

        objs = (iobj.list_children_by_class('SpikeTrain') for iobj in
                obj.values())
        objs = list(chain.from_iterable(objs))
        targ = [ep.spiketrain_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = max(len(iobj) for iobj in objs)

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(targ.values, res0.values)

        assert_frame_equal(targ, res0)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class MultiEventsToDataframeTestCase(unittest.TestCase):
    def setUp(self):
        if hasattr(self, 'assertItemsEqual'):
            self.assertCountEqual = self.assertItemsEqual

    def test__multi_events_to_dataframe__single(self):
        obj = fake_neo('Event', seed=0, n=5)

        res0 = ep.multi_events_to_dataframe(obj)
        res1 = ep.multi_events_to_dataframe(obj, parents=False)
        res2 = ep.multi_events_to_dataframe(obj, parents=True)
        res3 = ep.multi_events_to_dataframe(obj, child_first=True)
        res4 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=True)
        res5 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=True)
        res6 = ep.multi_events_to_dataframe(obj, child_first=False)
        res7 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=False)
        res8 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=False)

        targ = ep.event_to_dataframe(obj)

        keys = ep._extract_neo_attrs_safe(obj, parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = 1
        targlen = min(len(obj.times), len(obj.labels))

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))
        self.assertEqual(targwidth, len(res4.columns))
        self.assertEqual(targwidth, len(res5.columns))
        self.assertEqual(targwidth, len(res6.columns))
        self.assertEqual(targwidth, len(res7.columns))
        self.assertEqual(targwidth, len(res8.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))
        self.assertEqual(targlen, len(res4.index))
        self.assertEqual(targlen, len(res5.index))
        self.assertEqual(targlen, len(res6.index))
        self.assertEqual(targlen, len(res7.index))
        self.assertEqual(targlen, len(res8.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)
        self.assertCountEqual(keys, res4.columns.names)
        self.assertCountEqual(keys, res5.columns.names)
        self.assertCountEqual(keys, res6.columns.names)
        self.assertCountEqual(keys, res7.columns.names)
        self.assertCountEqual(keys, res8.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)
        assert_array_equal(targ.values, res3.values)
        assert_array_equal(targ.values, res4.values)
        assert_array_equal(targ.values, res5.values)
        assert_array_equal(targ.values, res6.values)
        assert_array_equal(targ.values, res7.values)
        assert_array_equal(targ.values, res8.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)
        assert_frame_equal(targ, res4)
        assert_frame_equal(targ, res5)
        assert_frame_equal(targ, res6)
        assert_frame_equal(targ, res7)
        assert_frame_equal(targ, res8)

    def test__multi_events_to_dataframe__segment_default(self):
        obj = fake_neo('Segment', seed=0, n=5)

        res0 = ep.multi_events_to_dataframe(obj)

        objs = obj.events

        targ = [ep.event_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_events_to_dataframe__block_noparents(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_events_to_dataframe(obj, parents=False)
        res1 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=True)
        res2 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=False)

        objs = obj.list_children_by_class('Event')

        targ = [ep.event_to_dataframe(iobj, parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_events_to_dataframe__block_parents_childfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_events_to_dataframe(obj)
        res1 = ep.multi_events_to_dataframe(obj, parents=True)
        res2 = ep.multi_events_to_dataframe(obj, child_first=True)
        res3 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=True)

        objs = obj.list_children_by_class('Event')

        targ = [ep.event_to_dataframe(iobj, parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res3.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_events_to_dataframe__block_parents_parentfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_events_to_dataframe(obj, child_first=False)
        res1 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=False)

        objs = obj.list_children_by_class('Event')

        targ = [ep.event_to_dataframe(iobj, parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_events_to_dataframe__list_noparents(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_events_to_dataframe(obj, parents=False)
        res1 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=True)
        res2 = ep.multi_events_to_dataframe(obj, parents=False,
                                            child_first=False)

        objs = (iobj.list_children_by_class('Event') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.event_to_dataframe(iobj, parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_events_to_dataframe__list_parents_childfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_events_to_dataframe(obj)
        res1 = ep.multi_events_to_dataframe(obj, parents=True)
        res2 = ep.multi_events_to_dataframe(obj, child_first=True)
        res3 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=True)

        objs = (iobj.list_children_by_class('Event') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.event_to_dataframe(iobj, parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res3.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_events_to_dataframe__list_parents_parentfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_events_to_dataframe(obj, child_first=False)
        res1 = ep.multi_events_to_dataframe(obj, parents=True,
                                            child_first=False)

        objs = (iobj.list_children_by_class('Event') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.event_to_dataframe(iobj, parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_events_to_dataframe__tuple_default(self):
        obj = tuple(fake_neo('Block', seed=i, n=3) for i in range(3))

        res0 = ep.multi_events_to_dataframe(obj)

        objs = (iobj.list_children_by_class('Event') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.event_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_events_to_dataframe__iter_default(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_events_to_dataframe(iter(obj))

        objs = (iobj.list_children_by_class('Event') for iobj in obj)
        objs = list(chain.from_iterable(objs))
        targ = [ep.event_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_events_to_dataframe__dict_default(self):
        obj = dict((i, fake_neo('Block', seed=i, n=3)) for i in range(3))

        res0 = ep.multi_events_to_dataframe(obj)

        objs = (iobj.list_children_by_class('Event') for iobj in
                obj.values())
        objs = list(chain.from_iterable(objs))
        targ = [ep.event_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.labels))]
                   for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class MultiEpochsToDataframeTestCase(unittest.TestCase):
    def setUp(self):
        if hasattr(self, 'assertItemsEqual'):
            self.assertCountEqual = self.assertItemsEqual

    def test__multi_epochs_to_dataframe__single(self):
        obj = fake_neo('Epoch', seed=0, n=5)

        res0 = ep.multi_epochs_to_dataframe(obj)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=False)
        res2 = ep.multi_epochs_to_dataframe(obj, parents=True)
        res3 = ep.multi_epochs_to_dataframe(obj, child_first=True)
        res4 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=True)
        res5 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=True)
        res6 = ep.multi_epochs_to_dataframe(obj, child_first=False)
        res7 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=False)
        res8 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=False)

        targ = ep.epoch_to_dataframe(obj)

        keys = ep._extract_neo_attrs_safe(obj, parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = 1
        targlen = min(len(obj.times), len(obj.durations), len(obj.labels))

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))
        self.assertEqual(targwidth, len(res4.columns))
        self.assertEqual(targwidth, len(res5.columns))
        self.assertEqual(targwidth, len(res6.columns))
        self.assertEqual(targwidth, len(res7.columns))
        self.assertEqual(targwidth, len(res8.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))
        self.assertEqual(targlen, len(res4.index))
        self.assertEqual(targlen, len(res5.index))
        self.assertEqual(targlen, len(res6.index))
        self.assertEqual(targlen, len(res7.index))
        self.assertEqual(targlen, len(res8.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)
        self.assertCountEqual(keys, res4.columns.names)
        self.assertCountEqual(keys, res5.columns.names)
        self.assertCountEqual(keys, res6.columns.names)
        self.assertCountEqual(keys, res7.columns.names)
        self.assertCountEqual(keys, res8.columns.names)

        assert_array_equal(targ.values, res0.values)
        assert_array_equal(targ.values, res1.values)
        assert_array_equal(targ.values, res2.values)
        assert_array_equal(targ.values, res3.values)
        assert_array_equal(targ.values, res4.values)
        assert_array_equal(targ.values, res5.values)
        assert_array_equal(targ.values, res6.values)
        assert_array_equal(targ.values, res7.values)
        assert_array_equal(targ.values, res8.values)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)
        assert_frame_equal(targ, res4)
        assert_frame_equal(targ, res5)
        assert_frame_equal(targ, res6)
        assert_frame_equal(targ, res7)
        assert_frame_equal(targ, res8)

    def test__multi_epochs_to_dataframe__segment_default(self):
        obj = fake_neo('Segment', seed=0, n=5)

        res0 = ep.multi_epochs_to_dataframe(obj)

        objs = obj.epochs

        targ = [ep.epoch_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)
        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_epochs_to_dataframe__block_noparents(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_epochs_to_dataframe(obj, parents=False)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=True)
        res2 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=False)

        objs = obj.list_children_by_class('Epoch')

        targ = [ep.epoch_to_dataframe(iobj, parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_epochs_to_dataframe__block_parents_childfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_epochs_to_dataframe(obj)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=True)
        res2 = ep.multi_epochs_to_dataframe(obj, child_first=True)
        res3 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=True)

        objs = obj.list_children_by_class('Epoch')

        targ = [ep.epoch_to_dataframe(iobj, parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res3.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_epochs_to_dataframe__block_parents_parentfirst(self):
        obj = fake_neo('Block', seed=0, n=3)

        res0 = ep.multi_epochs_to_dataframe(obj, child_first=False)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=False)

        objs = obj.list_children_by_class('Epoch')

        targ = [ep.epoch_to_dataframe(iobj, parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_epochs_to_dataframe__list_noparents(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_epochs_to_dataframe(obj, parents=False)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=True)
        res2 = ep.multi_epochs_to_dataframe(obj, parents=False,
                                            child_first=False)

        objs = (iobj.list_children_by_class('Epoch') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.epoch_to_dataframe(iobj, parents=False, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=False,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)

    def test__multi_epochs_to_dataframe__list_parents_childfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_epochs_to_dataframe(obj)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=True)
        res2 = ep.multi_epochs_to_dataframe(obj, child_first=True)
        res3 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=True)

        objs = (iobj.list_children_by_class('Epoch') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.epoch_to_dataframe(iobj, parents=True, child_first=True)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))
        self.assertEqual(targwidth, len(res2.columns))
        self.assertEqual(targwidth, len(res3.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))
        self.assertEqual(targlen, len(res2.index))
        self.assertEqual(targlen, len(res3.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)
        self.assertCountEqual(keys, res2.columns.names)
        self.assertCountEqual(keys, res3.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res2.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res3.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

    def test__multi_epochs_to_dataframe__list_parents_parentfirst(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_epochs_to_dataframe(obj, child_first=False)
        res1 = ep.multi_epochs_to_dataframe(obj, parents=True,
                                            child_first=False)

        objs = (iobj.list_children_by_class('Epoch') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.epoch_to_dataframe(iobj, parents=True, child_first=False)
                for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=False).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))
        self.assertEqual(targwidth, len(res1.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))
        self.assertEqual(targlen, len(res1.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)
        self.assertCountEqual(keys, res1.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))
        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res1.values, dtype=np.float))

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)

    def test__multi_epochs_to_dataframe__tuple_default(self):
        obj = tuple(fake_neo('Block', seed=i, n=3) for i in range(3))

        res0 = ep.multi_epochs_to_dataframe(obj)

        objs = (iobj.list_children_by_class('Epoch') for iobj in obj)
        objs = list(chain.from_iterable(objs))

        targ = [ep.epoch_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_epochs_to_dataframe__iter_default(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]

        res0 = ep.multi_epochs_to_dataframe(iter(obj))

        objs = (iobj.list_children_by_class('Epoch') for iobj in obj)
        objs = list(chain.from_iterable(objs))
        targ = [ep.epoch_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)

    def test__multi_epochs_to_dataframe__dict_default(self):
        obj = dict((i, fake_neo('Block', seed=i, n=3)) for i in range(3))

        res0 = ep.multi_epochs_to_dataframe(obj)

        objs = (iobj.list_children_by_class('Epoch') for iobj in
                obj.values())
        objs = list(chain.from_iterable(objs))
        targ = [ep.epoch_to_dataframe(iobj) for iobj in objs]
        targ = ep._sort_inds(pd.concat(targ, axis=1), axis=1)

        keys = ep._extract_neo_attrs_safe(objs[0], parents=True,
                                          child_first=True).keys()
        keys = list(keys)

        targwidth = len(objs)
        targlen = [iobj.times[:min(len(iobj.times), len(iobj.durations),
                                   len(iobj.labels))] for iobj in objs]
        targlen = len(np.unique(np.hstack(targlen)))

        self.assertGreater(len(objs), 0)

        self.assertEqual(targwidth, len(targ.columns))
        self.assertEqual(targwidth, len(res0.columns))

        self.assertEqual(targlen, len(targ.index))
        self.assertEqual(targlen, len(res0.index))

        self.assertCountEqual(keys, targ.columns.names)
        self.assertCountEqual(keys, res0.columns.names)

        assert_array_equal(
            np.array(targ.values, dtype=np.float),
            np.array(res0.values, dtype=np.float))

        assert_frame_equal(targ, res0)


@unittest.skipUnless(HAVE_PANDAS, 'requires pandas')
class SliceSpiketrainTestCase(unittest.TestCase):
    def setUp(self):
        obj = [fake_neo('SpikeTrain', seed=i, n=3) for i in range(10)]
        self.obj = ep.multi_spiketrains_to_dataframe(obj)

    def test_single_none(self):
        targ_start = self.obj.columns.get_level_values('t_start').values
        targ_stop = self.obj.columns.get_level_values('t_stop').values

        res0 = ep.slice_spiketrain(self.obj)
        res1 = ep.slice_spiketrain(self.obj, t_start=None)
        res2 = ep.slice_spiketrain(self.obj, t_stop=None)
        res3 = ep.slice_spiketrain(self.obj, t_start=None, t_stop=None)

        res0_start = res0.columns.get_level_values('t_start').values
        res1_start = res1.columns.get_level_values('t_start').values
        res2_start = res2.columns.get_level_values('t_start').values
        res3_start = res3.columns.get_level_values('t_start').values

        res0_stop = res0.columns.get_level_values('t_stop').values
        res1_stop = res1.columns.get_level_values('t_stop').values
        res2_stop = res2.columns.get_level_values('t_stop').values
        res3_stop = res3.columns.get_level_values('t_stop').values
        targ = self.obj

        self.assertFalse(res0 is targ)
        self.assertFalse(res1 is targ)
        self.assertFalse(res2 is targ)
        self.assertFalse(res3 is targ)

        assert_frame_equal(targ, res0)
        assert_frame_equal(targ, res1)
        assert_frame_equal(targ, res2)
        assert_frame_equal(targ, res3)

        assert_array_equal(targ_start, res0_start)
        assert_array_equal(targ_start, res1_start)
        assert_array_equal(targ_start, res2_start)
        assert_array_equal(targ_start, res3_start)

        assert_array_equal(targ_stop, res0_stop)
        assert_array_equal(targ_stop, res1_stop)
        assert_array_equal(targ_stop, res2_stop)
        assert_array_equal(targ_stop, res3_stop)

    def test_single_t_start(self):
        targ_start = .0001
        targ_stop = self.obj.columns.get_level_values('t_stop').values

        res0 = ep.slice_spiketrain(self.obj, t_start=targ_start)
        res1 = ep.slice_spiketrain(self.obj, t_start=targ_start, t_stop=None)

        res0_start = res0.columns.get_level_values('t_start').unique().tolist()
        res1_start = res1.columns.get_level_values('t_start').unique().tolist()

        res0_stop = res0.columns.get_level_values('t_stop').values
        res1_stop = res1.columns.get_level_values('t_stop').values

        targ = self.obj.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # targ already has nan values, ignore comparing with nan
            targ[targ < targ_start] = np.nan

        self.assertFalse(res0 is targ)
        self.assertFalse(res1 is targ)

        assert_array_equal(targ, res0.values)
        assert_array_equal(targ, res1.values)

        self.assertEqual([targ_start], res0_start)
        self.assertEqual([targ_start], res1_start)

        assert_array_equal(targ_stop, res0_stop)
        assert_array_equal(targ_stop, res1_stop)

    def test_single_t_stop(self):
        targ_start = self.obj.columns.get_level_values('t_start').values
        targ_stop = .0009

        res0 = ep.slice_spiketrain(self.obj, t_stop=targ_stop)
        res1 = ep.slice_spiketrain(self.obj, t_stop=targ_stop, t_start=None)

        res0_start = res0.columns.get_level_values('t_start').values
        res1_start = res1.columns.get_level_values('t_start').values

        res0_stop = res0.columns.get_level_values('t_stop').unique().tolist()
        res1_stop = res1.columns.get_level_values('t_stop').unique().tolist()

        targ = self.obj.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # targ already has nan values, ignore comparing with nan
            targ[targ > targ_stop] = np.nan

        self.assertFalse(res0 is targ)
        self.assertFalse(res1 is targ)

        assert_array_equal(targ, res0.values)
        assert_array_equal(targ, res1.values)

        assert_array_equal(targ_start, res0_start)
        assert_array_equal(targ_start, res1_start)

        self.assertEqual([targ_stop], res0_stop)
        self.assertEqual([targ_stop], res1_stop)

    def test_single_both(self):
        targ_start = .0001
        targ_stop = .0009

        res0 = ep.slice_spiketrain(self.obj,
                                   t_stop=targ_stop, t_start=targ_start)

        res0_start = res0.columns.get_level_values('t_start').unique().tolist()

        res0_stop = res0.columns.get_level_values('t_stop').unique().tolist()

        targ = self.obj.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # targ already has nan values, ignore comparing with nan
            targ[targ < targ_start] = np.nan
            targ[targ > targ_stop] = np.nan

        self.assertFalse(res0 is targ)

        assert_array_equal(targ, res0.values)

        self.assertEqual([targ_start], res0_start)

        self.assertEqual([targ_stop], res0_stop)


if __name__ == '__main__':
    unittest.main()
