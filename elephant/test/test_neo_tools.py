# -*- coding: utf-8 -*-
"""
Unit tests for the neo_tools module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

from itertools import chain
import unittest

from neo.test.generate_datasets import fake_neo, get_fake_values
from neo.test.tools import assert_same_sub_schema
from numpy.testing.utils import assert_array_equal

import elephant.neo_tools as nt


# A list of neo object attributes that contain arrays.
ARRAY_ATTRS = ['waveforms',
               'times',
               'durations',
               'labels',
               'index',
               'channel_names',
               'channel_ids',
               'coordinates',
               'array_annotations'
               ]


def strip_iter_values(targ, array_attrs=ARRAY_ATTRS):
    """Remove iterable, non-string values from a dictionary.

    `elephant.neo_tools.extract_neo_attrs` automatically strips out
    non-scalar values from attributes.  This function does the same to a
    manually-extracted dictionary.

    Parameters
    ----------
    targ : dict
           The dictionary of values to process.
    array_attrs : list of str objects, optional
                  The list of attribute names to remove.  If not specified,
                  uses `elephant.test.test_neo_tools.ARRAY_ATTRS`.

    Returns
    -------
    dict
        A copy of `targ` with the target values (if present) removed.

    Notes
    -----

    Always returns a copy, even if nothing was removed.

    This function has the values to remove hard-coded.  This is intentional
    to make sure that `extract_neo_attrs` is removing all the attributes
    it is supposed to and only the attributes it is supposed to.  Please do
    NOT change this to any sort of automatic detection, if it is missing
    values please add them manually.

    """
    targ = targ.copy()

    for attr in array_attrs:
        targ.pop(attr, None)
    return targ


class GetAllObjsTestCase(unittest.TestCase):
    def test__get_all_objs__float_valueerror(self):
        value = 5.
        with self.assertRaises(ValueError):
            nt._get_all_objs(value, 'Block')

    def test__get_all_objs__list_float_valueerror(self):
        value = [5.]
        with self.assertRaises(ValueError):
            nt._get_all_objs(value, 'Block')

    def test__get_all_objs__epoch_for_event_valueerror(self):
        value = fake_neo('Epoch', n=10, seed=0)
        with self.assertRaises(ValueError):
            nt._get_all_objs(value, 'Event')

    def test__get_all_objs__empty_list(self):
        targ = []
        value = []

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_nested_list(self):
        targ = []
        value = [[], [[], [[]]]]

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_dict(self):
        targ = []
        value = {}

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_nested_dict(self):
        targ = []
        value = {'a': {}, 'b': {'c': {}, 'd': {'e': {}}}}

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_itert(self):
        targ = []
        value = iter([])

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_nested_iter(self):
        targ = []
        value = iter([iter([]), iter([iter([]), iter([iter([])])])])

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__empty_nested_many(self):
        targ = []
        value = iter([[], {'c': [], 'd':(iter([]),)}])

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0)]
        value = fake_neo('SpikeTrain', n=10, seed=0)

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__list_spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0),
                fake_neo('SpikeTrain', n=10, seed=1)]
        value = [fake_neo('SpikeTrain', n=10, seed=0),
                 fake_neo('SpikeTrain', n=10, seed=1)]

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__nested_list_epoch(self):
        targ = [fake_neo('Epoch', n=10, seed=0),
                fake_neo('Epoch', n=10, seed=1)]
        value = [[fake_neo('Epoch', n=10, seed=0)],
                 fake_neo('Epoch', n=10, seed=1)]

        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__iter_spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0),
                fake_neo('SpikeTrain', n=10, seed=1)]
        value = iter([fake_neo('SpikeTrain', n=10, seed=0),
                      fake_neo('SpikeTrain', n=10, seed=1)])

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__nested_iter_epoch(self):
        targ = [fake_neo('Epoch', n=10, seed=0),
                fake_neo('Epoch', n=10, seed=1)]
        value = iter([iter([fake_neo('Epoch', n=10, seed=0)]),
                      fake_neo('Epoch', n=10, seed=1)])

        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__dict_spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0),
                fake_neo('SpikeTrain', n=10, seed=1)]
        value = {'a': fake_neo('SpikeTrain', n=10, seed=0),
                 'b': fake_neo('SpikeTrain', n=10, seed=1)}

        res = nt._get_all_objs(value, 'SpikeTrain')

        self.assertEqual(len(targ), len(res))
        for i, itarg in enumerate(targ):
            for ires in res:
                if itarg.annotations['seed'] == ires.annotations['seed']:
                    assert_same_sub_schema(itarg, ires)
                    break
            else:
                raise ValueError('Target %s not in result' % i)

    def test__get_all_objs__nested_dict_spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0),
                fake_neo('SpikeTrain', n=10, seed=1)]
        value = {'a': fake_neo('SpikeTrain', n=10, seed=0),
                 'b': {'c': fake_neo('SpikeTrain', n=10, seed=1)}}

        res = nt._get_all_objs(value, 'SpikeTrain')

        self.assertEqual(len(targ), len(res))
        for i, itarg in enumerate(targ):
            for ires in res:
                if itarg.annotations['seed'] == ires.annotations['seed']:
                    assert_same_sub_schema(itarg, ires)
                    break
            else:
                raise ValueError('Target %s not in result' % i)

    def test__get_all_objs__nested_many_spiketrain(self):
        targ = [fake_neo('SpikeTrain', n=10, seed=0),
                fake_neo('SpikeTrain', n=10, seed=1)]
        value = {'a': [fake_neo('SpikeTrain', n=10, seed=0)],
                 'b': iter([fake_neo('SpikeTrain', n=10, seed=1)])}

        res = nt._get_all_objs(value, 'SpikeTrain')

        self.assertEqual(len(targ), len(res))
        for i, itarg in enumerate(targ):
            for ires in res:
                if itarg.annotations['seed'] == ires.annotations['seed']:
                    assert_same_sub_schema(itarg, ires)
                    break
            else:
                raise ValueError('Target %s not in result' % i)

    def test__get_all_objs__unit_spiketrain(self):
        value = fake_neo('Unit', n=3, seed=0)
        targ = [fake_neo('SpikeTrain', n=3, seed=train.annotations['seed'])
                for train in value.spiketrains]

        for train in value.spiketrains:
            train.annotations.pop('i', None)
            train.annotations.pop('j', None)

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__block_epoch(self):
        value = fake_neo('Block', n=3, seed=0)
        targ = [fake_neo('Epoch', n=3, seed=train.annotations['seed'])
                for train in value.list_children_by_class('Epoch')]

        for epoch in value.list_children_by_class('Epoch'):
            epoch.annotations.pop('i', None)
            epoch.annotations.pop('j', None)

        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)


class ExtractNeoAttrsTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.block = fake_neo('Block', seed=0)

    def assert_dicts_equal(self, d1, d2):
        """Assert that two dictionaries are equal, taking into account arrays.

        Normally, `unittest.TestCase.assertEqual` doesn't work with
        dictionaries containing arrays.  This works around that.

        Parameters
        ----------

        d1, d2 : dict
                 The dictionaries to compare

        Returns
        -------

        Nothing

        Raises
        ------
        AssertionError : If the `d1` and `d2` are not equal.

        """
        try:
            self.assertEqual(d1, d2)
        except ValueError:
            for (key1, value1), (key2, value2) in zip(sorted(d1.items()),
                                                      sorted(d2.items())):
                self.assertEqual(key1, key2)
                try:
                    if hasattr(value1, 'keys') and hasattr(value2, 'keys'):
                        self.assert_dicts_equal(value1, value2)
                    elif hasattr(value1, 'dtype') and hasattr(value2, 'dtype'):
                        assert_array_equal(value1, value2)
                    else:
                        self.assertEqual(value1, value2)
                except BaseException as exc:
                    exc.args += ('key: %s' % key1,)
                    raise

    def test__extract_neo_attrs__spiketrain_noarray(self):
        obj = fake_neo('SpikeTrain', seed=0)
        targ = get_fake_values('SpikeTrain', seed=0)
        targ = strip_iter_values(targ)

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res11 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res21 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)

        self.assertEqual(targ, res00)
        self.assertEqual(targ, res10)
        self.assertEqual(targ, res20)
        self.assertEqual(targ, res01)
        self.assertEqual(targ, res11)
        self.assertEqual(targ, res21)

    def test__extract_neo_attrs__spiketrain_noarray_skip_none(self):
        obj = fake_neo('SpikeTrain', seed=0)
        targ = get_fake_values('SpikeTrain', seed=0)
        targ = strip_iter_values(targ)
        for key, value in targ.copy().items():
            if value is None:
                del targ[key]

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          skip_none=True)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True, skip_none=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False, skip_none=True)
        res01 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                          skip_none=True)
        res11 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True, skip_none=True)
        res21 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False, skip_none=True)

        self.assertEqual(targ, res00)
        self.assertEqual(targ, res10)
        self.assertEqual(targ, res20)
        self.assertEqual(targ, res01)
        self.assertEqual(targ, res11)
        self.assertEqual(targ, res21)

    def test__extract_neo_attrs__epoch_noarray(self):
        obj = fake_neo('Epoch', seed=0)
        targ = get_fake_values('Epoch', seed=0)
        targ = strip_iter_values(targ)

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res11 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res21 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)

        self.assertEqual(targ, res00)
        self.assertEqual(targ, res10)
        self.assertEqual(targ, res20)
        self.assertEqual(targ, res01)
        self.assertEqual(targ, res11)
        self.assertEqual(targ, res21)

    def test__extract_neo_attrs__event_noarray(self):
        obj = fake_neo('Event', seed=0)
        targ = get_fake_values('Event', seed=0)
        targ = strip_iter_values(targ)

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res11 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=True)
        res21 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                          child_first=False)

        self.assertEqual(targ, res00)
        self.assertEqual(targ, res10)
        self.assertEqual(targ, res20)
        self.assertEqual(targ, res01)
        self.assertEqual(targ, res11)
        self.assertEqual(targ, res21)

    def test__extract_neo_attrs__spiketrain_parents_empty_array(self):
        obj = fake_neo('SpikeTrain', seed=0)
        targ = get_fake_values('SpikeTrain', seed=0)
        del targ['times']

        res000 = nt.extract_neo_attributes(obj, parents=False)
        res100 = nt.extract_neo_attributes(
            obj, parents=False, child_first=True)
        res200 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)
        res010 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False)
        res110 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=True)
        res210 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=False)
        res001 = nt.extract_neo_attributes(obj, parents=True)
        res101 = nt.extract_neo_attributes(obj, parents=True, child_first=True)
        res201 = nt.extract_neo_attributes(
            obj, parents=True, child_first=False)
        res011 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res111 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=True)
        res211 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=False)

        self.assert_dicts_equal(targ, res000)
        self.assert_dicts_equal(targ, res100)
        self.assert_dicts_equal(targ, res200)
        self.assert_dicts_equal(targ, res010)
        self.assert_dicts_equal(targ, res110)
        self.assert_dicts_equal(targ, res210)
        self.assert_dicts_equal(targ, res001)
        self.assert_dicts_equal(targ, res101)
        self.assert_dicts_equal(targ, res201)
        self.assert_dicts_equal(targ, res011)
        self.assert_dicts_equal(targ, res111)
        self.assert_dicts_equal(targ, res211)

    @staticmethod
    def _fix_neo_issue_749(obj, targ):
        # TODO: remove once fixed
        # https://github.com/NeuralEnsemble/python-neo/issues/749
        num_times = len(targ['times'])
        obj = obj[:num_times]
        del targ['array_annotations']
        return obj

    def test__extract_neo_attrs__epoch_parents_empty_array(self):
        obj = fake_neo('Epoch', seed=0)
        targ = get_fake_values('Epoch', seed=0)

        obj = self._fix_neo_issue_749(obj, targ)
        del targ['times']

        res000 = nt.extract_neo_attributes(obj, parents=False)
        res100 = nt.extract_neo_attributes(
            obj, parents=False, child_first=True)
        res200 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)
        res010 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False)
        res110 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=True)
        res210 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=False)
        res001 = nt.extract_neo_attributes(obj, parents=True)
        res101 = nt.extract_neo_attributes(obj, parents=True, child_first=True)
        res201 = nt.extract_neo_attributes(
            obj, parents=True, child_first=False)
        res011 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res111 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=True)
        res211 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=False)

        self.assert_dicts_equal(targ, res000)
        self.assert_dicts_equal(targ, res100)
        self.assert_dicts_equal(targ, res200)
        self.assert_dicts_equal(targ, res010)
        self.assert_dicts_equal(targ, res110)
        self.assert_dicts_equal(targ, res210)
        self.assert_dicts_equal(targ, res001)
        self.assert_dicts_equal(targ, res101)
        self.assert_dicts_equal(targ, res201)
        self.assert_dicts_equal(targ, res011)
        self.assert_dicts_equal(targ, res111)
        self.assert_dicts_equal(targ, res211)

    def test__extract_neo_attrs__event_parents_empty_array(self):
        obj = fake_neo('Event', seed=0)
        targ = get_fake_values('Event', seed=0)
        del targ['times']

        res000 = nt.extract_neo_attributes(obj, parents=False)
        res100 = nt.extract_neo_attributes(
            obj, parents=False, child_first=True)
        res200 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)
        res010 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False)
        res110 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=True)
        res210 = nt.extract_neo_attributes(
            obj, parents=False, skip_array=False, child_first=False)
        res001 = nt.extract_neo_attributes(obj, parents=True)
        res101 = nt.extract_neo_attributes(obj, parents=True, child_first=True)
        res201 = nt.extract_neo_attributes(
            obj, parents=True, child_first=False)
        res011 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res111 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=True)
        res211 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                           child_first=False)

        self.assert_dicts_equal(targ, res000)
        self.assert_dicts_equal(targ, res100)
        self.assert_dicts_equal(targ, res200)
        self.assert_dicts_equal(targ, res010)
        self.assert_dicts_equal(targ, res110)
        self.assert_dicts_equal(targ, res210)
        self.assert_dicts_equal(targ, res001)
        self.assert_dicts_equal(targ, res101)
        self.assert_dicts_equal(targ, res201)
        self.assert_dicts_equal(targ, res011)
        self.assert_dicts_equal(targ, res111)
        self.assert_dicts_equal(targ, res211)

    def test__extract_neo_attrs__spiketrain_noparents_noarray(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        targ = get_fake_values('SpikeTrain', seed=obj.annotations['seed'])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res1['i']
        del res2['i']
        del res0['j']
        del res1['j']
        del res2['j']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__epoch_noparents_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        targ = get_fake_values('Epoch', seed=obj.annotations['seed'])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res1['i']
        del res2['i']
        del res0['j']
        del res1['j']
        del res2['j']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__event_noparents_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]
        targ = get_fake_values('Event', seed=obj.annotations['seed'])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res1['i']
        del res2['i']
        del res0['j']
        del res1['j']
        del res2['j']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__spiketrain_noparents_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        targ = get_fake_values('SpikeTrain', seed=obj.annotations['seed'])
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=False)
        res11 = nt.extract_neo_attributes(obj, parents=False, child_first=True)
        res21 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)

        del res00['i']
        del res10['i']
        del res20['i']
        del res01['i']
        del res11['i']
        del res21['i']
        del res00['j']
        del res10['j']
        del res20['j']
        del res01['j']
        del res11['j']
        del res21['j']

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res20)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)
        self.assert_dicts_equal(targ, res21)

    def test__extract_neo_attrs__epoch_noparents_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        targ = get_fake_values('Epoch', seed=obj.annotations['seed'])

        # 'times' is not in obj._necessary_attrs + obj._recommended_attrs
        obj = self._fix_neo_issue_749(obj, targ)
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=False)
        res11 = nt.extract_neo_attributes(obj, parents=False, child_first=True)
        res21 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)

        del res00['i']
        del res10['i']
        del res20['i']
        del res01['i']
        del res11['i']
        del res21['i']
        del res00['j']
        del res10['j']
        del res20['j']
        del res01['j']
        del res11['j']
        del res21['j']

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res20)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)
        self.assert_dicts_equal(targ, res21)

    def test__extract_neo_attrs__event_noparents_array(self):
        obj = self.block.list_children_by_class('Event')[0]
        targ = get_fake_values('Event', seed=obj.annotations['seed'])
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=False, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=True)
        res20 = nt.extract_neo_attributes(obj, parents=False, skip_array=False,
                                          child_first=False)
        res01 = nt.extract_neo_attributes(obj, parents=False)
        res11 = nt.extract_neo_attributes(obj, parents=False, child_first=True)
        res21 = nt.extract_neo_attributes(
            obj, parents=False, child_first=False)

        del res00['i']
        del res10['i']
        del res20['i']
        del res01['i']
        del res11['i']
        del res21['i']
        del res00['j']
        del res10['j']
        del res20['j']
        del res01['j']
        del res11['j']
        del res21['j']

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res20)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)
        self.assert_dicts_equal(targ, res21)

    def test__extract_neo_attrs__spiketrain_parents_childfirst_noarray(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]
        rcg = self.block.channel_indexes[0]
        unit = self.block.channel_indexes[0].units[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('ChannelIndex',
                                    seed=rcg.annotations['seed']))
        targ.update(get_fake_values('Unit', seed=unit.annotations['seed']))
        targ.update(get_fake_values('SpikeTrain',
                                    seed=obj.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        del res0['i']
        del res1['i']
        del res0['j']
        del res1['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']
        del res1['index']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__epoch_parents_childfirst_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Epoch', seed=obj.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        del res0['i']
        del res1['i']
        del res0['j']
        del res1['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']
        del res1['index']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__event_parents_childfirst_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Event', seed=obj.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        del res0['i']
        del res1['i']
        del res0['j']
        del res1['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']
        del res1['index']

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__spiketrain_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]
        rcg = self.block.channel_indexes[0]
        unit = self.block.channel_indexes[0].units[0]

        targ = get_fake_values('SpikeTrain', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Unit', seed=unit.annotations['seed']))
        targ.update(get_fake_values('ChannelIndex',
                                    seed=rcg.annotations['seed']))
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res0['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__epoch_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Epoch', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res0['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__event_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Event', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        del res0['i']
        del res0['j']
        # name clash between Block.index and ChannelIndex.index
        del res0['index']

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__spiketrain_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]
        unit = self.block.channel_indexes[0].units[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Unit', seed=unit.annotations['seed']))
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('SpikeTrain',
                                    seed=obj.annotations['seed']))
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        ignore_annotations = ('i', 'j', 'channel_names',
                              'channel_ids', 'coordinates')
        for res in (res00, res01, res10, res11):
            for attr in ignore_annotations:
                del res[attr]

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__epoch_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Epoch', seed=obj.annotations['seed']))

        obj = self._fix_neo_issue_749(obj, targ)
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        ignore_annotations = ('i', 'j')
        for res in (res00, res01, res10, res11):
            for attr in ignore_annotations:
                del res[attr]

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__event_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Block', seed=blk.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Event', seed=obj.annotations['seed']))
        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        del res00['i']
        del res10['i']
        del res01['i']
        del res11['i']
        del res00['j']
        del res10['j']
        del res01['j']
        del res11['j']

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__spiketrain_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]
        unit = self.block.channel_indexes[0].units[0]

        targ = get_fake_values('SpikeTrain', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Unit', seed=unit.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))
        del targ['times']
        del targ['index']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        ignore_annotations = ('i', 'j', 'index', 'channel_names',
                              'channel_ids', 'coordinates')
        for res in (res0, res1):
            for attr in ignore_annotations:
                del res[attr]

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)

    def test__extract_neo_attrs__epoch_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Epoch', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))

        obj = self._fix_neo_issue_749(obj, targ)
        del targ['times']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        del res0['i']
        del res1['i']
        del res0['j']
        del res1['j']

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)

    def test__extract_neo_attrs__event_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = get_fake_values('Event', seed=obj.annotations['seed'])
        targ.update(get_fake_values('Segment', seed=seg.annotations['seed']))
        targ.update(get_fake_values('Block', seed=blk.annotations['seed']))
        del targ['times']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        del res0['i']
        del res1['i']
        del res0['j']
        del res1['j']

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)


class GetAllSpiketrainsTestCase(unittest.TestCase):
    def test__get_all_spiketrains__spiketrain(self):
        obj = fake_neo('SpikeTrain', seed=0, n=5)
        res0 = nt.get_all_spiketrains(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    def test__get_all_spiketrains__unit(self):
        obj = fake_neo('Unit', seed=0, n=7)
        obj.spiketrains.append(obj.spiketrains[0])
        res0 = nt.get_all_spiketrains(obj)

        targ = fake_neo('Unit', seed=0, n=7).spiketrains

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__segment(self):
        obj = fake_neo('Segment', seed=0, n=5)
        obj.spiketrains.extend(obj.spiketrains)
        res0 = nt.get_all_spiketrains(obj)

        targ = fake_neo('Segment', seed=0, n=5).spiketrains

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__block(self):
        obj = fake_neo('Block', seed=0, n=3)
        iobj1 = obj.channel_indexes[0].units[0]
        obj.channel_indexes[0].units.append(iobj1)
        iobj2 = obj.channel_indexes[0].units[2].spiketrains[1]
        obj.channel_indexes[1].units[1].spiketrains.append(iobj2)
        res0 = nt.get_all_spiketrains(obj)

        targ = fake_neo('Block', seed=0, n=3)
        targ = targ.list_children_by_class('SpikeTrain')

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__list(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].channel_indexes[0].units[0]
        obj[2].channel_indexes[0].units.append(iobj1)
        iobj2 = obj[1].channel_indexes[1].units[2].spiketrains[1]
        obj[2].channel_indexes[0].units[1].spiketrains.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_spiketrains(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__tuple(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].channel_indexes[0].units[0]
        obj[2].channel_indexes[0].units.append(iobj1)
        iobj2 = obj[1].channel_indexes[1].units[2].spiketrains[1]
        obj[2].channel_indexes[0].units[1].spiketrains.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_spiketrains(tuple(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__iter(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].channel_indexes[0].units[0]
        obj[2].channel_indexes[0].units.append(iobj1)
        iobj2 = obj[1].channel_indexes[1].units[2].spiketrains[1]
        obj[2].channel_indexes[0].units[1].spiketrains.append(iobj2)
        obj.append(obj[1])
        res0 = nt.get_all_spiketrains(iter(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__dict(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].channel_indexes[0].units[0]
        obj[2].channel_indexes[0].units.append(iobj1)
        iobj2 = obj[1].channel_indexes[1].units[2].spiketrains[1]
        obj[2].channel_indexes[0].units[1].spiketrains.append(iobj2)
        obj.append(obj[1])
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_spiketrains(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


class GetAllEventsTestCase(unittest.TestCase):
    def test__get_all_events__event(self):
        obj = fake_neo('Event', seed=0, n=5)
        res0 = nt.get_all_events(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    def test__get_all_events__segment(self):
        obj = fake_neo('Segment', seed=0, n=5)
        obj.events.extend(obj.events)
        res0 = nt.get_all_events(obj)

        targ = fake_neo('Segment', seed=0, n=5).events

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__block(self):
        obj = fake_neo('Block', seed=0, n=3)
        iobj1 = obj.segments[0]
        obj.segments.append(iobj1)
        iobj2 = obj.segments[0].events[1]
        obj.segments[1].events.append(iobj2)
        res0 = nt.get_all_events(obj)

        targ = fake_neo('Block', seed=0, n=3)
        targ = targ.list_children_by_class('Event')

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__list(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_events(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__tuple(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_events(tuple(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__iter(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_events(iter(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__dict(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_events(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


class GetAllEpochsTestCase(unittest.TestCase):
    def test__get_all_epochs__epoch(self):
        obj = fake_neo('Epoch', seed=0, n=5)
        res0 = nt.get_all_epochs(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    def test__get_all_epochs__segment(self):
        obj = fake_neo('Segment', seed=0, n=5)
        obj.epochs.extend(obj.epochs)
        res0 = nt.get_all_epochs(obj)

        targ = fake_neo('Segment', seed=0, n=5).epochs

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__block(self):
        obj = fake_neo('Block', seed=0, n=3)
        iobj1 = obj.segments[0]
        obj.segments.append(iobj1)
        iobj2 = obj.segments[0].epochs[1]
        obj.segments[1].epochs.append(iobj2)
        res0 = nt.get_all_epochs(obj)

        targ = fake_neo('Block', seed=0, n=3)
        targ = targ.list_children_by_class('Epoch')

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__list(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_epochs(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__tuple(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_epochs(tuple(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__iter(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_epochs(iter(obj))

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__dict(self):
        obj = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_epochs(obj)

        targ = [fake_neo('Block', seed=i, n=3) for i in range(3)]
        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


if __name__ == '__main__':
    unittest.main()
