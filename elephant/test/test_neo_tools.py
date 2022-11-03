# -*- coding: utf-8 -*-
"""
Unit tests for the neo_tools module.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
import random
from itertools import chain
import copy
import unittest

import neo.core
# TODO: In Neo 0.10.0, SpikeTrainList ist not exposed in __init__.py of
# neo.core. Remove the following line if SpikeTrainList is accessible via
# neo.core
from neo.core.spiketrainlist import SpikeTrainList

from neo.test.generate_datasets import generate_one_simple_block, \
    generate_one_simple_segment, \
    random_event, random_epoch, random_spiketrain
from neo.test.tools import assert_same_sub_schema

from numpy.testing.utils import assert_array_equal

import elephant.neo_tools as nt

# A list of neo object attributes that contain arrays.
ARRAY_ATTRS = ['waveforms',
               'times',
               'durations',
               'labels',
               # 'index',
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
    def setUp(self):
        random.seed(4245)
        self.spiketrain = random_spiketrain(
            'Single SpikeTrain', seed=random.random())
        self.spiketrain_list = [
            random_spiketrain('SpikeTrain', seed=random.random()),
            random_spiketrain('SpikeTrain', seed=random.random())]
        self.spiketrain_dict = {
            'a': random_spiketrain('SpikeTrain', seed=random.random()),
            123: random_spiketrain('SpikeTrain', seed=random.random())}

        self.epoch = random_epoch()
        self.epoch_list = [
            random_epoch(), random_epoch()]
        self.epoch_dict = {
            'a': random_epoch(), 123: random_epoch()}

    def test__get_all_objs__float_valueerror(self):
        value = 5.
        with self.assertRaises(ValueError):
            nt._get_all_objs(value, 'Block')

    def test__get_all_objs__list_float_valueerror(self):
        value = [5.]
        with self.assertRaises(ValueError):
            nt._get_all_objs(value, 'Block')

    def test__get_all_objs__epoch_for_event_valueerror(self):
        value = self.epoch
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
        value = iter([[], {'c': [], 'd': (iter([]),)}])

        res = nt._get_all_objs(value, 'Block')

        self.assertEqual(targ, res)

    def test__get_all_objs__spiketrain(self):
        value = self.spiketrain
        targ = [self.spiketrain]

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__list_spiketrain(self):
        value = self.spiketrain_list
        targ = self.spiketrain_list

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__nested_list_epoch(self):
        targ = self.epoch_list
        value = [self.epoch_list]

        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__iter_spiketrain(self):
        targ = self.spiketrain_list
        value = iter([self.spiketrain_list[0],
                      self.spiketrain_list[1]])

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__nested_iter_epoch(self):
        targ = self.epoch_list
        value = iter([iter(self.epoch_list)])

        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__dict_spiketrain(self):
        targ = [self.spiketrain_dict['a'], self.spiketrain_dict[123]]
        value = self.spiketrain_dict

        res = nt._get_all_objs(value, 'SpikeTrain')

        self.assertEqual(len(targ), len(res))
        for t, r in zip(targ, res):
            assert_same_sub_schema(t, r)

    def test__get_all_objs__nested_dict_spiketrain(self):
        targ = self.spiketrain_list
        value = {'a': self.spiketrain_list[0],
                 'b': {'c': self.spiketrain_list[1]}}

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
        targ = self.spiketrain_list
        value = {'a': [self.spiketrain_list[0]],
                 'b': iter([self.spiketrain_list[1]])}

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
        value = neo.core.Group(
            self.spiketrain_list,
            name='Unit')
        targ = self.spiketrain_list

        for train in value.spiketrains:
            train.annotations.pop('i', None)
            train.annotations.pop('j', None)

        res = nt._get_all_objs(value, 'SpikeTrain')

        assert_same_sub_schema(targ, res)

    def test__get_all_objs__block_epoch(self):
        value = generate_one_simple_block('Block', n=3, seed=0)
        targ = [train for train in value.list_children_by_class('Epoch')]
        res = nt._get_all_objs(value, 'Epoch')

        assert_same_sub_schema(targ, res)


class ExtractNeoAttrsTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.block = generate_one_simple_block(
            nb_segment=3,
            supported_objects=[
                neo.core.Block, neo.core.Segment,
                neo.core.SpikeTrain,
                neo.core.Event, neo.core.Epoch])

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
        obj = random_spiketrain()

        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        obj = random_spiketrain()

        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        obj = random_epoch()
        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        obj = random_event()
        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        obj = random_spiketrain()
        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        obj = random_epoch()
        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

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
        obj = random_event()
        targ = copy.deepcopy(obj.annotations)
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
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
        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__epoch_noparents_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        # 'times' is not in obj._necessary_attrs + obj._recommended_attrs
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__event_noparents_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=False, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=True)
        res2 = nt.extract_neo_attributes(obj, parents=False, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)
        self.assertEqual(targ, res2)

    def test__extract_neo_attrs__spiketrain_noparents_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        # 'times' is not in obj._necessary_attrs + obj._recommended_attrs
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

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res20)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)
        self.assert_dicts_equal(targ, res21)

    def test__extract_neo_attrs__epoch_noparents_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        # 'times' is not in obj._necessary_attrs + obj._recommended_attrs
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

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res20)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)
        self.assert_dicts_equal(targ, res21)

    def test__extract_neo_attrs__event_noparents_array(self):
        obj = self.block.list_children_by_class('Event')[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        # 'times' is not in obj._necessary_attrs + obj._recommended_attrs
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

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__epoch_parents_childfirst_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__event_parents_childfirst_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True)
        res1 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=True)

        self.assertEqual(targ, res0)
        self.assertEqual(targ, res1)

    def test__extract_neo_attrs__spiketrain_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])
        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__epoch_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__event_parents_parentfirst_noarray(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ = strip_iter_values(targ)

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=True,
                                         child_first=False)

        self.assertEqual(targ, res0)

    def test__extract_neo_attrs__spiketrain_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__epoch_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__event_parents_childfirst_array(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(blk.annotations)
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(obj.annotations))
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        del targ['times']

        res00 = nt.extract_neo_attributes(obj, parents=True, skip_array=False)
        res10 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                          child_first=True)
        res01 = nt.extract_neo_attributes(obj, parents=True)
        res11 = nt.extract_neo_attributes(obj, parents=True, child_first=True)

        self.assert_dicts_equal(targ, res00)
        self.assert_dicts_equal(targ, res10)
        self.assert_dicts_equal(targ, res01)
        self.assert_dicts_equal(targ, res11)

    def test__extract_neo_attrs__spiketrain_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('SpikeTrain')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.SpikeTrain._necessary_attrs +
                neo.SpikeTrain._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        del targ['times']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)

    def test__extract_neo_attrs__epoch_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('Epoch')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Epoch._necessary_attrs +
                neo.Epoch._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        del targ['times']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)

    def test__extract_neo_attrs__event_parents_parentfirst_array(self):
        obj = self.block.list_children_by_class('Event')[0]
        blk = self.block
        seg = self.block.segments[0]

        targ = copy.deepcopy(obj.annotations)
        targ["array_annotations"] = copy.deepcopy(
            dict(obj.array_annotations))
        for i, attr in enumerate(
                neo.Event._necessary_attrs +
                neo.Event._recommended_attrs):
            targ[attr[0]] = getattr(obj, attr[0])

        targ.update(copy.deepcopy(seg.annotations))
        for i, attr in enumerate(
                neo.Segment._necessary_attrs +
                neo.Segment._recommended_attrs):
            targ[attr[0]] = getattr(seg, attr[0])

        targ.update(copy.deepcopy(blk.annotations))
        for i, attr in enumerate(
                neo.Block._necessary_attrs +
                neo.Block._recommended_attrs):
            targ[attr[0]] = getattr(blk, attr[0])

        del targ['times']

        res0 = nt.extract_neo_attributes(obj, parents=True, skip_array=False,
                                         child_first=False)
        res1 = nt.extract_neo_attributes(obj, parents=True, child_first=False)

        self.assert_dicts_equal(targ, res0)
        self.assert_dicts_equal(targ, res1)


class GetAllSpiketrainsTestCase(unittest.TestCase):
    def test__get_all_spiketrains__spiketrain(self):
        obj = random_spiketrain()
        res0 = nt.get_all_spiketrains(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    # Todo: Units are no longer supported, but is a test for
    # neo.Group required instead?
    # def test__get_all_spiketrains__unit(self):
    #     obj = generate_one_simple_block(
    #         nb_segment=3,
    #         supported_objects=[
    #             neo.core.Block, neo.core.Segment,
    #             neo.core.SpikeTrain, neo.core.Group])
    #     targ = copy.deepcopy(obj)
    #
    #     obj.groups[0].spiketrains.append(obj.groups[0].spiketrains[0])
    #     res0 = nt.get_all_spiketrains(obj)
    #
    #     targ = targ.spiketrains
    #
    #     self.assertTrue(len(res0) > 0)
    #
    #     self.assertEqual(len(targ), len(res0))
    #
    #     assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__segment(self):
        obj = generate_one_simple_segment(
            supported_objects=[neo.core.Segment, neo.core.SpikeTrain])
        targ = copy.deepcopy(obj)
        obj.spiketrains.append(obj.spiketrains[0])
        # TODO: The following is the original line of the test, however, this
        # fails with Neo 0.10.0
        # Reinstate once issue is fixed
        # obj.spiketrains.extend(obj.spiketrains)

        res0 = nt.get_all_spiketrains(obj)

        targ = targ.spiketrains

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__block(self):
        obj = generate_one_simple_block(
            nb_segment=3,
            supported_objects=[
                neo.core.Block, neo.core.Segment, neo.core.SpikeTrain])
        targ = copy.deepcopy(obj)

        iobj1 = obj.segments[0]
        obj.segments.append(iobj1)
        iobj2 = obj.segments[0].spiketrains[1]
        obj.segments[1].spiketrains.append(iobj2)
        res0 = nt.get_all_spiketrains(obj)

        targ = SpikeTrainList(targ.list_children_by_class('SpikeTrain'))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__list(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.SpikeTrain])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].spiketrains[1]
        obj[2].segments[1].spiketrains.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_spiketrains(obj)

        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = SpikeTrainList(list(chain.from_iterable(targ)))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__tuple(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.SpikeTrain])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].spiketrains[1]
        obj[2].segments[1].spiketrains.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_spiketrains(tuple(obj))

        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = SpikeTrainList(list(chain.from_iterable(targ)))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__iter(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.SpikeTrain])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].spiketrains[1]
        obj[2].segments[1].spiketrains.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_spiketrains(obj)
        res0 = nt.get_all_spiketrains(iter(obj))

        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = SpikeTrainList(list(chain.from_iterable(targ)))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_spiketrains__dict(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.SpikeTrain])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].spiketrains[1]
        obj[2].segments[1].spiketrains.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_spiketrains(obj)
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_spiketrains(obj)

        targ = [iobj.list_children_by_class('SpikeTrain') for iobj in targ]
        targ = SpikeTrainList(list(chain.from_iterable(targ)))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


class GetAllEventsTestCase(unittest.TestCase):
    def test__get_all_events__event(self):
        obj = random_event()
        res0 = nt.get_all_events(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    def test__get_all_events__segment(self):
        obj = generate_one_simple_segment(
            supported_objects=[neo.core.Segment, neo.core.Event])
        targ = copy.deepcopy(obj)

        obj.events.extend(obj.events)
        res0 = nt.get_all_events(obj)

        targ = targ.events

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__block(self):
        obj = generate_one_simple_block(
            nb_segment=3,
            supported_objects=[
                neo.core.Block, neo.core.Segment, neo.core.Event])
        targ = copy.deepcopy(obj)

        iobj1 = obj.segments[0]
        obj.segments.append(iobj1)
        iobj2 = obj.segments[0].events[1]
        obj.segments[1].events.append(iobj2)
        res0 = nt.get_all_events(obj)

        targ = targ.list_children_by_class('Event')

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__list(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Event])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_events(obj)

        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__tuple(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Event])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_events(tuple(obj))

        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__iter(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Event])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_events(iter(obj))

        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_events__dict(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Event])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].events[1]
        obj[2].segments[1].events.append(iobj2)
        obj.append(obj[0])
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_events(obj)

        targ = [iobj.list_children_by_class('Event') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


class GetAllEpochsTestCase(unittest.TestCase):
    def test__get_all_epochs__epoch(self):
        obj = random_epoch()
        res0 = nt.get_all_epochs(obj)

        targ = obj

        self.assertEqual(1, len(res0))

        assert_same_sub_schema(targ, res0[0])

    def test__get_all_epochs__segment(self):
        obj = generate_one_simple_segment(
            supported_objects=[neo.core.Segment, neo.core.Epoch])
        targ = copy.deepcopy(obj)
        obj.epochs.extend(obj.epochs)
        res0 = nt.get_all_epochs(obj)

        targ = targ.epochs

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__block(self):
        obj = generate_one_simple_block(
            nb_segment=3,
            supported_objects=[
                neo.core.Block, neo.core.Segment, neo.core.Epoch])
        targ = copy.deepcopy(obj)

        iobj1 = obj.segments[0]
        obj.segments.append(iobj1)
        iobj2 = obj.segments[0].epochs[1]
        obj.segments[1].epochs.append(iobj2)
        res0 = nt.get_all_epochs(obj)

        targ = targ.list_children_by_class('Epoch')

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__list(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Epoch])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[-1])
        res0 = nt.get_all_epochs(obj)

        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__tuple(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Epoch])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_epochs(tuple(obj))

        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__iter(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Epoch])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        res0 = nt.get_all_epochs(iter(obj))

        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)

    def test__get_all_epochs__dict(self):
        obj = [
            generate_one_simple_block(
                nb_segment=3,
                supported_objects=[
                    neo.core.Block, neo.core.Segment, neo.core.Epoch])
            for _ in range(3)]
        targ = copy.deepcopy(obj)
        obj.append(obj[-1])
        iobj1 = obj[2].segments[0]
        obj[2].segments.append(iobj1)
        iobj2 = obj[1].segments[2].epochs[1]
        obj[2].segments[1].epochs.append(iobj2)
        obj.append(obj[0])
        obj = dict((i, iobj) for i, iobj in enumerate(obj))
        res0 = nt.get_all_epochs(obj)

        targ = [iobj.list_children_by_class('Epoch') for iobj in targ]
        targ = list(chain.from_iterable(targ))

        self.assertTrue(len(res0) > 0)

        self.assertEqual(len(targ), len(res0))

        assert_same_sub_schema(targ, res0)


if __name__ == '__main__':
    unittest.main()
