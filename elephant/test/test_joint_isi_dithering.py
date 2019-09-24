"""
unittests for joint-isi dithering module.

Original implementation by: Peter Bouss [p.bouss@fz-juelich.de]
:copyright: Copyright 2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import unittest
import neo
import numpy as np
import quantities as pq
import elephant.joint_isi_dithering as jisid
import elephant.spike_train_generation as stg

python_version_major = sys.version_info.major

np.random.seed(0)


class JointISITestCase(unittest.TestCase):

    def test_joint_isi_dithering_format(self):

        rate = 100.*pq.Hz
        t_stop = 1.*pq.s
        st = stg.homogeneous_poisson_process(rate, t_stop=t_stop)
        n_surr = 2
        dither = 10 * pq.ms
        surrs = jisid.joint_isi_dithering(st, n_surr=n_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_joint_isi_dithering_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = jisid.joint_isi_dithering(st, n_surr=1)[0]
        self.assertEqual(len(surrog), 0)


def suite():
    suite = unittest.makeSuite(JointISITestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
