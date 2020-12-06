import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal

from elephant.parallel import SingleProcess, ProcessPoolExecutor
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import mean_firing_rate


class TestParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executors_cls = [SingleProcess, ProcessPoolExecutor]
        try:
            from elephant.parallel.mpi import MPIPoolExecutor, MPICommExecutor
            cls.executors_cls.extend([MPIPoolExecutor, MPICommExecutor])
        except ImportError:
            # mpi4py is not installed
            pass

        np.random.seed(28)
        n_spiketrains = 10
        cls.spiketrains = tuple(
            homogeneous_poisson_process(
                rate=10 * pq.Hz, t_stop=10 * pq.s, as_array=True)
            for _ in range(n_spiketrains)
        )
        cls.mean_fr = tuple(map(mean_firing_rate, cls.spiketrains))

    def test_mean_firing_rate(self):
        for executor_cls in self.executors_cls:
            with self.subTest(executor_cls=executor_cls):
                executor = executor_cls()
                mean_fr = executor.execute(handler=mean_firing_rate,
                                           args_iterate=self.spiketrains)
                assert_array_almost_equal(mean_fr, self.mean_fr)


if __name__ == '__main__':
    unittest.main()
