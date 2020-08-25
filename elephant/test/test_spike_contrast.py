import unittest
import elephant.spike_contrast as spc
import elephant.spike_train_generation as stgen
import numpy as np
from quantities import Hz, ms
import json


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_spike_contrast(self):
        np.random.seed(24)  # to make the results reproducible
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_4 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_5 = stgen.homogeneous_poisson_process(rate=1*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_6 = stgen.homogeneous_poisson_process(rate=1*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_trains = np.array(
                             [spike_train_1, spike_train_2, spike_train_3, spike_train_4, spike_train_5, spike_train_6])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 10000)), 0.2098687702924583)

    def test_spike_contrast_1(self):
        spike_train = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_trains = np.array([spike_train, spike_train])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 10000)), 1.0)

    def test_spike_contrast_2(self):
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)

        spike_trains = np.array([spike_train_1, spike_train_2, spike_train_3])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 20000)), 0.5)

    def test_get_theta_and_n_per_bin(self):
        spike_trains = np.array([[1, 1, 1],
                                 [2, 2, 2],
                                 [3, 3, 2.5],
                                 [9, 9, 0]])
        spike_trains = np.where(spike_trains == 0, np.nan, spike_trains)
        bin_size = 5
        t_start = 0
        t_stop = 10
        theta, n = spc.get_theta_and_n_per_bin(spike_trains, t_start, t_stop, bin_size)
        expected_theta = np.array([9, 3, 2])
        expected_n = np.array([3, 3, 2])
        self.assertTrue((theta == expected_theta).all())
        self.assertTrue((n == expected_n).all())

    def test_binning_half_overlap(self):
        spike_train_i = np.array([1, 2, 3, 9])
        t_start = 0
        t_stop = 10
        bin_size = 5
        self.assertTrue(([3, 1, 1] == spc.binning_half_overlap(spike_train_i, t_start, t_stop, bin_size)).all())
        
    def test_spike_contrast_with_Izhikevich_network_auto(self):
        with open("elephant/test/Data_Izhikevich_network.json", "r") as read_file:
            data = json.load(read_file)

        for i in range(1, 21):
            network_number = "network" + str(i)
            for y in range(1, 21):
                simulation_number = "simulation" + str(y)
                spike_trains_raw = data[network_number][simulation_number]["spiketrains"]
                synchrony = data[network_number][simulation_number]["synchrony"]
                spike_trains = np.array([
                    np.array(spike_trains_raw[0]), np.array(spike_trains_raw[1]), np.array(spike_trains_raw[2]),
                    np.array(spike_trains_raw[3]), np.array(spike_trains_raw[4]), np.array(spike_trains_raw[5]),
                    np.array(spike_trains_raw[6]), np.array(spike_trains_raw[7]), np.array(spike_trains_raw[8]),
                    np.array(spike_trains_raw[9]), np.array(spike_trains_raw[10]), np.array(spike_trains_raw[11]),
                    np.array(spike_trains_raw[12]), np.array(spike_trains_raw[13]), np.array(spike_trains_raw[14]),
                    np.array(spike_trains_raw[15]), np.array(spike_trains_raw[16]), np.array(spike_trains_raw[17]),
                    np.array(spike_trains_raw[18]), np.array(spike_trains_raw[19]), np.array(spike_trains_raw[20]),
                    np.array(spike_trains_raw[21]), np.array(spike_trains_raw[22]), np.array(spike_trains_raw[23]),
                    np.array(spike_trains_raw[24]), np.array(spike_trains_raw[25]), np.array(spike_trains_raw[26]),
                    np.array(spike_trains_raw[27]), np.array(spike_trains_raw[28]), np.array(spike_trains_raw[29]),
                    np.array(spike_trains_raw[30]), np.array(spike_trains_raw[31]), np.array(spike_trains_raw[32]),
                    np.array(spike_trains_raw[33]), np.array(spike_trains_raw[34]), np.array(spike_trains_raw[35]),
                    np.array(spike_trains_raw[36]), np.array(spike_trains_raw[37]), np.array(spike_trains_raw[38]),
                    np.array(spike_trains_raw[39]), np.array(spike_trains_raw[40]), np.array(spike_trains_raw[41]),
                    np.array(spike_trains_raw[42]), np.array(spike_trains_raw[43]), np.array(spike_trains_raw[44]),
                    np.array(spike_trains_raw[45]), np.array(spike_trains_raw[46]), np.array(spike_trains_raw[47]),
                    np.array(spike_trains_raw[48]), np.array(spike_trains_raw[49]), np.array(spike_trains_raw[50]),
                    np.array(spike_trains_raw[51]), np.array(spike_trains_raw[52]), np.array(spike_trains_raw[53]),
                    np.array(spike_trains_raw[54]), np.array(spike_trains_raw[55]), np.array(spike_trains_raw[56]),
                    np.array(spike_trains_raw[57]), np.array(spike_trains_raw[58]), np.array(spike_trains_raw[59])])

                self.assertEqual(round(spc.spike_contrast(spike_trains, 0, 2), 5), synchrony)

if __name__ == '__main__':
    unittest.main()
