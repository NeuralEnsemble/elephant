"""
Unit tests for the Unitary Events analysis

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import types
import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal

import elephant.unitary_event_analysis as ue
from elephant.test.download import download, ELEPHANT_TMP_DIR
from numpy.testing import assert_array_almost_equal
from elephant.spike_train_generation import homogeneous_poisson_process


class UETestCase(unittest.TestCase):

    def setUp(self):
        sts1_with_trial = [[26., 48., 78., 144., 178.],
                           [4., 45., 85., 123., 156., 185.],
                           [22., 53., 73., 88., 120., 147., 167., 193.],
                           [23., 49., 74., 116., 142., 166., 189.],
                           [5., 34., 54., 80., 108., 128., 150., 181.],
                           [18., 61., 107., 170.],
                           [62., 98., 131., 161.],
                           [37., 63., 86., 131., 168.],
                           [39., 76., 100., 127., 153., 198.],
                           [3., 35., 60., 88., 108., 141., 171., 184.],
                           [39., 170.],
                           [25., 68., 170.],
                           [19., 57., 84., 116., 157., 192.],
                           [17., 80., 131., 172.],
                           [33., 65., 124., 162., 192.],
                           [58., 87., 185.],
                           [19., 101., 174.],
                           [84., 118., 156., 198., 199.],
                           [5., 55., 67., 96., 114., 148., 172., 199.],
                           [61., 105., 131., 169., 195.],
                           [26., 96., 129., 157.],
                           [41., 85., 157., 199.],
                           [6., 30., 53., 76., 109., 142., 167., 194.],
                           [159.],
                           [6., 51., 78., 113., 154., 183.],
                           [138.],
                           [23., 59., 154., 185.],
                           [12., 14., 52., 54., 109., 145., 192.],
                           [29., 61., 84., 122., 145., 168.],
                           [26., 99.],
                           [3., 31., 55., 85., 108., 158., 191.],
                           [5., 37., 70., 119., 170.],
                           [38., 79., 117., 157., 192.],
                           [174.],
                           [114.],
                           []]
        sts2_with_trial = [[3., 119.],
                           [54., 155., 183.],
                           [35., 133.],
                           [25., 100., 176.],
                           [9., 98.],
                           [6., 97., 198.],
                           [7., 62., 148.],
                           [100., 158.],
                           [7., 62., 122., 179., 191.],
                           [125., 182.],
                           [30., 55., 127., 157., 196.],
                           [27., 70., 173.],
                           [82., 84., 198.],
                           [11., 29., 137.],
                           [5., 49., 61., 101., 142., 190.],
                           [78., 162., 178.],
                           [13., 14., 130., 172.],
                           [22.],
                           [16., 55., 109., 113., 175.],
                           [17., 33., 63., 102., 144., 189., 190.],
                           [58.],
                           [27., 30., 99., 145., 176.],
                           [10., 58., 116., 182.],
                           [14., 68., 104., 126., 162., 194.],
                           [56., 129., 196.],
                           [50., 78., 105., 152., 190., 197.],
                           [24., 66., 113., 117., 161.],
                           [9., 31., 81., 95., 136., 154.],
                           [10., 115., 185., 191.],
                           [71., 140., 157.],
                           [15., 27., 88., 102., 103., 151., 181., 188.],
                           [51., 75., 95., 134., 195.],
                           [18., 55., 75., 131., 186.],
                           [10., 16., 41., 42., 75., 127.],
                           [62., 76., 102., 145., 171., 183.],
                           [66., 71., 85., 140., 154.]]
        self.sts1_neo = [neo.SpikeTrain(
            i * pq.ms, t_stop=200 * pq.ms) for i in sts1_with_trial]
        self.sts2_neo = [neo.SpikeTrain(
            i * pq.ms, t_stop=200 * pq.ms) for i in sts2_with_trial]
        self.binary_sts = np.array([[[1, 1, 1, 1, 0],
                                     [0, 1, 1, 1, 0],
                                     [0, 1, 1, 0, 1]],
                                    [[1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1],
                                     [1, 1, 0, 1, 0]]])

    def test_hash_default(self):
        m = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                      [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        expected = np.array([77, 43, 23])
        h = ue.hash_from_pattern(m)
        self.assertTrue(np.all(expected == h))

    def test_hash_default_longpattern(self):
        m = np.zeros((100, 2))
        m[0, 0] = 1
        expected = np.array([2**99, 0])
        h = ue.hash_from_pattern(m)
        self.assertTrue(np.all(expected == h))

    def test_hash_inverse_longpattern(self):
        n_patterns = 100
        m = np.random.randint(low=0, high=2, size=(n_patterns, 2))
        h = ue.hash_from_pattern(m)
        m_inv = ue.inverse_hash_from_pattern(h, N=n_patterns)
        assert_array_equal(m, m_inv)

    def test_hash_ValueError_wrong_entries(self):
        m = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1], [1, 1, 0],
                      [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        self.assertRaises(ValueError, ue.hash_from_pattern, m)

    def test_hash_base_not_two(self):
        m = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                      [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        m = m.T
        base = 3
        expected = np.array([0, 9, 3, 1, 12, 10, 4, 13])
        h = ue.hash_from_pattern(m, base=base)
        self.assertTrue(np.all(expected == h))

    def test_invhash_ValueError(self):
        """
        The hash is larger than sum(2 ** range(N)).
        """
        self.assertRaises(
            ValueError, ue.inverse_hash_from_pattern, [128, 8], 4)

    def test_invhash_default_base(self):
        N = 3
        h = np.array([0, 4, 2, 1, 6, 5, 3, 7])
        expected = np.array(
            [[0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1, 1],
             [0, 0, 0, 1, 0, 1, 1, 1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.all(expected == m))

    def test_invhash_base_not_two(self):
        N = 3
        h = np.array([1, 4, 13])
        base = 3
        expected = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        m = ue.inverse_hash_from_pattern(h, N, base)
        self.assertTrue(np.all(expected == m))

    def test_invhash_shape_mat(self):
        N = 8
        h = np.array([178, 212, 232])
        expected = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [
                            1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.shape(m)[0] == N)

    def test_hash_invhash_consistency(self):
        m = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        inv_h = ue.hash_from_pattern(m)
        m1 = ue.inverse_hash_from_pattern(inv_h, N=8)
        self.assertTrue(np.all(m == m1))

    def test_n_emp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1], [1, 0, 1, 1, 1]])
        pattern_hash = [3, 15]
        expected1 = np.array([2., 1.])
        expected2 = [[0, 2], [4]]
        nemp, nemp_indices = ue.n_emp_mat(mat, pattern_hash)
        self.assertTrue(np.all(nemp == expected1))
        for item_cnt, item in enumerate(nemp_indices):
            self.assertTrue(np.allclose(expected2[item_cnt], item))

    def test_n_emp_mat_sum_trial_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([4, 6])
        N = 3
        expected1 = np.array([1., 3.])
        expected2 = [[[0], [3]], [[], [2, 4]]]
        n_emp, n_emp_idx = ue.n_emp_mat_sum_trial(mat, pattern_hash)
        self.assertTrue(np.all(n_emp == expected1))
        for item0_cnt, item0 in enumerate(n_emp_idx):
            for item1_cnt, item1 in enumerate(item0):
                self.assertTrue(
                    np.allclose(expected2[item0_cnt][item1_cnt], item1))

    def test_n_exp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1], [1, 0, 1, 1, 1]])
        pattern_hash = [3, 11]
        expected = np.array([1.536, 1.024])
        nexp = ue.n_exp_mat(mat, pattern_hash)
        self.assertTrue(np.allclose(expected, nexp))

    def test_n_exp_mat_sum_trial_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([5, 6])
        expected = np.array([1.56, 2.56])
        n_exp = ue.n_exp_mat_sum_trial(mat, pattern_hash)
        self.assertTrue(np.allclose(n_exp, expected))

    def test_n_exp_mat_sum_trial_TrialAverage(self):
        mat = self.binary_sts
        pattern_hash = np.array([5, 6])
        expected = np.array([1.62, 2.52])
        n_exp = ue.n_exp_mat_sum_trial(
            mat, pattern_hash, method='analytic_TrialAverage')
        self.assertTrue(np.allclose(n_exp, expected))

    def test_n_exp_mat_sum_trial_surrogate(self):
        mat = self.binary_sts
        pattern_hash = np.array([5])
        n_exp_anal = ue.n_exp_mat_sum_trial(
            mat, pattern_hash, method='analytic_TrialAverage')
        n_exp_surr = ue.n_exp_mat_sum_trial(
            mat, pattern_hash, method='surrogate_TrialByTrial',
            n_surrogates=1000)
        self.assertLess(
            a=np.abs(n_exp_anal[0] - np.mean(n_exp_surr)) / n_exp_anal[0],
            b=0.1)

    def test_gen_pval_anal_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5, 6])
        expected = np.array([1.56, 2.56])
        pval_func, n_exp = ue.gen_pval_anal(mat, pattern_hash)
        self.assertTrue(np.allclose(n_exp, expected))
        self.assertTrue(isinstance(pval_func, types.FunctionType))

    def test_jointJ_default(self):
        p_val = np.array([0.31271072, 0.01175031])
        expected = np.array([0.3419968, 1.92481736])
        self.assertTrue(np.allclose(ue.jointJ(p_val), expected))

    def test__rate_mat_avg_trial_default(self):
        mat = self.binary_sts
        expected = [0.9, 0.7, 0.6]
        self.assertTrue(np.allclose(expected, ue._rate_mat_avg_trial(mat)))

    def test__bintime(self):
        t = 13 * pq.ms
        bin_size = 3 * pq.ms
        expected = 4
        self.assertTrue(np.allclose(expected, ue._bintime(t, bin_size)))

    def test__winpos(self):
        t_start = 10 * pq.ms
        t_stop = 46 * pq.ms
        winsize = 15 * pq.ms
        winstep = 3 * pq.ms
        expected = [10., 13., 16., 19., 22., 25., 28., 31.] * pq.ms
        self.assertTrue(
            np.allclose(
                ue._winpos(t_start, t_stop, winsize,
                           winstep).rescale('ms').magnitude,
                expected.rescale('ms').magnitude))

    def test__UE_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([4, 6])
        expected_S = np.array([-0.26226523, 0.04959301])
        expected_idx = [[[0], [3]], [[], [2, 4]]]
        expected_nemp = np.array([1., 3.])
        expected_nexp = np.array([1.04, 2.56])
        expected_rate = np.array([0.9, 0.7, 0.6])
        S, rate_avg, n_exp, n_emp, indices = ue._UE(mat, pattern_hash)
        self.assertTrue(np.allclose(S, expected_S))
        self.assertTrue(np.allclose(n_exp, expected_nexp))
        self.assertTrue(np.allclose(n_emp, expected_nemp))
        self.assertTrue(np.allclose(expected_rate, rate_avg))
        for item0_cnt, item0 in enumerate(indices):
            for item1_cnt, item1 in enumerate(item0):
                self.assertTrue(
                    np.allclose(expected_idx[item0_cnt][item1_cnt], item1))

    def test__UE_surrogate(self):
        mat = self.binary_sts
        pattern_hash = np.array([4])
        _, rate_avg_surr, _, n_emp_surr, indices_surr =\
            ue._UE(
                mat,
                pattern_hash,
                method='surrogate_TrialByTrial',
                n_surrogates=100)
        _, rate_avg, _, n_emp, indices =\
            ue._UE(mat, pattern_hash, method='analytic_TrialByTrial')
        self.assertTrue(np.allclose(n_emp, n_emp_surr))
        self.assertTrue(np.allclose(rate_avg, rate_avg_surr))
        for item0_cnt, item0 in enumerate(indices):
            for item1_cnt, item1 in enumerate(item0):
                self.assertTrue(
                    np.allclose(
                        indices_surr[item0_cnt][item1_cnt],
                        item1))

    def test_jointJ_window_analysis(self):
        sts1 = self.sts1_neo
        sts2 = self.sts2_neo
        data = np.vstack((sts1, sts2)).T
        win_size = 100 * pq.ms
        bin_size = 5 * pq.ms
        win_step = 20 * pq.ms
        pattern_hash = [3]
        UE_dic = ue.jointJ_window_analysis(spiketrains=data,
                                           pattern_hash=pattern_hash,
                                           bin_size=bin_size,
                                           win_size=win_size,
                                           win_step=win_step)
        expected_Js = np.array(
            [0.57953708, 0.47348757, 0.1729669,
             0.01883295, -0.21934742, -0.80608759])
        expected_n_emp = np.array(
            [9., 9., 7., 7., 6., 6.])
        expected_n_exp = np.array(
            [6.5, 6.85, 6.05, 6.6, 6.45, 8.7])
        expected_rate = np.array(
            [[0.02166667, 0.01861111],
             [0.02277778, 0.01777778],
             [0.02111111, 0.01777778],
             [0.02277778, 0.01888889],
             [0.02305556, 0.01722222],
             [0.02388889, 0.02055556]]) * pq.kHz
        expected_indecis_tril26 = [4., 4.]
        expected_indecis_tril4 = [1.]
        assert_array_almost_equal(UE_dic['Js'].squeeze(), expected_Js)
        assert_array_almost_equal(UE_dic['n_emp'].squeeze(), expected_n_emp)
        assert_array_almost_equal(UE_dic['n_exp'].squeeze(), expected_n_exp)
        assert_array_almost_equal(UE_dic['rate_avg'].squeeze(), expected_rate)
        assert_array_almost_equal(UE_dic['indices']['trial26'],
                                  expected_indecis_tril26)
        assert_array_almost_equal(UE_dic['indices']['trial4'],
                                  expected_indecis_tril4)

        # check the input parameters
        input_params = UE_dic['input_parameters']
        self.assertEqual(input_params['pattern_hash'], pattern_hash)
        self.assertEqual(input_params['bin_size'], bin_size)
        self.assertEqual(input_params['win_size'], win_size)
        self.assertEqual(input_params['win_step'], win_step)
        self.assertEqual(input_params['method'], 'analytic_TrialByTrial')
        self.assertEqual(input_params['t_start'], 0 * pq.s)
        self.assertEqual(input_params['t_stop'], 200 * pq.ms)

    @staticmethod
    def load_gdf2Neo(fname, trigger, t_pre, t_post):
        """
        load and convert the gdf file to Neo format by
        cutting and aligning around a given trigger
        # codes for trigger events (extracted from a
        # documentation of an old file after
        # contacting Dr. Alexa Rihle)
        # 700 : ST (correct) 701, 702, 703, 704*
        # 500 : ST (error =5) 501, 502, 503, 504*
        # 1000: ST (if no selec) 1001,1002,1003,1004*
        # 11  : PS 111, 112, 113, 114
        # 12  : RS 121, 122, 123, 124
        # 13  : RT 131, 132, 133, 134
        # 14  : MT 141, 142, 143, 144
        # 15  : ES 151, 152, 153, 154
        # 16  : ES 161, 162, 163, 164
        # 17  : ES 171, 172, 173, 174
        # 19  : RW 191, 192, 193, 194
        # 20  : ET 201, 202, 203, 204
        """
        data = np.loadtxt(fname)

        if trigger == 'PS_4':
            trigger_code = 114
        if trigger == 'RS_4':
            trigger_code = 124
        if trigger == 'RS':
            trigger_code = 12
        if trigger == 'ES':
            trigger_code = 15
        # specify units
        units_id = np.unique(data[:, 0][data[:, 0] < 7])
        # indecies of the trigger
        sel_tr_idx = np.where(data[:, 0] == trigger_code)[0]
        # cutting the data by aligning on the trigger
        data_tr = []
        for id_tmp in units_id:
            data_sel_units = []
            for i_cnt, i in enumerate(sel_tr_idx):
                start_tmp = data[i][1] - t_pre.magnitude
                stop_tmp = data[i][1] + t_post.magnitude
                sel_data_tmp = np.array(
                    data[np.where((data[:, 1] <= stop_tmp) &
                                  (data[:, 1] >= start_tmp))])
                sp_units_tmp = sel_data_tmp[:, 1][
                    np.where(sel_data_tmp[:, 0] == id_tmp)[0]]
                if len(sp_units_tmp) > 0:
                    aligned_time = sp_units_tmp - start_tmp
                    data_sel_units.append(neo.SpikeTrain(
                        aligned_time * pq.ms, t_start=0 * pq.ms,
                        t_stop=t_pre + t_post))
                else:
                    data_sel_units.append(neo.SpikeTrain(
                        [] * pq.ms, t_start=0 * pq.ms,
                        t_stop=t_pre + t_post))
            data_tr.append(data_sel_units)
        data_tr.reverse()
        spiketrain = np.vstack([i for i in data_tr]).T
        return spiketrain

    # Test if the result of newly implemented Unitary Events in Elephant is
    # consistent with the result of Riehle et al 1997 Science
    # (see Rostami et al (2016) [Re] Science, 3(1):1-17).
    def test_Riehle_et_al_97_UE(self):
        url = "http://raw.githubusercontent.com/ReScience-Archives/Rostami-" \
              "Ito-Denker-Gruen-2017/master/data"
        files_to_download = (
            ("extracted_data.npy", "c4903666ce8a8a31274d6b11238a5ac3"),
            ("winny131_23.gdf", "cc2958f7b4fb14dbab71e17bba49bd10")
        )
        for filename, checksum in files_to_download:
            # The files will be downloaded to ELEPHANT_TMP_DIR
            download(url=f"{url}/{filename}", checksum=checksum)

        # load spike data of figure 2 of Riehle et al 1997
        spiketrain = self.load_gdf2Neo(ELEPHANT_TMP_DIR / "winny131_23.gdf",
                                       trigger='RS_4',
                                       t_pre=1799 * pq.ms,
                                       t_post=300 * pq.ms)

        # calculating UE ...
        winsize = 100 * pq.ms
        bin_size = 5 * pq.ms
        winstep = 5 * pq.ms
        pattern_hash = [3]
        t_start = spiketrain[0][0].t_start
        t_stop = spiketrain[0][0].t_stop
        t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)
        significance_level = 0.05

        UE = ue.jointJ_window_analysis(spiketrain,
                                       pattern_hash=pattern_hash,
                                       bin_size=bin_size,
                                       win_size=winsize,
                                       win_step=winstep,
                                       method='analytic_TrialAverage')
        # load extracted data from figure 2 of Riehle et al 1997
        extracted_data = np.load(ELEPHANT_TMP_DIR / 'extracted_data.npy',
                                 encoding='latin1', allow_pickle=True).item()
        Js_sig = ue.jointJ(significance_level)
        sig_idx_win = np.where(UE['Js'] >= Js_sig)[0]
        diff_UE_rep = []
        y_cnt = 0
        for trial_id in range(len(spiketrain)):
            trial_id_str = "trial{}".format(trial_id)
            indices_unique = np.unique(UE['indices'][trial_id_str])
            if len(indices_unique) > 0:
                # choose only the significant coincidences
                indices_unique_significant = []
                for j in sig_idx_win:
                    significant = indices_unique[np.where(
                        (indices_unique * bin_size >= t_winpos[j]) &
                        (indices_unique * bin_size < t_winpos[j] + winsize))]
                    indices_unique_significant.extend(significant)
                x_tmp = np.unique(indices_unique_significant) * \
                    bin_size.magnitude
                if len(x_tmp) > 0:
                    ue_trial = np.sort(extracted_data['ue'][y_cnt])
                    diff_UE_rep = np.append(
                        diff_UE_rep, x_tmp - ue_trial)
                    y_cnt += +1
        np.testing.assert_array_less(np.abs(diff_UE_rep), 0.3)

    def test_multiple_neurons(self):
        np.random.seed(12)
        spiketrains = [[homogeneous_poisson_process(
            rate=50 * pq.Hz, t_stop=1 * pq.s)
            for _ in range(5)] for neuron in range(3)]

        spiketrains = np.stack(spiketrains, axis=1)
        UE_dic = ue.jointJ_window_analysis(spiketrains, bin_size=5 * pq.ms,
                                           win_size=300 * pq.ms,
                                           win_step=100 * pq.ms)

        js_expected = [[0.6081138], [0.17796665], [-1.2601125],
                       [-0.2790147], [0.07804556], [0.7861176], [0.23452221],
                       [0.11624397]]
        indices_expected = {'trial2': [20, 30, 20, 30, 104, 104, 104],
                            'trial3': [21, 21, 65, 65, 65, 128, 128, 128],
                            'trial4': [8, 172, 172],
                            'trial0': [104, 106, 104, 106, 104, 106],
                            'trial1': [158, 158, 158, 188]}
        n_emp_expected = [[4.], [4.], [1.], [4.], [4.], [5.], [3.], [3.]]
        n_exp_expected = [[2.2858334], [3.2066667], [2.955], [4.485833],
                          [3.4622223], [2.723611], [2.166111], [2.4122221]]
        rate_expected = [[[0.04666667, 0.03266666, 0.04333333]],
                         [[0.04733333, 0.03666667, 0.044]],
                         [[0.04533333, 0.03466666, 0.046]],
                         [[0.04933333, 0.04466667, 0.04933333]],
                         [[0.04466667, 0.04266667, 0.046]],
                         [[0.04133333, 0.04466667, 0.044]],
                         [[0.04133333, 0.03666667, 0.04266667]],
                         [[0.03933334, 0.03866667, 0.04666667]]] * 1 / pq.ms
        input_parameters_expected = {'pattern_hash': [7],
                                     'bin_size': 5 * pq.ms,
                                     'win_size': 300 * pq.ms,
                                     'win_step': 100 * pq.ms,
                                     'method': 'analytic_TrialByTrial',
                                     't_start': 0 * pq.s,
                                     't_stop': 1 * pq.s, 'n_surrogates': 100}
        assert_array_almost_equal(UE_dic['Js'], js_expected)
        assert_array_almost_equal(UE_dic['n_emp'], n_emp_expected)
        assert_array_almost_equal(UE_dic['n_exp'], n_exp_expected)
        assert_array_almost_equal(UE_dic['rate_avg'], rate_expected)
        self.assertEqual(sorted(UE_dic['indices'].keys()),
                         sorted(indices_expected.keys()))
        for trial_key in indices_expected.keys():
            assert_array_equal(indices_expected[trial_key],
                               UE_dic['indices'][trial_key])
        self.assertEqual(UE_dic['input_parameters'], input_parameters_expected)


if __name__ == '__main__':
    unittest.main()
