"""
Unit tests for the Unitary Events analysis

:copyright: Copyright 2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
import quantities as pq
import types
import elephant.unitary_event_analysis as ue
import neo

class UETestCase(unittest.TestCase):

    def setUp(self):
        sts1_with_trial = [[  26.,   48.,   78.,  144.,  178.],
                           [   4.,   45.,   85.,  123.,  156.,  185.],
                           [  22.,   53.,   73.,   88.,  120.,  147.,  167.,  193.],
                           [  23.,   49.,   74.,  116.,  142.,  166.,  189.],
                           [   5.,   34.,   54.,   80.,  108.,  128.,  150.,  181.],
                           [  18.,   61.,  107.,  170.],
                           [  62.,   98.,  131.,  161.],
                           [  37.,   63.,   86.,  131.,  168.],
                           [  39.,   76.,  100.,  127.,  153.,  198.],
                           [   3.,   35.,   60.,   88.,  108.,  141.,  171.,  184.],
                           [  39.,  170.],
                           [  25.,   68.,  170.],
                           [  19.,   57.,   84.,  116.,  157.,  192.],
                           [  17.,   80.,  131.,  172.],
                           [  33.,   65.,  124.,  162.,  192.],
                           [  58.,   87.,  185.],
                           [  19.,  101.,  174.],
                           [  84.,  118.,  156.,  198.,  199.],
                           [   5.,   55.,   67.,   96.,  114.,  148.,  172.,  199.],
                           [  61.,  105.,  131.,  169.,  195.],
                           [  26.,   96.,  129.,  157.],
                           [  41.,   85.,  157.,  199.],
                           [   6.,   30.,   53.,   76.,  109.,  142.,  167.,  194.],
                           [ 159.],
                           [   6.,   51.,   78.,  113.,  154.,  183.],
                           [ 138.],
                           [  23.,   59.,  154.,  185.],
                           [  12.,   14.,   52.,   54.,  109.,  145.,  192.],
                           [  29.,   61.,   84.,  122.,  145.,  168.],
                           [ 26.,  99.],
                           [   3.,   31.,   55.,   85.,  108.,  158.,  191.],
                           [   5.,   37.,   70.,  119.,  170.],
                           [  38.,   79.,  117.,  157.,  192.],
                           [ 174.],
                           [ 114.],
                           []]
        sts2_with_trial = [[   3.,  119.],
                           [  54.,  155.,  183.],
                           [  35.,  133.],
                           [  25.,  100.,  176.],
                           [  9.,  98.],
                           [   6.,   97.,  198.],
                           [   7.,   62.,  148.],
                           [ 100.,  158.],
                           [   7.,   62.,  122.,  179.,  191.],
                           [ 125.,  182.],
                           [  30.,   55.,  127.,  157.,  196.],
                           [  27.,   70.,  173.],
                           [  82.,   84.,  198.],
                           [  11.,   29.,  137.],
                           [   5.,   49.,   61.,  101.,  142.,  190.],
                           [  78.,  162.,  178.],
                           [  13.,   14.,  130.,  172.],
                           [ 22.],
                           [  16.,   55.,  109.,  113.,  175.],
                           [  17.,   33.,   63.,  102.,  144.,  189.,  190.],
                           [ 58.],
                           [  27.,   30.,   99.,  145.,  176.],
                           [  10.,   58.,  116.,  182.],
                           [  14.,   68.,  104.,  126.,  162.,  194.],
                           [  56.,  129.,  196.],
                           [  50.,   78.,  105.,  152.,  190.,  197.],
                           [  24.,   66.,  113.,  117.,  161.],
                           [   9.,   31.,   81.,   95.,  136.,  154.],
                           [  10.,  115.,  185.,  191.],
                           [  71.,  140.,  157.],
                           [  15.,   27.,   88.,  102.,  103.,  151.,  181.,  188.],
                           [  51.,   75.,   95.,  134.,  195.],
                           [  18.,   55.,   75.,  131.,  186.],
                           [  10.,   16.,   41.,   42.,   75.,  127.],
                           [  62.,   76.,  102.,  145.,  171.,  183.],
                           [  66.,   71.,   85.,  140.,  154.]]
        self.sts1_neo = [neo.SpikeTrain(
            i*pq.ms,t_stop = 200*pq.ms) for i in sts1_with_trial]
        self.sts2_neo = [neo.SpikeTrain(
            i*pq.ms,t_stop = 200*pq.ms) for i in sts2_with_trial]
        self.binary_sts = np.array([[[1, 1, 1, 1, 0],
                                     [0, 1, 1, 1, 0],
                                     [0, 1, 1, 0, 1]],
                                    [[1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1],
                                     [1, 1, 0, 1, 0]]])

    def test_hash_default(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        expected = np.array([77,43,23])
        h = ue.hash_from_pattern(m, N=8)
        self.assertTrue(np.all(expected == h))

    def test_hash_default_longpattern(self):
        m = np.zeros((100,2))
        m[0,0] = 1
        expected = np.array([2**99,0])
        h = ue.hash_from_pattern(m, N=100)
        self.assertTrue(np.all(expected == h))

    def test_hash_ValueError_wrong_orientation(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError, ue.hash_from_pattern, m, N=3)

    def test_hash_ValueError_wrong_entries(self):
        m = np.array([[0,0,0], [1,0,0], [0,2,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError, ue.hash_from_pattern, m, N=3)

    def test_hash_base_not_two(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        m = m.T
        base = 3
        expected = np.array([0,9,3,1,12,10,4,13])
        h = ue.hash_from_pattern(m, N=3, base=base)
        self.assertTrue(np.all(expected == h))

    ## TODO: write a test for ValueError in inverse_hash_from_pattern
    def test_invhash_ValueError(self):
        self.assertRaises(ValueError, ue.inverse_hash_from_pattern, [128, 8], 4)

    def test_invhash_default_base(self):
        N = 3
        h = np.array([0, 4, 2, 1, 6, 5, 3, 7])
        expected = np.array([[0, 1, 0, 0, 1, 1, 0, 1],[0, 0, 1, 0, 1, 0, 1, 1],[0, 0, 0, 1, 0, 1, 1, 1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.all(expected == m))

    def test_invhash_base_not_two(self):
        N = 3
        h = np.array([1,4,13])
        base = 3
        expected = np.array([[0,0,1],[0,1,1],[1,1,1]])
        m = ue.inverse_hash_from_pattern(h, N, base)
        self.assertTrue(np.all(expected == m))

    def test_invhash_shape_mat(self):
        N = 8
        h = np.array([178, 212, 232])
        expected = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.shape(m)[0] == N)

    def test_hash_invhash_consistency(self):
        m = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])
        inv_h = ue.hash_from_pattern(m, N=8)
        m1 = ue.inverse_hash_from_pattern(inv_h, N = 8)
        self.assertTrue(np.all(m == m1))

    def test_n_emp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 15]
        expected1 = np.array([ 2.,  1.])
        expected2 = [[0, 2], [4]]
        nemp,nemp_indices = ue.n_emp_mat(mat,N,pattern_hash)
        self.assertTrue(np.all(nemp == expected1))
        for item_cnt,item in enumerate(nemp_indices):
            self.assertTrue(np.allclose(expected2[item_cnt],item))

    def test_n_emp_mat_sum_trial_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([4,6])
        N = 3
        expected1 = np.array([ 1.,  3.])
        expected2 = [[[0], [3]],[[],[2,4]]]
        n_emp, n_emp_idx = ue.n_emp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.all(n_emp == expected1))
        for item0_cnt,item0 in enumerate(n_emp_idx):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(expected2[item0_cnt][item1_cnt],item1))

    def test_n_emp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_emp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])

    def test_n_exp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 11]
        expected = np.array([ 1.536,  1.024])
        nexp = ue.n_exp_mat(mat,N,pattern_hash)
        self.assertTrue(np.allclose(expected,nexp))

    def test_n_exp_mat_sum_trial_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.56,  2.56])
        n_exp = ue.n_exp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.allclose(n_exp,expected))

    def test_n_exp_mat_sum_trial_TrialAverage(self):
        mat = self.binary_sts
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.62,  2.52])
        n_exp = ue.n_exp_mat_sum_trial(mat, N, pattern_hash, method='analytic_TrialAverage')
        self.assertTrue(np.allclose(n_exp,expected))

    def test_n_exp_mat_sum_trial_surrogate(self):
        mat = self.binary_sts
        pattern_hash = np.array([5])
        N = 3
        n_exp_anal = ue.n_exp_mat_sum_trial(mat, N, pattern_hash, method='analytic_TrialAverage')
        n_exp_surr = ue.n_exp_mat_sum_trial(mat, N, pattern_hash, method='surrogate_TrialByTrial',n_surr = 1000)
        self.assertLess((np.abs(n_exp_anal[0]-np.mean(n_exp_surr))/n_exp_anal[0]),0.1)

    def test_n_exp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_exp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])

    def test_gen_pval_anal_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.56,  2.56])
        pval_func,n_exp = ue.gen_pval_anal(mat, N,pattern_hash)
        self.assertTrue(np.allclose(n_exp,expected))
        self.assertTrue(isinstance(pval_func, types.FunctionType))

    def test_jointJ_default(self):
        p_val = np.array([0.31271072,  0.01175031])
        expected = np.array([0.3419968 ,  1.92481736])
        self.assertTrue(np.allclose(ue.jointJ(p_val),expected))

    def test__rate_mat_avg_trial_default(self):
        mat = self.binary_sts
        expected = [0.9, 0.7,0.6]
        self.assertTrue(np.allclose(expected,ue._rate_mat_avg_trial(mat)))

    def test__bintime(self):
        t = 13*pq.ms
        binsize = 3*pq.ms
        expected = 4
        self.assertTrue(np.allclose(expected,ue._bintime(t,binsize)))
    def test__winpos(self):
        t_start = 10*pq.ms
        t_stop = 46*pq.ms
        winsize = 15*pq.ms
        winstep = 3*pq.ms
        expected = [ 10., 13., 16., 19., 22., 25., 28., 31.]*pq.ms
        self.assertTrue(
            np.allclose(
                ue._winpos(
                    t_start, t_stop, winsize,
                    winstep).rescale('ms').magnitude,
                expected.rescale('ms').magnitude))

    def test__UE_default(self):
        mat = self.binary_sts
        pattern_hash = np.array([4,6])
        N = 3
        expected_S = np.array([-0.26226523,  0.04959301])
        expected_idx = [[[0], [3]], [[], [2, 4]]]
        expected_nemp = np.array([ 1.,  3.])
        expected_nexp = np.array([ 1.04,  2.56])
        expected_rate = np.array([ 0.9,  0.7,  0.6])
        S, rate_avg, n_exp, n_emp,indices = ue._UE(mat,N,pattern_hash)
        self.assertTrue(np.allclose(S ,expected_S))
        self.assertTrue(np.allclose(n_exp ,expected_nexp))
        self.assertTrue(np.allclose(n_emp ,expected_nemp))
        self.assertTrue(np.allclose(expected_rate ,rate_avg))
        for item0_cnt,item0 in enumerate(indices):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(expected_idx[item0_cnt][item1_cnt],item1))

    def test__UE_surrogate(self):
        mat = self.binary_sts
        pattern_hash = np.array([4])
        N = 3
        _, rate_avg_surr, _, n_emp_surr,indices_surr =\
        ue._UE(mat, N, pattern_hash, method='surrogate_TrialByTrial', n_surr=100)
        _, rate_avg, _, n_emp,indices =\
        ue._UE(mat, N, pattern_hash, method='analytic_TrialByTrial')
        self.assertTrue(np.allclose(n_emp ,n_emp_surr))
        self.assertTrue(np.allclose(rate_avg ,rate_avg_surr))
        for item0_cnt,item0 in enumerate(indices):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(indices_surr[item0_cnt][item1_cnt],item1))

    def test_jointJ_window_analysis(self):
        sts1 = self.sts1_neo
        sts2 = self.sts2_neo
        data = np.vstack((sts1,sts2)).T
        winsize = 100*pq.ms
        binsize = 5*pq.ms
        winstep = 20*pq.ms
        pattern_hash = [3]
        UE_dic = ue.jointJ_window_analysis(data, binsize, winsize, winstep, pattern_hash)
        expected_Js = np.array(
            [ 0.57953708,  0.47348757,  0.1729669 ,  
              0.01883295, -0.21934742,-0.80608759])
        expected_n_emp = np.array(
            [ 9.,  9.,  7.,  7.,  6.,  6.])
        expected_n_exp = np.array(
            [ 6.5 ,  6.85,  6.05,  6.6 ,  6.45,  8.7 ])
        expected_rate = np.array(
            [[ 0.02166667,  0.01861111],
             [ 0.02277778,  0.01777778],
             [ 0.02111111,  0.01777778],
             [ 0.02277778,  0.01888889],
             [ 0.02305556,  0.01722222],
             [ 0.02388889,  0.02055556]])*pq.kHz
        expected_indecis_tril26 = [ 4.,    4.]
        expected_indecis_tril4 = [ 1.]
        self.assertTrue(np.allclose(UE_dic['Js'] ,expected_Js))
        self.assertTrue(np.allclose(UE_dic['n_emp'] ,expected_n_emp))
        self.assertTrue(np.allclose(UE_dic['n_exp'] ,expected_n_exp))
        self.assertTrue(np.allclose(
            UE_dic['rate_avg'].rescale('Hz').magnitude ,
            expected_rate.rescale('Hz').magnitude))
        self.assertTrue(np.allclose(
            UE_dic['indices']['trial26'],expected_indecis_tril26))
        self.assertTrue(np.allclose(
            UE_dic['indices']['trial4'],expected_indecis_tril4))
def suite():
    suite = unittest.makeSuite(UETestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

