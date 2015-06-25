import numpy as np
import utility_functions as utils

class CSD(object):
    '''Baseclass of the CSD'''
    def __init__(self, ele_pos, pots):
        '''ele_pos.shape = (N, dim); numpy array and pots.shape = (N, t); numpy array
        Where dim is the dimension of the electrodes, N is the number of
        electrodes & t is the time points of measurements.

        '''
        self.validate(ele_pos, pots)
        self.ele_pos = ele_pos
        self.pots = pots

    def validate(self, ele_pos, pots):
        '''Basic checks to see if inputs are okay'''
        if ele_pos.shape[0] != pots.shape[0]:
            raise Exception("Number of measured potentials is not equal "
                            "to electrode number!")
        if ele_pos.shape[0] < 1+ele_pos.shape[1]: #Dim+1
            raise Exception("Number of electrodes must be at least :",
                            1+ele_pos.shape[1])
        if utils.check_for_duplicated_electrodes(ele_pos) is False:
            raise Exception("Error! Duplicated electrode!")
        return

    def method(self):
        '''Place holder for the actual method that computes the CSD.

        '''
        pass

    def values(self, pos_csd):
        '''Place holder for obtaining CSD at the pos_csd locations, it uses the method
        function to obtain the CSD.

        '''
        #return self.csd
        pass

    def sanity(self, true_csd, pos_csd):
        '''Useful for comparing TrueCSD with reconstructed CSD. Computes, the RMS error
        between the true_csd and the reconstructed csd at pos_csd using the
        method defined.

        '''
        csd = self.values(pos_csd)
        RMSE = np.sqrt(np.mean(np.square(true_csd - csd)))
        return RMSE

if __name__ == '__main__':
    ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5], [1.2, 1.2]])
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
    test_class = CSD(ele_pos, pots)
