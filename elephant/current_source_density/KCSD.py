"""
This script is used to generate Current Source Density Estimates, 
using the kCSD method Jan et.al (2012).

This was written by :
Chaitanya Chintaluri, 
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
"""
import numpy as np
from scipy import integrate, interpolate
from scipy.spatial import distance
from numpy.linalg import LinAlgError

from CSD import CSD

class KCSD(CSD):
    """KCSD - The base class for all the KCSD variants.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, electrodes. 
    The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        super(KCSD, self).__init__(ele_pos, pots)
        self.parameters(**kwargs)
        self.estimate_at() 
        self.place_basis() 
        self.create_src_dist_tables()
        self.method()
        return

    def parameters(self, **kwargs):
        """Defining the default values of the method passed as kwargs
        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.src_type = kwargs.get('src_type', 'gauss')
        self.sigma = kwargs.get('sigma', 1.0)
        self.h = kwargs.get('h', 1.0)
        self.n_src_init = kwargs.get('n_src_init', 1000)
        self.lambd = kwargs.get('lambd', 0.0)
        self.R_init = kwargs.get('R_init', 0.23)
        self.ext_x = kwargs.get('ext_x', 0.0)
        self.xmin = kwargs.get('xmin', np.min(self.ele_pos[:, 0]))
        self.xmax = kwargs.get('xmax', np.max(self.ele_pos[:, 0]))
        self.gdx = kwargs.get('gdx', 0.01*(self.xmax - self.xmin)) 
        if self.dim >= 2:
            self.ext_y = kwargs.get('ext_y', 0.0)
            self.ymin = kwargs.get('ymin', np.min(self.ele_pos[:, 1]))
            self.ymax = kwargs.get('ymax', np.max(self.ele_pos[:, 1]))
            self.gdy = kwargs.get('gdy', 0.01*(self.ymax - self.ymin))
        if self.dim == 3:
            self.ext_z = kwargs.get('ext_z', 0.0)
            self.zmin = kwargs.get('zmin', np.min(self.ele_pos[:, 2]))
            self.zmax = kwargs.get('zmax', np.max(self.ele_pos[:, 2]))
            self.gdz = kwargs.get('gdz', 0.01*(self.zmax - self.zmin))
        return
        
    def estimate_at(self):
        pass

    def place_basis(self):
        pass

    def create_src_dist_tables(self):
        pass

    def method(self):
        """Actual sequence of methods called for KCSD
        Defines:
        self.k_pot and self.k_interp_cross matrices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.create_lookup()                                #Look up table 
        self.update_b_pot()                                 #update kernel
        self.update_b_src()                                 #update crskernel
        self.update_b_interp_pot()                          #update pot interp
        return

    def create_lookup(self, dist_table_density=20):
        """Creates a table for easy potential estimation from CSD.
        Updates and Returns the potentials due to a 
        given basis source like a lookup
        table whose shape=(dist_table_density,)--> set in KCSD2D_Helpers.py

        Parameters
        ----------
        dist_table_density : int
            number of distance values at which potentials are computed.
            Default 100

        Returns
        -------
        None
        """
        xs = np.logspace(0., np.log10(self.dist_max+1.), dist_table_density)
        xs = xs - 1.0 #starting from 0
        dist_table = np.zeros(len(xs))
        for i, pos in enumerate(xs):
            dist_table[i] = self.forward_model(pos, 
                                               self.R, 
                                               self.h, 
                                               self.sigma, 
                                               self.basis)
        self.interpolate_pot_at = interpolate.interp1d(xs, dist_table, kind='cubic')
        return

    def update_b_pot(self):
        """Updates the b_pot  - array is (#_basis_sources, #_electrodes)
        Updates the  k_pot - array is (#_electrodes, #_electrodes) K(x,x') 
        Eq9,Jan2012
        Calculates b_pot - matrix containing the values of all
        the potential basis functions in all the electrode positions
        (essential for calculating the cross_matrix).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.b_pot = self.interpolate_pot_at(self.src_ele_dists)
        self.k_pot = np.dot(self.b_pot.T, self.b_pot) #K(x,x') Eq9,Jan2012
        self.k_pot /= self.n_src
        return
        
    def update_b_src(self):
        """Updates the b_src in the shape of (#_est_pts, #_basis_sources)
        Updates the k_interp_cross - K_t(x,y) Eq17
        Calculate b_src - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.b_src = self.basis(self.src_estm_dists, self.R).T
        self.k_interp_cross = np.dot(self.b_src, self.b_pot) #K_t(x,y) Eq17
        self.k_interp_cross /= self.n_src
        return
        
    def update_b_interp_pot(self):
        """Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        Updates b_interp_pot
        Updates k_interp_pot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.b_interp_pot = self.interpolate_pot_at(self.src_estm_dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        self.k_interp_pot /= self.n_src
        return
    
    def values(self, estimate='CSD'):
        """Computes the values of the quantity of interest

        Parameters
        ----------
        estimate : 'CSD' or 'POT'
            What quantity is to be estimated
            Defaults to 'CSD'

        Returns
        -------
        estimated quantity of shape (ngx, ngy, ngz, nt)
        """
        if estimate == 'CSD': #Maybe used for estimating the potentials also.
            estimation_table = self.k_interp_cross 
        elif estimate == 'POT':
            estimation_table = self.k_interp_pot
        else:
            print 'Invalid quantity to be measured, pass either CSD or POT'
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        estimation = np.zeros((self.n_estm, self.n_time))
        for t in xrange(self.n_time):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in xrange(self.n_ele):
                estimation[:, t] += estimation_table[:, i] *beta[i] # C*(x) Eq 18
        return self.process_estimate(estimation)

    def process_estimate(self, estimation):
        if self.dim == 1:
            estimation = estimation.reshape(self.ngx, self.n_time)
        elif self.dim == 2:
            estimation = estimation.reshape(self.ngx, self.ngy, self.n_time)
        elif self.dim == 3:
            estimation = estimation.reshape(self.ngx, self.ngy, self.ngz, self.n_time)
        return estimation

    def update_R(self, R):
        """Used in Cross validation

        Parameters
        ----------
        R : float

        Returns
        -------
        None
        """
        self.R = R
        self.dist_max = max(np.max(self.src_ele_dists), 
                            np.max(self.src_estm_dists)) + self.R
        self.method()
        return

    def update_lambda(self, lambd):
        """Used in Cross validation

        Parameters
        ----------
        lambd : float

        Returns
        -------
        None
        """
        self.lambd = lambd
        return

    def cross_validate(self, lambdas=None, Rs=None): 
        """Method defines the cross validation.
        By default only cross_validates over lambda, 
        When no argument is passed, it takes
        lambdas = np.logspace(-2,-25,25,base=10.)
        and Rs = np.array(self.R).flatten()
        otherwise pass necessary numpy arrays

        Parameters
        ----------
        lambdas : numpy array
        Rs : numpy array

        Returns
        -------
        R : post cross validation
        Lambda : post cross validation
        """
        if lambdas is None:                           #when None
            print 'No lambda given, using defaults'
            lambdas = np.logspace(-2,-25,25,base=10.) #Default multiple lambda
            lambdas = np.hstack((lambdas, np.array((0.0))))
        elif lambdas.size == 1:                       #resize when one entry
            lambdas = lambdas.flatten()
        if Rs is None:                                #when None
            Rs = np.array((self.R)).flatten()         #Default over one R value
        errs = np.zeros((Rs.size, lambdas.size))
        index_generator = []                          
        for ii in range(self.n_ele):
            idx_test = [ii]                           
            idx_train = range(self.n_ele)
            idx_train.remove(ii)                      #Leave one out
            index_generator.append((idx_train, idx_test))
        for R_idx,R in enumerate(Rs):                 #Iterate over R
            self.update_R(R)
            print 'Cross validating R (all lambda) :', R
            for lambd_idx,lambd in enumerate(lambdas): #Iterate over lambdas
                errs[R_idx, lambd_idx] = self.compute_cverror(lambd, 
                                                              index_generator)
        err_idx = np.where(errs==np.min(errs))         #Index of the least error
        cv_R = Rs[err_idx[0]][0]      #First occurance of the least error's
        cv_lambda = lambdas[err_idx[1]][0]
        self.cv_error = np.min(errs)  #otherwise is None
        self.update_R(cv_R)           #Update solver
        self.update_lambda(cv_lambda)
        print 'R, lambda :', cv_R, cv_lambda
        return cv_R, cv_lambda

    def compute_cverror(self, lambd, index_generator):
        """Useful for Cross validation error calculations

        Parameters
        ----------
        lambd : float
        index_generator : list

        Returns
        -------
        err : float
            the sum of the error computed.
        """
        err = 0
        for idx_train, idx_test in index_generator:
            B_train = self.k_pot[np.ix_(idx_train, idx_train)]
            V_train = self.pots[idx_train]
            V_test = self.pots[idx_test]
            I_matrix = np.identity(len(idx_train))
            B_new = np.matrix(B_train) + (lambd*I_matrix)
            try:
                beta_new = np.dot(np.matrix(B_new).I, np.matrix(V_train))
                B_test = self.k_pot[np.ix_(idx_test, idx_train)]
                V_est = np.zeros((len(idx_test), self.pots.shape[1]))
                for ii in range(len(idx_train)):
                    for tt in range(self.pots.shape[1]):
                        V_est[:, tt] += beta_new[ii, tt] * B_test[:, ii]
                err += np.linalg.norm(V_est-V_test)
            except LinAlgError:
                print 'Encoutered Singular Matrix Error: try changing ele_pos'
                #err = 10000. #singluar matrix errors!
                raise
        return err

if __name__ == '__main__':
    print 'Invalid usage, use this an inheritable class only'
