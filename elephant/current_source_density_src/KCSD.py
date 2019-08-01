#!/usr/bin/env python
"""This script is used to generate Current Source Density Estimates, using the
kCSD method Jan et.al (2012).
This was written by :
[1]Chaitanya Chintaluri,
[2]Michal Czerwinski,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
KCSD1D[1][2], KCSD2D[1], KCSD3D[1], MoIKCSD[1]
"""
from __future__ import division

import numpy as np
from scipy import special, integrate, interpolate
from scipy.spatial import distance
from numpy.linalg import LinAlgError

from . import utility_functions as utils
from . import basis_functions as basis

skmonaco_available = False


class CSD(object):
    """CSD - The base class for KCSD methods."""
    def __init__(self, ele_pos, pots):
        self.validate(ele_pos, pots)
        self.ele_pos = ele_pos
        self.pots = pots
        self.n_ele = self.ele_pos.shape[0]
        self.n_time = self.pots.shape[1]
        self.dim = self.ele_pos.shape[1]
        self.cv_error = None

    def validate(self, ele_pos, pots):
        """Basic checks to see if inputs are okay

        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        """
        if ele_pos.shape[0] != pots.shape[0]:
            raise Exception("Number of measured potentials is not equal "
                            "to electrode number!")
        if ele_pos.shape[0] < 1+ele_pos.shape[1]: #Dim+1
            raise Exception("Number of electrodes must be at least :",
                            1+ele_pos.shape[1])
        if utils.contains_duplicated_electrodes(ele_pos):
            raise Exception("Error! Duplicated electrode!")

    def sanity(self, true_csd, pos_csd):
        """Useful for comparing TrueCSD with reconstructed CSD. Computes, the RMS error
        between the true_csd and the reconstructed csd at pos_csd using the
        method defined.
        Parameters
        ----------
        true_csd : csd values used to generate potentials
        pos_csd : csd estimatation from the method
        Returns
        -------
        RMSE : root mean squared difference
        """
        csd = self.values(pos_csd)
        RMSE = np.sqrt(np.mean(np.square(true_csd - csd)))
        return RMSE

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

    def parameters(self, **kwargs):
        """Defining the default values of the method passed as kwargs
        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class
        """
        self.src_type = kwargs.pop('src_type', 'gauss')
        self.sigma = kwargs.pop('sigma', 1.0)
        self.h = kwargs.pop('h', 1.0)
        self.n_src_init = kwargs.pop('n_src_init', 1000)
        self.lambd = kwargs.pop('lambd', 0.0)
        self.R_init = kwargs.pop('R_init', 0.23)
        self.ext_x = kwargs.pop('ext_x', 0.0)
        self.xmin = kwargs.pop('xmin', np.min(self.ele_pos[:, 0]))
        self.xmax = kwargs.pop('xmax', np.max(self.ele_pos[:, 0]))
        self.gdx = kwargs.pop('gdx', 0.01*(self.xmax - self.xmin))
        if self.dim >= 2:
            self.ext_y = kwargs.pop('ext_y', 0.0)
            self.ymin = kwargs.pop('ymin', np.min(self.ele_pos[:, 1]))
            self.ymax = kwargs.pop('ymax', np.max(self.ele_pos[:, 1]))
            self.gdy = kwargs.pop('gdy', 0.01*(self.ymax - self.ymin))
        if self.dim == 3:
            self.ext_z = kwargs.pop('ext_z', 0.0)
            self.zmin = kwargs.pop('zmin', np.min(self.ele_pos[:, 2]))
            self.zmax = kwargs.pop('zmax', np.max(self.ele_pos[:, 2]))
            self.gdz = kwargs.pop('gdz', 0.01*(self.zmax - self.zmin))
        if kwargs:
            raise TypeError('Invalid keyword arguments:', kwargs.keys())

    def method(self):
        """Actual sequence of methods called for KCSD
        Defines:
        self.k_pot and self.k_interp_cross matrices
        Parameters
        ----------
        None
        """
        self.create_lookup()                                #Look up table
        self.update_b_pot()                                 #update kernel
        self.update_b_src()                                 #update crskernel
        self.update_b_interp_pot()                          #update pot interp

    def create_lookup(self, dist_table_density=20):
        """Creates a table for easy potential estimation from CSD.
        Updates and Returns the potentials due to a
        given basis source like a lookup
        table whose shape=(dist_table_density,)
        Parameters
        ----------
        dist_table_density : int
            number of distance values at which potentials are computed.
            Default 100
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
        """
        self.b_pot = self.interpolate_pot_at(self.src_ele_dists)
        self.k_pot = np.dot(self.b_pot.T, self.b_pot) #K(x,x') Eq9,Jan2012
        self.k_pot /= self.n_src

    def update_b_src(self):
        """Updates the b_src in the shape of (#_est_pts, #_basis_sources)
        Updates the k_interp_cross - K_t(x,y) Eq17
        Calculate b_src - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)
        Parameters
        ----------
        None
        """
        self.b_src = self.basis(self.src_estm_dists, self.R).T
        self.k_interp_cross = np.dot(self.b_src, self.b_pot) #K_t(x,y) Eq17
        self.k_interp_cross /= self.n_src

    def update_b_interp_pot(self):
        """Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        Updates b_interp_pot
        Updates k_interp_pot
        Parameters
        ----------
        None
        """
        self.b_interp_pot = self.interpolate_pot_at(self.src_estm_dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        self.k_interp_pot /= self.n_src

    def values(self, estimate='CSD'):
        """Computes the values of the quantity of interest
        Parameters
        ----------
        estimate : 'CSD' or 'POT'
            What quantity is to be estimated
            Defaults to 'CSD'
        Returns
        -------
        estimation : np.array
            estimated quantity of shape (ngx, ngy, ngz, nt)
        """
        if estimate == 'CSD': #Maybe used for estimating the potentials also.
            estimation_table = self.k_interp_cross
        elif estimate == 'POT':
            estimation_table = self.k_interp_pot
        else:
            print('Invalid quantity to be measured, pass either CSD or POT')
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        estimation = np.zeros((self.n_estm, self.n_time))
        for t in range(self.n_time):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in range(self.n_ele):
                estimation[:, t] += estimation_table[:, i] *beta[i] # C*(x) Eq 18
        return self.process_estimate(estimation)

    def process_estimate(self, estimation):
        """Function used to rearrange estimation according to dimension, to be
        used by the fuctions values
        Parameters
        ----------
        estimation : np.array
        Returns
        -------
        estimation : np.array
            estimated quantity of shape (ngx, ngy, ngz, nt)
        """
        if self.dim == 1:
            estimation = estimation.reshape(self.ngx, self.n_time)
        elif self.dim == 2:
            estimation = estimation.reshape(self.ngx, self.ngy, self.n_time)
        elif self.dim == 3:
            estimation = estimation.reshape(self.ngx, self.ngy, self.ngz, self.n_time)
        return estimation

    def update_R(self, R):
        """Update the width of the basis fuction - Used in Cross validation
        Parameters
        ----------
        R : float
        """
        self.R = R
        self.dist_max = max(np.max(self.src_ele_dists),
                            np.max(self.src_estm_dists)) + self.R
        self.method()

    def update_lambda(self, lambd):
        """Update the lambda parameter of regularization, Used in Cross validation
        Parameters
        ----------
        lambd : float
        """
        self.lambd = lambd

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
            print('No lambda given, using defaults')
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
            idx_train = list(range(self.n_ele))
            idx_train.remove(ii)                      #Leave one out
            index_generator.append((idx_train, idx_test))
        for R_idx,R in enumerate(Rs):                 #Iterate over R
            self.update_R(R)
            print('Cross validating R (all lambda) :', R)
            for lambd_idx,lambd in enumerate(lambdas): #Iterate over lambdas
                errs[R_idx, lambd_idx] = self.compute_cverror(lambd,
                                                              index_generator)
        err_idx = np.where(errs==np.min(errs))         #Index of the least error
        cv_R = Rs[err_idx[0]][0]      #First occurance of the least error's
        cv_lambda = lambdas[err_idx[1]][0]
        self.cv_error = np.min(errs)  #otherwise is None
        self.update_R(cv_R)           #Update solver
        self.update_lambda(cv_lambda)
        print('R, lambda :', cv_R, cv_lambda)
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
                raise LinAlgError('Encoutered Singular Matrix Error: try changing ele_pos slightly')
        return err

class KCSD1D(KCSD):
    """KCSD1D - The 1D variant for the Kernel Current Source Density method.
    This estimates the Current Source Density, for a given configuration of
    electrod positions and recorded potentials, in the case of 1D recording
    electrodes (laminar probes). The method implented here is based on the
    original paper by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize KCSD1D Class.
        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the tissue in S/m
                Defaults to 1 S/m
            n_src_init : int
                requested number of sources
                Defaults to 300
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed cylindrical slice
                Defaults to 1.
            xmin, xmax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
            ext_x : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                Defaults to 0.
            gdx : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.
        Raises
        ------
        LinAlgException
            If the matrix is not numerically invertible.
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        super(KCSD1D, self).__init__(ele_pos, pots, **kwargs)

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:
        self.n_estm = self.estm_x.size
        self.ngx = self.estm_x.shape
        self.estm_x : Locations at which CSD is requested.
        Parameters
        ----------
        None
        """
        nx = (self.xmax - self.xmin)/self.gdx
        self.estm_x = np.mgrid[self.xmin:self.xmax:np.complex(0,nx)]
        self.n_estm = self.estm_x.size
        self.ngx = self.estm_x.shape[0]

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources,
        Defines
        source_type : basis_fuctions.basis_1D.keys()
        self.R based on R_init
        self.dist_max as maximum distance between electrode and basis
        self.nsx = self.src_x.shape
        self.src_x : Locations at which basis sources are placed.
        Parameters
        ----------
        None
        """
        source_type = self.src_type
        try:
            self.basis = basis.basis_1D[source_type]
        except KeyError:
            raise KeyError('Invalid source_type for basis! available are:',
                           basis.basis_1D.keys())
        (self.src_x, self.R) = utils.distribute_srcs_1D(self.estm_x,
                                                        self.n_src_init,
                                                        self.ext_x,
                                                        self.R_init )
        self.n_src = self.src_x.size
        self.nsx = self.src_x.shape

    def create_src_dist_tables(self):
        """Creates distance tables between sources, electrode and estm points
        Parameters
        ----------
        None
        """
        src_loc = np.array((self.src_x.ravel()))
        src_loc = src_loc.reshape((len(src_loc), 1))
        est_loc = np.array((self.estm_x.ravel()))
        est_loc = est_loc.reshape((len(est_loc), 1))
        self.src_ele_dists = distance.cdist(src_loc, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc, est_loc,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R

    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Eq 26 kCSD by Jan,2012
        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_1D.key
        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        pot, err = integrate.quad(self.int_pot_1D,
                                  -R, R,
                                  args=(x, R, h, src_type))
        pot *= 1./(2.0*sigma)
        return pot

    def int_pot_1D(self, xp, x, R, h, basis_func):
        """FWD model function.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        Eq 26 kCSD by Jan,2012

        Parameters
        ----------
        xp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source

        Returns
        -------
        pot : float
        """
        m = np.sqrt((x-xp)**2 + h**2) - abs(x-xp)
        m *= basis_func(abs(xp), R)  #xp is the distance
        return m

class KCSD2D(KCSD):
    """KCSD2D - The 2D variant for the Kernel Current Source Density method.
    This estimates the Current Source Density, for a given configuration of
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize KCSD2D Class.
        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the tissue in S/m
                Defaults to 1 S/m
            n_src_init : int
                requested number of sources
                Defaults to 1000
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed tissue slice
                Defaults to 1.
            xmin, xmax, ymin, ymax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
                Defaults to min(ele_pos(y)), and max(ele_pos(y))
            ext_x, ext_y : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                length of space extension: y_min-ext_y ... y_max+ext_y
                Defaults to 0.
            gdx, gdy : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
                Defaults to 0.01(ymax-ymin)
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.
        Raises
        ------
        LinAlgError
            Could not invert the matrix, try changing the ele_pos slightly
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        super(KCSD2D, self).__init__(ele_pos, pots, **kwargs)

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy = self.estm_x.shape
        self.estm_x, self.estm_y : Locations at which CSD is requested.
        Parameters
        ----------
        None
        """
        nx = (self.xmax - self.xmin)/self.gdx
        ny = (self.ymax - self.ymin)/self.gdy
        self.estm_x, self.estm_y = np.mgrid[self.xmin:self.xmax:np.complex(0,nx),
                                            self.ymin:self.ymax:np.complex(0,ny)]
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy = self.estm_x.shape

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources,
        Defines
        source_type : basis_fuctions.basis_2D.keys()
        self.R based on R_init
        self.dist_max as maximum distance between electrode and basis
        self.nsx, self.nsy = self.src_x.shape
        self.src_x, self.src_y : Locations at which basis sources are placed.
        Parameters
        ----------
        None
        """
        source_type = self.src_type
        try:
            self.basis = basis.basis_2D[source_type]
        except KeyError:
            raise KeyError('Invalid source_type for basis! available are:',
                           basis.basis_2D.keys())
        (self.src_x, self.src_y, self.R) = utils.distribute_srcs_2D(self.estm_x,
                                                                    self.estm_y,
                                                                    self.n_src_init,
                                                                    self.ext_x,
                                                                    self.ext_y,
                                                                    self.R_init )
        self.n_src = self.src_x.size
        self.nsx, self.nsy = self.src_x.shape

    def create_src_dist_tables(self):
        """Creates distance tables between sources, electrode and estm points
        Parameters
        ----------
        None
        """
        src_loc = np.array((self.src_x.ravel(), self.src_y.ravel()))
        est_loc = np.array((self.estm_x.ravel(), self.estm_y.ravel()))
        self.src_ele_dists = distance.cdist(src_loc.T, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc.T, est_loc.T,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R

    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Eq 22 kCSD by Jan,2012
        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_2D.key
        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        pot, err = integrate.dblquad(self.int_pot_2D,
                                     -R, R,
                                     lambda x: -R,
                                     lambda x: R,
                                     args=(x, R, h, src_type))
        pot *= 1./(2.0*np.pi*sigma)  #Potential basis functions bi_x_y
        return pot

    def int_pot_2D(self, xp, yp, x, R, h, basis_func):
        """FWD model function.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        Parameters
        ----------
        xp, yp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source
        Returns
        -------
        pot : float
        """
        y = ((x-xp)**2 + yp**2)**(0.5)
        if y < 0.00001:
            y = 0.00001
        dist = np.sqrt(xp**2 + yp**2)
        pot = np.arcsinh(h/y)*basis_func(dist, R)
        return pot

class MoIKCSD(KCSD2D):
    """MoIKCSD - CSD while including the forward modeling effects of saline.

    This estimates the Current Source Density, for a given configuration of
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes from an MEA electrode plane using the Method of Images.
    The method implented here is based on kCSD method by Jan Potworowski
    et.al. 2012, which was extended in Ness, Chintaluri 2015 for MEA.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize MoIKCSD Class.
        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the tissue in S/m
                Defaults to 1 S/m
            sigma_S : float
                conductance of the saline (medium) in S/m
                Default is 5 S/m (5 times more conductive)
            n_src_init : int
                requested number of sources
                Defaults to 1000
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed tissue slice
                Defaults to 1.
            xmin, xmax, ymin, ymax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
                Defaults to min(ele_pos(y)), and max(ele_pos(y))
            ext_x, ext_y : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                length of space extension: y_min-ext_y ... y_max+ext_y
                Defaults to 0.
            gdx, gdy : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
                Defaults to 0.01(ymax-ymin)
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.
            MoI_iters : int
                Number of interations in method of images.
                Default is 20
        """
        self.MoI_iters = kwargs.pop('MoI_iters', 20)
        self.sigma_S = kwargs.pop('sigma_S', 5.0)
        self.sigma = kwargs.pop('sigma', 1.0)
        W_TS = (self.sigma - self.sigma_S) / (self.sigma + self.sigma_S)
        self.iters = np.arange(self.MoI_iters) + 1  #Eq 6, Ness (2015)
        self.iter_factor = W_TS**self.iters
        super(MoIKCSD, self).__init__(ele_pos, pots, **kwargs)

    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Eq 22 kCSD by Jan,2012
        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_2D.key
        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        pot, err = integrate.dblquad(self.int_pot_2D_moi, -R, R,
                                     lambda x: -R,
                                     lambda x: R,
                                     args=(x, R, h, src_type))
        pot *= 1./(2.0*np.pi*sigma)
        return pot

    def int_pot_2D_moi(self, xp, yp, x, R, h, basis_func):
        """FWD model function. Incorporates the Method of Images.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        #Eq 20, Ness(2015)
        Parameters
        ----------
        xp, yp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source
        Returns
        -------
        pot : float
        """
        L = ((x-xp)**2 + yp**2)**(0.5)
        if L < 0.00001:
            L = 0.00001
        correction = np.arcsinh((h-(2*h*self.iters))/L) + np.arcsinh((h+(2*h*self.iters))/L)
        pot = np.arcsinh(h/L) + np.sum(self.iter_factor*correction)
        dist = np.sqrt(xp**2 + yp**2)
        pot *= basis_func(dist, R) #Eq 20, Ness et.al.
        return pot

class KCSD3D(KCSD):
    """KCSD3D - The 3D variant for the Kernel Current Source Density method.
    This estimates the Current Source Density, for a given configuration of
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize KCSD3D Class.
        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the tissue in S/m
                Defaults to 1 S/m
            n_src_init : int
                requested number of sources
                Defaults to 1000
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed tissue slice
                Defaults to 1.
            xmin, xmax, ymin, ymax, zmin, zmax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
                Defaults to min(ele_pos(y)), and max(ele_pos(y))
                Defaults to min(ele_pos(z)), and max(ele_pos(z))
            ext_x, ext_y, ext_z : float
                length of space extension: xmin-ext_x ... xmax+ext_x
                length of space extension: ymin-ext_y ... ymax+ext_y
                length of space extension: zmin-ext_z ... zmax+ext_z
                Defaults to 0.
            gdx, gdy, gdz : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
                Defaults to 0.01(ymax-ymin)
                Defaults to 0.01(zmax-zmin)
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.
        Raises
        ------
        LinAlgError
            Could not invert the matrix, try changing the ele_pos slightly
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        super(KCSD3D, self).__init__(ele_pos, pots, **kwargs)

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape
        self.estm_x, self.estm_y, self.estm_z : Pts. at which CSD is requested
        Parameters
        ----------
        None
        """
        nx = (self.xmax - self.xmin)/self.gdx
        ny = (self.ymax - self.ymin)/self.gdy
        nz = (self.zmax - self.zmin)/self.gdz
        self.estm_x, self.estm_y, self.estm_z = np.mgrid[self.xmin:self.xmax:np.complex(0,nx),
                                                         self.ymin:self.ymax:np.complex(0,ny),
                                                         self.zmin:self.zmax:np.complex(0,nz)]
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources,
        Defines
        source_type : basis_fuctions.basis_2D.keys()
        self.R based on R_init
        self.dist_max as maximum distance between electrode and basis
        self.nsx, self.nsy, self.nsz = self.src_x.shape
        self.src_x, self.src_y, self.src_z : Locations at which basis sources are placed.
        Parameters
        ----------
        None
        """
        source_type = self.src_type
        try:
            self.basis = basis.basis_3D[source_type]
        except KeyError:
            raise KeyError('Invalid source_type for basis! available are:',
                           basis.basis_3D.keys())
        (self.src_x, self.src_y, self.src_z, self.R) = utils.distribute_srcs_3D(self.estm_x,
                                                                                self.estm_y,
                                                                                self.estm_z,
                                                                                self.n_src_init,
                                                                                self.ext_x,
                                                                                self.ext_y,
                                                                                self.ext_z,
                                                                                self.R_init)

        self.n_src = self.src_x.size
        self.nsx, self.nsy, self.nsz = self.src_x.shape

    def create_src_dist_tables(self):
        """Creates distance tables between sources, electrode and estm points
        Parameters
        ----------
        None
        """
        src_loc = np.array((self.src_x.ravel(),
                            self.src_y.ravel(),
                            self.src_z.ravel()))
        est_loc = np.array((self.estm_x.ravel(),
                            self.estm_y.ravel(),
                            self.estm_z.ravel()))
        self.src_ele_dists = distance.cdist(src_loc.T, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc.T, est_loc.T,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R

    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Utlizies sk monaco monte carlo method if available, otherwise defaults
        to scipy integrate
        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_3D.key
        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        if src_type.__name__ == "gauss_3D":
            if x == 0: x=0.0001
            pot = special.erf(x/(np.sqrt(2)*R/3.0)) / x
        elif src_type.__name__ == "gauss_lim_3D":
            if x == 0: x=0.0001
            d = R/3.
            if x < R:
                e = np.exp(-(x/ (np.sqrt(2)*d))**2)
                erf = special.erf(x / (np.sqrt(2)*d))
                pot = 4* np.pi * ( (d**2)*(e - np.exp(-4.5)) +
                                   (1/x)*((np.sqrt(np.pi/2)*(d**3)*erf) - x*(d**2)*e))
            else:
                pot = 15.28828*(d)**3 / x
            pot /= (np.sqrt(2*np.pi)*d)**3
        elif src_type.__name__ == "step_3D":
            Q = 4.*np.pi*(R**3)/3.
            if x < R:
                pot = (Q * (3 - (x/R)**2)) / (2.*R)
            else:
                pot = Q / x
            pot *= 3/(4*np.pi*R**3)
        else:
            if skmonaco_available:
                pot, err = mcmiser(self.int_pot_3D_mc,
                                   npoints=1e5,
                                   xl=[-R, -R, -R],
                                   xu=[R, R, R],
                                   seed=42,
                                   nprocs=num_cores,
                                   args=(x, R, h, src_type))
            else:
                pot, err = integrate.tplquad(self.int_pot_3D,
                                             -R,
                                             R,
                                             lambda x: -R,
                                             lambda x: R,
                                             lambda x, y: -R,
                                             lambda x, y: R,
                                             args=(x, R, h, src_type))
        pot *= 1./(4.0*np.pi*sigma)
        return pot

    def int_pot_3D(self, xp, yp, zp, x, R, h, basis_func):
        """FWD model function.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        Parameters
        ----------
        xp, yp, zp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source
        Returns
        -------
        pot : float
        """
        y = ((x-xp)**2 + yp**2 + zp**2)**0.5
        if y < 0.00001:
            y = 0.00001
        dist = np.sqrt(xp**2 + yp**2 + zp**2)
        pot = 1.0/y
        pot *= basis_func(dist, R)
        return pot

    def int_pot_3D_mc(self, xyz, x, R, h, basis_func):
        """
        The same as int_pot_3D, just different input: x,y,z <-- xyz (tuple)
        FWD model function, using Monte Carlo Method of integration
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        Parameters
        ----------
        xp, yp, zp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source
        Returns
        -------
        pot : float
        """
        xp, yp, zp = xyz
        return self.int_pot_3D(xp, yp, zp, x, R, h, basis_func)

if __name__ == '__main__':
    print('Checking 1D')
    ele_pos = np.array(([-0.1],[0], [0.5], [1.], [1.4], [2.], [2.3]))
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
    k = KCSD1D(ele_pos, pots,
               gdx=0.01, n_src_init=300,
               ext_x=0.0, src_type='gauss')
    k.cross_validate()
    print(k.values())

    print('Checking 2D')
    ele_pos = np.array([[-0.2, -0.2],[0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5],
                        [1.2, 1.2]])
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
    k = KCSD2D(ele_pos, pots,
               gdx=0.05, gdy=0.05,
               xmin=-2.0, xmax=2.0,
               ymin=-2.0, ymax=2.0,
               src_type='gauss')
    k.cross_validate()
    print(k.values())

    print('Checking MoIKCSD')
    k = MoIKCSD(ele_pos, pots,
                gdx=0.05, gdy=0.05,
                xmin=-2.0, xmax=2.0,
                ymin=-2.0, ymax= 2.0)
    k.cross_validate()

    print('Checking KCSD3D')
    ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                        (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
                        (0.5, 0.5, 0.5)])
    pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
    k = KCSD3D(ele_pos, pots,
               gdx=0.02, gdy=0.02, gdz=0.02,
               n_src_init=1000, src_type='gauss_lim')
    k.cross_validate()
    print(k.values())
