"""
This script is used to generate Current Source Density Estimates, 
using the kCSD method Jan et.al (2012) for 1D case

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Michal Czerwinski, Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
import numpy as np
from scipy import integrate, interpolate
from scipy.spatial import distance

from KCSD import KCSD
import utility_functions as utils
import basis_functions as basis

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
                space conductance of the medium
                Defaults to 1.
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

        Returns
        -------
        None

        Raises
        ------
        LinAlgException
            If the matrix is not numerically invertible.
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        super(KCSD1D, self).__init__(ele_pos, pots, **kwargs)
        return

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:         
        self.n_estm = self.estm_x.size
        self.ngx = self.estm_x.shape
        self.estm_x : Locations at which CSD is requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        nx = (self.xmax - self.xmin)/self.gdx
        self.estm_x = np.mgrid[self.xmin:self.xmax:np.complex(0,nx)]
        self.n_estm = self.estm_x.size
        self.ngx = self.estm_x.shape[0]
        return

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

        Returns
        -------
        None
        """
        # check If valid basis source type passed:
        source_type = self.src_type
        try:
            self.basis = basis.basis_1D[source_type]
        except:
            print 'Invalid source_type for basis! available are:', basis.basis_1D.keys()
            raise KeyError
        #Mesh where the source basis are placed is at self.src_x
        (self.src_x, self.R) = utils.distribute_srcs_1D(self.estm_x,
                                                        self.n_src_init,
                                                        self.ext_x,
                                                        self.R_init )
        self.n_src = self.src_x.size
        self.nsx = self.src_x.shape
        return

    def create_src_dist_tables(self):
        """Creates distance tables between sources, electrode and estm points

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        src_loc = np.array((self.src_x.ravel()))
        src_loc = src_loc.reshape((len(src_loc), 1))
        est_loc = np.array((self.estm_x.ravel()))
        est_loc = est_loc.reshape((len(est_loc), 1))
        self.src_ele_dists = distance.cdist(src_loc, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc, est_loc,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R
        return

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
        #1/2sigma normalization not here
        m = np.sqrt((x-xp)**2 + h**2) - abs(x-xp)
        m *= basis_func(abs(xp), R)  #xp is the distance
        return m


if __name__ == '__main__':
    ele_pos = np.array(([-0.1],[0], [0.5], [1.], [1.4], [2.], [2.3]))
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
    k = KCSD1D(ele_pos, pots,
               gdx=0.01, n_src_init=300,
               ext_x=0.0, src_type='gauss')
    #k.cross_validate(lambdas=np.array((0.0)), Rs=np.array([0.21, 0.23, 0.24]))
    #k.cross_validate()
    #print k.values()
    #k.cross_validate(Rs=np.array(0.14).reshape(1))
    #k.cross_validate(Rs=np.array((0.01,0.02,0.04))) 
    #print k.values()
