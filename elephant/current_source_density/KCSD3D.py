"""
This script is used to generate Current Source Density Estimates, 
using the kCSD method Jan et.al (2012) for 3D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
import numpy as np
from scipy.spatial import distance
from scipy import special, interpolate, integrate    
### Dependencies of elephant!
# try:
#     from skmonaco import mcmiser
#     skmonaco_available = True
#     import multiprocessing
#     num_cores = multiprocessing.cpu_count()
# except ImportError:
#     skmonaco_available = False
skmonaco_available = False
    
from KCSD import KCSD
import utility_functions as utils
import basis_functions as basis
    
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
                space conductance of the medium
                Defaults to 1.
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

        Returns
        -------
        None

        Raises
        ------
        LinAlgError 
            Could not invert the matrix, try changing the ele_pos slightly
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        super(KCSD3D, self).__init__(ele_pos, pots, **kwargs)
        return

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:         
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape
        self.estm_x, self.estm_y, self.estm_z : Pts. at which CSD is requested

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #Number of points where estimation is to be made.
        nx = (self.xmax - self.xmin)/self.gdx
        ny = (self.ymax - self.ymin)/self.gdy
        nz = (self.zmax - self.zmin)/self.gdz
        #Making a mesh of points where estimation is to be made.
        self.estm_x, self.estm_y, self.estm_z = np.mgrid[self.xmin:self.xmax:np.complex(0,nx), 
                                                         self.ymin:self.ymax:np.complex(0,ny),
                                                         self.zmin:self.zmax:np.complex(0,nz)]
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape
        return

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

        Returns
        -------
        None
        """
        #If Valid basis source type passed?
        source_type = self.src_type
        try:
            self.basis = basis.basis_3D[source_type]
        except:
            print 'Invalid source_type for basis! available are:', basis.basis_3D.keys()
            raise KeyError
        #Mesh where the source basis are placed is at self.src_x 
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
        src_loc = np.array((self.src_x.ravel(), 
                            self.src_y.ravel(), 
                            self.src_z.ravel()))
        est_loc = np.array((self.estm_x.ravel(), 
                            self.estm_y.ravel(), 
                            self.estm_z.ravel()))
        self.src_ele_dists = distance.cdist(src_loc.T, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc.T, est_loc.T,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R
        return

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
                #4*pi*((1/a)*(integrate(r**2 * exp(-r**2 / (2*d**2)) *dr ) between 0 and a ) + 
                #(integrate(r *exp(-r**2 / (2*d**2)) * dr) between a and 3*d))
                e = np.exp(-(x/ (np.sqrt(2)*d))**2)
                erf = special.erf(x / (np.sqrt(2)*d))
                pot = 4* np.pi * ( (d**2)*(e - np.exp(-4.5)) +
                                   (1/x)*((np.sqrt(np.pi/2)*(d**3)*erf) - x*(d**2)*e))
            else:
                #4*pi*integrate((r**2)*exp(-(r**2 / (2*d**2)))*dr) between 0 and 3*d
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
    ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                        (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
                        (0.5, 0.5, 0.5)])
    pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
    params = {}
    k = KCSD3D(ele_pos, pots,
               gdx=0.02, gdy=0.02, gdz=0.02,
               n_src_init=1000, src_type='gauss_lim')
    k.cross_validate()
    #k.cross_validate(Rs=np.array(0.14).reshape(1))
    #k.cross_validate(Rs=np.array((0.01,0.02,0.04))) 
