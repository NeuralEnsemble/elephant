"""
This script is used to generate basis sources for the 
kCSD method Jan et.al (2012) for 3D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Michal Czerwinski, Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
from __future__ import division

import numpy as np

def gauss(d, stdev, dim):
    """Gaussian function

    Parameters
    ----------
    d : floats or np.arrays
        Distance array to the point of evaluation 
    stdev : float
        cutoff range
    dim : int
        dimension of the gaussian function

    Returns
    -------
    Z : floats or np.arrays
        function evaluated
    """
    Z = np.exp(-(d**2) / (2* stdev**2) ) / (np.sqrt(2*np.pi)*stdev)**dim
    return Z

def step_1D(d, R):
    """Returns normalized 1D step function.

    Parameters
    ----------
    d : floats or np.arrays
        Distance array to the point of evaluation 
    R : float
        cutoff range

    Returns
    -------
    s : Value of the function (d  <= R) / R
    """
    s = (d  <= R)
    s = s / R #normalize with width
    return s        

def gauss_1D(d, three_stdev):
    """Returns normalized gaussian 2D scale function

    Parameters
    ----------
    d : floats or np.arrays
        Distance array to the point of evaluation
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : (three_std/3)*(1/2*pi)*(exp(-0.5)*stddev**(-2) *(d**2))
    """
    stdev = three_stdev/3.0
    Z = gauss(d, stdev, 1)
    return Z

def gauss_lim_1D(d, three_stdev):
    """Returns gausian 2D function cut off after 3 standard deviations.

    Parameters
    ----------
    d : floats or np.arrays
        Distance array to the point of evaluation
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : (three_std/3)*(1/2*pi)*(exp(-0.5)*stddev**(-2) *((x-mu)**2)), 
        cut off = three_stdev
    """
    Z = gauss_1D(d, three_stdev)
    Z *= (d < three_stdev)
    return Z

def step_2D(d, R):
    """Returns normalized 2D step function.

    Parameters
    ----------
    d : float or np.arrays
        Distance array to the point of evaluation
    R : float
        cutoff range
    
    Returns
    -------
    s : step function
    """
    s = (d <= R) / (np.pi*(R**2))
    return s        

def gauss_2D(d, three_stdev):
    """Returns normalized gaussian 2D scale function

    Parameters
    ----------
    d : floats or np.arrays
         distance at which we need the function evaluated
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : function
        Normalized gaussian 2D function
    """
    stdev = three_stdev/3.0
    Z = gauss(d, stdev, 2)
    return Z

def gauss_lim_2D(d, three_stdev):
    """Returns gausian 2D function cut off after 3 standard deviations.

    Parameters
    ----------
    d : floats or np.arrays
         distance at which we need the function evaluated
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : function
        Normalized gaussian 2D function cut off after three_stdev
    """
    Z = (d <= three_stdev)*gauss_2D(d, three_stdev)
    return Z

def gauss_3D(d, three_stdev):
    """Returns normalized gaussian 3D scale function

    Parameters
    ----------
    d : floats or np.arrays
        distance at which we need the function evaluated        
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : funtion
        Normalized gaussian 3D function
    """
    stdev = three_stdev/3.0
    Z = gauss(d, stdev, 3)
    return Z

def gauss_lim_3D(d, three_stdev):
    """Returns normalized gaussian 3D scale function cut off after 3stdev

    Parameters
    ----------
    d : floats or np.arrays
        distance at which we need the function evaluated        
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : funtion
        Normalized gaussian 3D function cutoff three_Stdev
    """
    Z = gauss_3D(d, three_stdev)
    Z = Z * (d < (three_stdev))
    return Z

def step_3D(d, R):
    """Returns normalized 3D step function.

    Parameters
    ----------
    d : floats or np.arrays
        distance at which we need the function evaluated        
    R : float
        cutoff range

    Returns
    -------
    s : step function in 3D
    """
    
    s = 3/(4*np.pi*R**3)*(d <= R)
    return s

basis_1D = {
    "step": step_1D,
    "gauss": gauss_1D,
    "gauss_lim": gauss_lim_1D,
}


basis_2D = {
    "step": step_2D,
    "gauss": gauss_2D,
    "gauss_lim": gauss_lim_2D,
}

basis_3D = {
    "step": step_3D,
    "gauss": gauss_3D,
    "gauss_lim": gauss_lim_3D,
}
