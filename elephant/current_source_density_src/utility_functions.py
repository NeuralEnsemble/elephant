# -*- coding: utf-8 -*-
"""
These are some useful functions used in CSD methods,
They include CSD source profiles to be used as ground truths,
placement of electrodes in 1D, 2D and 3D., etc
These scripts are based on Grzegorz Parka's,
Google Summer of Code 2014, INFC/pykCSD
This was written by :
Michal Czerwinski, Chaitanya Chintaluri
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
from __future__ import division

import numpy as np
from numpy import exp
import quantities as pq


def patch_quantities():
    """patch quantities with the SI unit Siemens if it does not exist"""
    for symbol, prefix, definition, u_symbol in zip(
        ['siemens', 'S', 'mS', 'uS', 'nS', 'pS'],
        ['', '', 'milli', 'micro', 'nano', 'pico'],
        [pq.A / pq.V, pq.A / pq.V, 'S', 'mS', 'uS', 'nS'],
        [None, None, None, None, u'µS', None]):
        if type(definition) is str:
            definition = lastdefinition / 1000
        if not hasattr(pq, symbol):
            setattr(pq, symbol, pq.UnitQuantity(
                prefix + 'siemens',
                definition,
                symbol=symbol,
                u_symbol=u_symbol))
        lastdefinition = definition
    return


def contains_duplicated_electrodes(elec_pos):
    """Checks for duplicate electrodes
    Parameters
    ----------
    elec_pos : np.array

    Returns
    -------
    has_duplicated_elec : Boolean
    """
    unique_elec_pos = set(map(tuple, elec_pos))
    has_duplicated_elec = len(unique_elec_pos) < len(elec_pos)
    return has_duplicated_elec


def distribute_srcs_1D(X, n_src, ext_x, R_init):
    """Distribute sources in 1D equally spaced
    Parameters
    ----------
    X : np.arrays
        points at which CSD will be estimated
    n_src : int
        number of sources to be included in the model
    ext_x : floats
        how much should the sources extend the area X
    R_init : float
        Same as R in 1D case
    Returns
    -------
    X_src : np.arrays
        positions of the sources
    R : float
        effective radius of the basis element
    """
    X_src = np.mgrid[(np.min(X) - ext_x):(np.max(X) + ext_x):
                     np.complex(0, n_src)]
    R = R_init
    return X_src, R


def distribute_srcs_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """Distribute n_src's in the given area evenly
    Parameters
    ----------
    X, Y : np.arrays
        points at which CSD will be estimated
    n_src : int
        demanded number of sources to be included in the model
    ext_x, ext_y : floats
        how should the sources extend the area X, Y
    R_init : float
        demanded radius of the basis element
    Returns
    -------
    X_src, Y_src : np.arrays
        positions of the sources
    nx, ny : ints
        number of sources in directions x,y
        new n_src = nx * ny may not be equal to the demanded number of sources
    R : float
        effective radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lx_n = Lx + (2 * ext_x)
    Ly_n = Ly + (2 * ext_y)
    [nx, ny, Lx_nn, Ly_nn, ds] = get_src_params_2D(Lx_n, Ly_n, n_src)
    ext_x_n = (Lx_nn - Lx) / 2
    ext_y_n = (Ly_nn - Ly) / 2
    X_src, Y_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):
                            np.complex(0, nx),
                            (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):
                            np.complex(0, ny)]
    # d = round(R_init / ds)
    R = R_init  # R = d * ds
    return X_src, Y_src, R


def get_src_params_2D(Lx, Ly, n_src):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly
    Parameters
    ----------
    Lx, Ly : floats
        lengths in the directions x, y of the area,
        the sources should be placed
    n_src : int
        demanded number of sources

    Returns
    -------
    nx, ny : ints
        number of sources in directions x, y
        new n_src = nx * ny may not be equal to the demanded number of sources
    Lx_n, Ly_n : floats
        updated lengths in the directions x, y
    ds : float
        spacing between the sources
    """
    coeff = [Ly, Lx - Ly, -Lx * n_src]
    rts = np.roots(coeff)
    r = [r for r in rts if type(r) is not complex and r > 0]
    nx = r[0]
    ny = n_src / nx
    ds = Lx / (nx - 1)
    nx = np.floor(nx) + 1
    ny = np.floor(ny) + 1
    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds
    return (nx, ny, Lx_n, Ly_n, ds)


def distribute_srcs_3D(X, Y, Z, n_src, ext_x, ext_y, ext_z, R_init):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly * Lz
    Parameters
    ----------
    X, Y, Z : np.arrays
        points at which CSD will be estimated
    n_src : int
        desired number of sources we want to include in the model
    ext_x, ext_y, ext_z : floats
        how should the sources extend over the area X,Y,Z
    R_init : float
        demanded radius of the basis element

    Returns
    -------
    X_src, Y_src, Z_src : np.arrays
        positions of the sources in 3D space
    nx, ny, nz : ints
        number of sources in directions x,y,z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources

    R : float
        updated radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lz = np.max(Z) - np.min(Z)
    Lx_n = Lx + 2 * ext_x
    Ly_n = Ly + 2 * ext_y
    Lz_n = Lz + 2 * ext_z
    (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n,
                                                              Ly_n,
                                                              Lz_n,
                                                              n_src)
    ext_x_n = (Lx_nn - Lx) / 2
    ext_y_n = (Ly_nn - Ly) / 2
    ext_z_n = (Lz_nn - Lz) / 2
    X_src, Y_src, Z_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):
                                   np.complex(0, nx),
                                   (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):
                                   np.complex(0, ny),
                                   (np.min(Z) - ext_z_n):(np.max(Z) + ext_z_n):
                                   np.complex(0, nz)]
    # d = np.round(R_init / ds)
    R = R_init
    return (X_src, Y_src, Z_src, R)


def get_src_params_3D(Lx, Ly, Lz, n_src):
    """Helps to evenly distribute n_src sources in a cuboid of size Lx * Ly * Lz
    Parameters
    ----------
    Lx, Ly, Lz : floats
        lengths in the directions x, y, z of the area,
        the sources should be placed
    n_src : int
        demanded number of sources to be included in the model
    Returns
    -------
    nx, ny, nz : ints
        number of sources in directions x, y, z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources
    Lx_n, Ly_n, Lz_n : floats
        updated lengths in the directions x, y, z
    ds : float
        spacing between the sources (grid nodes)
    """
    V = Lx * Ly * Lz
    V_unit = V / n_src
    L_unit = V_unit**(1. / 3.)
    nx = np.ceil(Lx / L_unit)
    ny = np.ceil(Ly / L_unit)
    nz = np.ceil(Lz / L_unit)
    ds = Lx / (nx - 1)
    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds
    Lz_n = (nz - 1) * ds
    return (nx, ny, nz, Lx_n, Ly_n, Lz_n, ds)


def generate_electrodes(dim, xlims=[0.1, 0.9], ylims=[0.1, 0.9],
                        zlims=[0.1, 0.9], res=5):
    """Generates electrodes, helpful for FWD funtion.
        Parameters
        ----------
        dim : int
            Dimensionality of the electrodes, 1,2 or 3
        xlims : [start, end]
            Spatial limits of the electrodes
        ylims : [start, end]
            Spatial limits of the electrodes
        zlims : [start, end]
            Spatial limits of the electrodes
        res : int
            How many electrodes in each dimension
        Returns
        -------
        ele_x, ele_y, ele_z : flattened np.array of the electrode pos

    """
    if dim == 1:
        ele_x = np.mgrid[xlims[0]: xlims[1]: np.complex(0, res)]
        ele_x = ele_x.flatten()
        return ele_x
    elif dim == 2:
        ele_x, ele_y = np.mgrid[xlims[0]: xlims[1]: np.complex(0, res),
                                ylims[0]: ylims[1]: np.complex(0, res)]
        ele_x = ele_x.flatten()
        ele_y = ele_y.flatten()
        return ele_x, ele_y
    elif dim == 3:
        ele_x, ele_y, ele_z = np.mgrid[xlims[0]: xlims[1]: np.complex(0, res),
                                       ylims[0]: ylims[1]: np.complex(0, res),
                                       zlims[0]: zlims[1]: np.complex(0, res)]
        ele_x = ele_x.flatten()
        ele_y = ele_y.flatten()
        ele_z = ele_z.flatten()
        return ele_x, ele_y, ele_z


def gauss_1d_dipole(x):
    """1D Gaussian dipole source is placed between 0 and 1
       to be used to test the CSD

       Parameters
       ----------
       x : np.array
           Spatial pts. at which the true csd is evaluated

       Returns
       -------
       f : np.array
           The value of the csd at the requested points
    """
    src = 0.5*exp(-((x-0.7)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    snk = -0.5*exp(-((x-0.3)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    f = src+snk
    return f

def large_source_2D(x, y):
    """2D Gaussian large source profile - to use to test csd
       Parameters
       ----------
       x : np.array
           Spatial x pts. at which the true csd is evaluated
       y : np.array
           Spatial y pts. at which the true csd is evaluated
       Returns
       -------
       f : np.array
           The value of the csd at the requested points
    """
    zz = [0.4, -0.3, -0.1, 0.6]
    zs = [0.2, 0.3, 0.4, 0.2]
    f1 = 0.5965*exp( (-1*(x-0.1350)**2 - (y-0.8628)**2) /0.4464)* exp(-(-zz[0])**2 / zs[0]) /exp(-(zz[0])**2/zs[0])
    f2 = -0.9269*exp( (-2*(x-0.1848)**2 - (y-0.0897)**2) /0.2046)* exp(-(-zz[1])**2 / zs[1]) /exp(-(zz[1])**2/zs[1]);
    f3 = 0.5910*exp( (-3*(x-1.3189)**2 - (y-0.3522)**2) /0.2129)* exp(-(-zz[2])**2 / zs[2]) /exp(-(zz[2])**2/zs[2]);
    f4 = -0.1963*exp( (-4*(x-1.3386)**2 - (y-0.5297)**2) /0.2507)* exp(-(-zz[3])**2 / zs[3]) /exp(-(zz[3])**2/zs[3]);
    f = f1+f2+f3+f4
    return f

def small_source_2D(x, y):
    """2D Gaussian small source profile - to be used to test csd
       Parameters
       ----------
       x : np.array
           Spatial x pts. at which the true csd is evaluated
       y : np.array
           Spatial y pts. at which the true csd is evaluated
       Returns
       -------
       f : np.array
           The value of the csd at the requested points
    """
    def gauss2d(x,y,p):
        rcen_x = p[0] * np.cos(p[5]) - p[1] * np.sin(p[5])
        rcen_y = p[0] * np.sin(p[5]) + p[1] * np.cos(p[5])
        xp = x * np.cos(p[5]) - y * np.sin(p[5])
        yp = x * np.sin(p[5]) + y * np.cos(p[5])

        g = p[4]*exp(-(((rcen_x-xp)/p[2])**2+
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    f1 = gauss2d(x,y,[0.3,0.7,0.038,0.058,0.5,0.])
    f2 = gauss2d(x,y,[0.3,0.6,0.038,0.058,-0.5,0.])
    f3 = gauss2d(x,y,[0.45,0.7,0.038,0.058,0.5,0.])
    f4 = gauss2d(x,y,[0.45,0.6,0.038,0.058,-0.5,0.])
    f = f1+f2+f3+f4
    return f

def gauss_3d_dipole(x, y, z):
    """3D Gaussian dipole profile - to be used to test csd.
       Parameters
       ----------
       x : np.array
           Spatial x pts. at which the true csd is evaluated
       y : np.array
           Spatial y pts. at which the true csd is evaluated
       z : np.array
           Spatial z pts. at which the true csd is evaluated
       Returns
       -------
       f : np.array
           The value of the csd at the requested points
    """
    x0, y0, z0 = 0.3, 0.7, 0.3
    x1, y1, z1 = 0.6, 0.5, 0.7
    sig_2 = 0.023
    A = (2*np.pi*sig_2)**-1
    f1 = A*exp( (-(x-x0)**2 -(y-y0)**2 -(z-z0)**2) / (2*sig_2) )
    f2 = -1*A*exp( (-(x-x1)**2 -(y-y1)**2 -(z-z1)**2) / (2*sig_2) )
    f = f1+f2
    return f
