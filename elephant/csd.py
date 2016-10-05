#!/usr/bin/env python
"""This script is used to generate Current Source Density Estimates
This was written by :
Chaitanya Chintaluri,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
"""
from __future__ import division

import neo
import quantities as pq
import numpy as np

from scipy import io
from scipy.integrate import simps
from elephant.csd import KCSD
from elephant.csd import icsd
import elephant.csd.utility_functions as utils

utils.patch_quantities()

available_1d = ['StandardCSD', 'DeltaiCSD', 'StepiCSD', 'SplineiCSD', 'KCSD1D']
available_2d = ['KCSD2D', 'MoIKCSD']
available_3d = ['KCSD3D']

kernel_methods = ['KCSD1D', 'KCSD2D', 'KCSD3D', 'MoIKCSD']
icsd_methods = ['DeltaiCSD', 'StepiCSD', 'SplineiCSD']

espen_implemented = ['StandardCSD', 'DeltaiCSD', 'StepiCSD', 'SplineiCSD']


def estimate_csd(lfp, coords=None, method=None, params={}, cv_params={}):
    """Fuction call to compute the current source density.
        Parameters
        ----------
        lfp : list(neo.AnalogSignal type objects)
            positions of electrodes can be added as neo.RecordingChannel
            coordinate or sent externally as a func argument (See coords)
        coords : [Optional] corresponding spatial coordinates of the electrodes
            Defaults to None
            Otherwise looks for RecordingChannels coordinate
        method : string
            Pick a method corresonding to the setup, in this implenetation
            For Laminar probe style (1D), use 'KCSD1D' or 'StandardCSD',
             or 'DeltaiCSD' or 'StepiCSD' or 'SplineiCSD'
            For MEA probe style (2D),  use 'KCSD2D', or 'MoIKCSD'
            For array of laminar probes (3D), use 'KCSD3D'
            Defaults to None
        params : dict
            The parameters corresponding to the method chosen
            See the documentation of the individual method
            Default is {} - picks the best parameters,
        cv_params : dict
            The kernel methods have the Crossvalidation possibility, and
            the parameters can be pass here.
            Default is {} - CV only over lambda in this case.
        Returns
        -------
        Estimated CSD
           neo.AnalogSignalArray Object
           annotated with the spatial coordinates

        Raises
        ------
        AttributeError
            No units specified for electrode spatial coordinates
        ValueError
            Invalid function arguments, wrong method name, or
            mismatching coordinates
        TypeError
            Invalid cv_param argument passed
    """
    if not isinstance(lfp[0], neo.AnalogSignal):
        print('Parameter `lfp` must be a list(neo.AnalogSignal type objects)')
        raise TypeError
    if coords is None:
        coords = []
        for ii in lfp:
            coords.append(ii.recordingchannel.coordinate.rescale(pq.mm))
    else:
        scaled_coords = []
        for coord in coords:
            try:
                scaled_coords.append(coord.rescale(pq.mm))
            except AttributeError:
                raise AttributeError('No units given for electrode spatial \
                coordinates')
        coords = scaled_coords
    if method is None:
        raise ValueError('Must specify a method of CSD implementation')
    if len(coords) != len(lfp):
        raise ValueError('Number of signals and coords is not same')
    for ii in coords:  # CHECK for Dimensionality of electrodes
        if len(ii) > 3:
            raise ValueError('Invalid number of coordinate positions')
    dim = len(coords[0])  # TODO : Generic co-ordinates!
    if dim == 1 and (method not in available_1d):
        raise ValueError('Invalid method, Available options are:',
                         available_1d)
    if dim == 2 and (method not in available_2d):
        raise ValueError('Invalid method, Available options are:',
                         available_2d)
    if dim == 3 and (method not in available_3d):
        raise ValueError('Invalid method, Available options are:',
                         available_3d)

    if method in kernel_methods:
        input_array = np.zeros((len(lfp), lfp[0].magnitude.shape[0]))
        for ii, jj in enumerate(lfp):
            input_array[ii, :] = jj.rescale(pq.mV).magnitude
        kernel_method = getattr(KCSD, method)  # fetch the class 'KCSD1D'
        k = kernel_method(np.array(coords), input_array, **params)
        if bool(cv_params):  # not empty then
            if len(cv_params.keys() and ['Rs', 'lambdas']) != 2:
                raise TypeError('Invalid cv_params argument passed')
            k.cross_validate(**cv_params)
        estm_csd = k.values()
        estm_csd = np.rollaxis(estm_csd, -1, 0)
        output = neo.AnalogSignalArray(estm_csd * pq.uA / pq.mm**3,
                                       t_start=lfp[0].t_start,
                                       sampling_rate=lfp[0].sampling_rate)
        if dim == 1:
            output.annotate(x_coords=k.estm_x)
        elif dim == 2:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y)
        elif dim == 3:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y,
                            z_coords=k.estm_z)
    elif method in espen_implemented:

        coords = np.array(coords) * coords[0].units  # MOVE TO ICSD?
        if method in icsd_methods:
            try:
                coords = coords.rescale(params['diam'].units)
            except KeyError:  # Then why specify as a default in icsd?
                print("Parameter diam must be specified for iCSD \
                      methods: {}".format(", ".join(icsd_methods)))
                raise ValueError

        if 'f_type' in params:
            if (params['f_type'] is not 'identity') and (params['f_order'] is
                                                         None):
                print("The order of {} filter must be \
                      specified".format(params['f_type']))
                raise ValueError
        lfp = neo.AnalogSignalArray(np.asarray(lfp).T, units=lfp[0].units,
                                    sampling_rate=lfp[0].sampling_rate)
        lfp_pqarr = lfp.magnitude.T * lfp.units
        csd_method = getattr(icsd, method)  # fetch class from icsd.py file
        csd_estimator = csd_method(lfp_pqarr, coords, **params)
        csd_pqarr = csd_estimator.get_csd()

        # What to do of the csd_filtered!
        csd_pqarr_filtered = csd_estimator.filter_csd(csd_pqarr)
        csd_filtered = neo.AnalogSignalArray(csd_pqarr_filtered.T,
                                             t_start=lfp[0].t_start,
                                             sampling_rate=lfp[0].sampling_rate)

        # MISSING ANNOTATIONS! At which points is this the CSD!
        output = neo.AnalogSignalArray(csd_pqarr.T, t_start=lfp.t_start,
                                       sampling_rate=lfp.sampling_rate)
    return output


def generate_lfp(csd_profile, ele_xx, ele_yy=None, ele_zz=None,
                 xlims=[0., 1.], ylims=[0., 1.], zlims=[0., 1.], res=50):
    """Forward modelling for the getting the potentials for testing CSD

        Parameters
        ----------
        csd_profile : fuction that computes True CSD profile
            Available options are
            1D : gauss_1d_dipole
            2D : large_source_2D and small_source_2D
            3D : gauss_3d_dipole
        ele_xx : np.array
            Positions of the x coordinates of the electrodes
        ele_yy : np.array
            Positions of the y coordinates of the electrodes
            Defaults ot None, use in 2D or 3D cases only
        ele_zz : np.array
            Positions of the z coordinates of the electrodes
            Defaults ot None, use in 3D case only
        x_lims : [start, end]
            The starting spatial coordinate and the ending for integration
            Defaults to [0.,1.]
        y_lims : [start, end]
            The starting spatial coordinate and the ending for integration
            Defaults to [0.,1.], use only in 2D and 3D case
        z_lims : [start, end]
            The starting spatial coordinate and the ending for integration
            Defaults to [0.,1.], use only in 3D case
        res : int
            The resolution of the integration
            Defaults to 50
        Returns
        -------
        LFP : list(neo.AnalogSignal type objects)
           The potentials created by the csd profile at the electrode positions
           The electrode postions are attached as RecordingChannel's coordinate
    """
    def integrate_1D(x0, csd_x, csd, h):
        m = np.sqrt((csd_x - x0)**2 + h**2) - abs(csd_x - x0)
        y = csd * m
        I = simps(y, csd_x)
        return I

    def integrate_2D(x, y, xlin, ylin, csd, h, X, Y):
        Ny = ylin.shape[0]
        m = np.sqrt((x - X)**2 + (y - Y)**2)
        m[m < 0.0000001] = 0.0000001
        y = np.arcsinh(2 * h / m) * csd
        I = np.zeros(Ny)
        for i in range(Ny):
            I[i] = simps(y[:, i], ylin)
        F = simps(I, xlin)
        return F

    def integrate_3D(x, y, z, xlim, ylim, zlim, csd, xlin, ylin, zlin,
                     X, Y, Z):
        Nz = zlin.shape[0]
        Ny = ylin.shape[0]
        m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
        m[m < 0.0000001] = 0.0000001
        z = csd / m
        Iy = np.zeros(Ny)
        for j in range(Ny):
            Iz = np.zeros(Nz)
            for i in range(Nz):
                Iz[i] = simps(z[:, j, i], zlin)
            Iy[j] = simps(Iz, ylin)
        F = simps(Iy, xlin)
        return F
    dim = 1
    if ele_zz is not None:
        dim = 3
    elif ele_yy is not None:
        dim = 2
    x = np.linspace(xlims[0], xlims[1], res)
    if dim >= 2:
        y = np.linspace(ylims[0], ylims[1], res)
    if dim == 3:
        z = np.linspace(zlims[0], zlims[1], res)
    sigma = 1.0
    h = 50.
    pots = np.zeros(len(ele_xx))
    if dim == 1:
        chrg_x = np.linspace(xlims[0], xlims[1], res)
        csd = csd_profile(chrg_x)
        for ii in range(len(ele_xx)):
            pots[ii] = integrate_1D(ele_xx[ii], chrg_x, csd, h)
        pots /= 2. * sigma  # eq.: 26 from Potworowski et al
        ele_pos = ele_xx
    elif dim == 2:
        chrg_x, chrg_y = np.mgrid[xlims[0]:xlims[1]:np.complex(0, res),
                                  ylims[0]:ylims[1]:np.complex(0, res)]
        csd = csd_profile(chrg_x, chrg_y)
        for ii in range(len(ele_xx)):
            pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii],
                                    x, y, csd, h, chrg_x, chrg_y)
        pots /= 2 * np.pi * sigma
        ele_pos = np.vstack((ele_xx, ele_yy)).T
    elif dim == 3:
        chrg_x, chrg_y, chrg_z = np.mgrid[xlims[0]:xlims[1]:np.complex(0, res),
                                          ylims[0]:ylims[1]:np.complex(0, res),
                                          zlims[0]:zlims[1]:np.complex(0, res)]
        csd = csd_profile(chrg_x, chrg_y, chrg_z)
        xlin = chrg_x[:, 0, 0]
        ylin = chrg_y[0, :, 0]
        zlin = chrg_z[0, 0, :]
        for ii in range(len(ele_xx)):
            pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                    xlims, ylims, zlims, csd,
                                    xlin, ylin, zlin,
                                    chrg_x, chrg_y, chrg_z)
        pots /= 4 * np.pi * sigma
        ele_pos = np.vstack((ele_xx, ele_yy, ele_zz)).T
    pots = np.reshape(pots, (-1, 1)) * pq.mV
    ele_pos = ele_pos * pq.mm
    lfp = []
    for ii in range(len(pots)):
        rc = neo.RecordingChannel()
        rc.coordinate = ele_pos[ii]
        asig = neo.AnalogSignal(pots[ii], sampling_rate=pq.kHz)
        rc.analogsignals = [asig]
        rc.create_relationship()
        lfp.append(asig)
    # lfp = neo.AnalogSignalArray(lfp, sampling_rate=1000*pq.Hz, units='mV')
    return lfp

if __name__ == '__main__':
    dim = 1
    if dim == 1:
        ele_pos = utils.generate_electrodes(dim=1).reshape(5, 1)
        lfp = generate_lfp(utils.gauss_1d_dipole, ele_pos)

        # test_data = io.loadmat('./csd/test_data.mat')
        # lfp_data = test_data['pot1'] * 1e-2 * pq.V
        # z_data = np.linspace(100E-6, 2300E-6, 23).reshape(23, 1) * pq.m
        # lfp = []
        # for ii in range(lfp_data.shape[0]):
        #     rc = neo.RecordingChannel()
        #     rc.coordinate = z_data[ii]
        #     asig = neo.AnalogSignal(lfp_data[ii, :], sampling_rate=2.0 *
        #                             pq.kHz)
        #     rc.analogsignals = [asig]
        #     rc.create_relationship()
        #     lfp.append(asig)

        # test_method = 'StandardCSD'
        # test_params = {'f_type': 'gaussian', 'f_order': (3, 1)}

        test_method = 'DeltaiCSD'
        test_params = {'f_type': 'gaussian', 'f_order': (3, 1),
                       'diam': 500E-6 * pq.m}

        # test_method = 'StepiCSD'
        # test_params = {'f_type': 'gaussian', 'f_order': (3, 1),
        #                'diam': 500E-6 * pq.m, 'h': 1e-4 * pq.m }

        # test_method = 'SplineiCSD'
        # test_params = {'f_type': 'gaussian', 'f_order': (3, 1),
        #                'diam': 500E-6 * pq.m}

        # test_method = 'KCSD1D'
        # test_params = {'h': 50.}
    elif dim == 2:
        xx_ele, yy_ele = utils.generate_electrodes(dim=2)
        lfp = generate_lfp(utils.large_source_2D, xx_ele, yy_ele)
        test_method = 'KCSD2D'
        test_params = {'sigma': 1.}
    elif dim == 3:
        xx_ele, yy_ele, zz_ele = utils.generate_electrodes(dim=3, res=3)
        lfp = generate_lfp(utils.gauss_3d_dipole, xx_ele, yy_ele, zz_ele)
        test_method = 'KCSD3D'
        test_params = {'gdx': 0.1, 'gdy': 0.1, 'gdz': 0.1, 'src_type': 'step'}

    if test_method in kernel_methods:
        result = estimate_csd(lfp, method=test_method, params=test_params,
                              cv_params={'Rs': np.array((0.1, 0.25, 0.5))})
    elif test_method in espen_implemented:
        result = estimate_csd(lfp, method=test_method, params=test_params)

    print(result)
    print(result.t_start)
    print(result.sampling_rate)
    print(len(result.times))
