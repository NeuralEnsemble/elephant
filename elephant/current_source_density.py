# -*- coding: utf-8 -*-
"""
*\"Current Source Density analysis (CSD) is a class of methods of analysis of
extracellular electric potentials recorded at multiple sites leading to
estimates of current sources generating the measured potentials. It is usually
applied to low-frequency part of the potential (called the Local Field
Potential, LFP) and to simultaneous recordings or to recordings taken with
fixed time reference to the onset of specific stimulus (Evoked Potentials).\"*
(Definition by Prof.Daniel K. Wójcik for Encyclopedia of Computational
Neuroscience.)

CSD is also called as Source Localization or Source Imaging in the EEG circles.
Here are CSD methods for different types of electrode configurations.

- 1D - laminar probe like electrodes.
- 2D - Microelectrode Array like
- 3D - UtahArray or multiple laminar probes.

The following methods have been implemented so far

- 1D: StandardCSD, DeltaiCSD, SplineiCSD, StepiCSD, KCSD1D
- 2D: KCSD2D, MoIKCSD (Saline layer on top of slice)
- 3D: KCSD3D

Each listed method has certain advantages. The KCSD methods, for instance, can
handle broken or irregular electrode configurations electrode.

.. autosummary::
    :toctree: _toctree/current_source_density

    estimate_csd
    generate_lfp

"""

from __future__ import division, print_function, unicode_literals

import neo
import numpy as np
import quantities as pq
from scipy.integrate import simps

import elephant.current_source_density_src.utility_functions as utils
from elephant.current_source_density_src import KCSD, icsd
from elephant.utils import deprecated_alias

__all__ = [
    "estimate_csd",
    "generate_lfp"
]

utils.patch_quantities()

available_1d = ['StandardCSD', 'DeltaiCSD', 'StepiCSD', 'SplineiCSD', 'KCSD1D']
available_2d = ['KCSD2D', 'MoIKCSD']
available_3d = ['KCSD3D']

kernel_methods = ['KCSD1D', 'KCSD2D', 'KCSD3D', 'MoIKCSD']
icsd_methods = ['DeltaiCSD', 'StepiCSD', 'SplineiCSD']

py_iCSD_toolbox = ['StandardCSD'] + icsd_methods


@deprecated_alias(coords='coordinates')
def estimate_csd(lfp, coordinates=None, method=None,
                 process_estimate=True, **kwargs):
    """
    Function call to compute the current source density (CSD) from
    extracellular potential recordings (local field potentials - LFP) using
    laminar electrodes or multi-contact electrodes with 2D or 3D geometries.

    Parameters
    ----------
    lfp : neo.AnalogSignal
        positions of electrodes can be added as neo.RecordingChannel
        coordinate or sent externally as a func argument (See coords)
    coordinates : [Optional] corresponding spatial coordinates of the
        electrodes.
        Defaults to None
        Otherwise looks for ChannelIndex coordinate
    method : string
        Pick a method corresponding to the setup, in this implementation
        For Laminar probe style (1D), use 'KCSD1D' or 'StandardCSD',
         or 'DeltaiCSD' or 'StepiCSD' or 'SplineiCSD'
        For MEA probe style (2D),  use 'KCSD2D', or 'MoIKCSD'
        For array of laminar probes (3D), use 'KCSD3D'
        Defaults to None
    process_estimate : bool
        In the py_iCSD_toolbox this corresponds to the filter_csd -
        the parameters are passed as kwargs here ie., f_type and f_order
        In the kcsd methods this corresponds to cross_validate -
        the parameters are passed as kwargs here ie., lambdas and Rs
        Defaults to True
    kwargs : parameters to each method
        The parameters corresponding to the method chosen
        See the documentation of the individual method
        Default is {} - picks the best parameters,

    Returns
    -------
    Estimated CSD
       neo.AnalogSignal object
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
    if not isinstance(lfp, neo.AnalogSignal):
        raise TypeError('Parameter `lfp` must be a neo.AnalogSignal object')
    if coordinates is None:
        coordinates = lfp.channel_index.coordinates
    else:
        scaled_coords = []
        for coord in coordinates:
            try:
                scaled_coords.append(coord.rescale(pq.mm))
            except AttributeError:
                raise AttributeError('No units given for electrode spatial \
                coordinates')
        coordinates = scaled_coords
    if method is None:
        raise ValueError('Must specify a method of CSD implementation')
    if len(coordinates) != lfp.shape[1]:
        raise ValueError('Number of signals and coords is not same')
    for ii in coordinates:  # CHECK for Dimensionality of electrodes
        if len(ii) > 3:
            raise ValueError('Invalid number of coordinate positions')
    dim = len(coordinates[0])  # TODO : Generic co-ordinates!
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
        lambdas = kwargs.pop('lambdas', None)
        Rs = kwargs.pop('Rs', None)
        k = kernel_method(np.array(coordinates), input_array.T, **kwargs)
        if process_estimate:
            k.cross_validate(lambdas, Rs)
        estm_csd = k.values()
        estm_csd = np.rollaxis(estm_csd, -1, 0)
        output = neo.AnalogSignal(estm_csd * pq.uA / pq.mm**3,
                                  t_start=lfp.t_start,
                                  sampling_rate=lfp.sampling_rate)

        if dim == 1:
            output.annotate(x_coords=k.estm_x)
        elif dim == 2:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y)
        elif dim == 3:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y,
                            z_coords=k.estm_z)
    elif method in py_iCSD_toolbox:

        coordinates = np.array(coordinates) * coordinates[0].units

        if method in icsd_methods:
            try:
                coordinates = coordinates.rescale(kwargs['diam'].units)
            except KeyError:  # Then why specify as a default in icsd?
                # All iCSD methods explicitly assume a source
                # diameter in contrast to the stdCSD  that
                # implicitly assume infinite source radius
                raise ValueError("Parameter diam must be specified for iCSD \
                                  methods: {}".format(", ".join(icsd_methods)))

        if 'f_type' in kwargs:
            if (kwargs['f_type'] != 'identity') and  \
               (kwargs['f_order'] is None):
                raise ValueError("The order of {} filter must be \
                                  specified".format(kwargs['f_type']))

        lfp = neo.AnalogSignal(np.asarray(lfp).T, units=lfp.units,
                               sampling_rate=lfp.sampling_rate)
        csd_method = getattr(icsd, method)  # fetch class from icsd.py file
        csd_estimator = csd_method(lfp=lfp.magnitude * lfp.units,
                                   coord_electrode=coordinates.flatten(),
                                   **kwargs)
        csd_pqarr = csd_estimator.get_csd()

        if process_estimate:
            csd_pqarr_filtered = csd_estimator.filter_csd(csd_pqarr)
            output = neo.AnalogSignal(csd_pqarr_filtered.T,
                                      t_start=lfp.t_start,
                                      sampling_rate=lfp.sampling_rate)
        else:
            output = neo.AnalogSignal(csd_pqarr.T, t_start=lfp.t_start,
                                      sampling_rate=lfp.sampling_rate)
        output.annotate(x_coords=coordinates)
    return output


@deprecated_alias(ele_xx='x_positions', ele_yy='y_positions',
                  ele_zz='z_positions', xlims='x_limits', ylims='y_limits',
                  zlims='z_limits', res='resolution')
def generate_lfp(csd_profile, x_positions, y_positions=None, z_positions=None,
                 x_limits=[0., 1.], y_limits=[0., 1.], z_limits=[0., 1.],
                 resolution=50):
    """
    Forward modelling for getting the potentials for testing Current Source
    Density (CSD).

    Parameters
    ----------
    csd_profile : callable
        A function that computes true CSD profile.
        Available options are (see ./csd/utility_functions.py)
        1D : gauss_1d_dipole
        2D : large_source_2D and small_source_2D
        3D : gauss_3d_dipole
    x_positions : np.ndarray
        Positions of the x coordinates of the electrodes
    y_positions : np.ndarray, optional
        Positions of the y coordinates of the electrodes
        Defaults to None, use in 2D or 3D cases only
    z_positions : np.ndarray, optional
        Positions of the z coordinates of the electrodes
        Defaults to None, use in 3D case only
    x_limits : list, optional
        A list of [start, end].
        The starting spatial coordinate and the ending for integration
        Defaults to [0.,1.]
    y_limits : list, optional
        A list of [start, end].
        The starting spatial coordinate and the ending for integration
        Defaults to [0.,1.], use only in 2D and 3D case
    z_limits : list, optional
        A list of [start, end].
        The starting spatial coordinate and the ending for integration
        Defaults to [0.,1.], use only in 3D case
    resolution : int, optional
        The resolution of the integration
        Defaults to 50

    Returns
    -------
    LFP : neo.AnalogSignal
       The potentials created by the csd profile at the electrode positions.
       The electrode positions are attached as RecordingChannel's coordinate.
    """

    def integrate_1D(x0, csd_x, csd, h):
        m = np.sqrt((csd_x - x0) ** 2 + h ** 2) - abs(csd_x - x0)
        y = csd * m
        I = simps(y, csd_x)
        return I

    def integrate_2D(x, y, xlin, ylin, csd, h, X, Y):
        x = np.reshape(x, (1, 1, len(x)))
        y = np.reshape(y, (1, 1, len(y)))
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)
        csd = np.expand_dims(csd, axis=2)
        m = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
        np.clip(m, a_min=0.0000001, a_max=None, out=m)
        y = np.arcsinh(2 * h / m) * csd
        I = simps(y.T, ylin)
        F = simps(I, xlin)
        return F

    def integrate_3D(x, y, z, csd, xlin, ylin, zlin, X, Y, Z):
        m = np.sqrt((x - X) ** 2 + (y - Y) ** 2 + (z - Z) ** 2)
        np.clip(m, a_min=0.0000001, a_max=None, out=m)
        z = csd / m
        Iy = simps(np.transpose(z, (1, 0, 2)), zlin)
        Iy = simps(Iy, ylin)
        F = simps(Iy, xlin)
        return F

    dim = 1
    if z_positions is not None:
        dim = 3
    elif y_positions is not None:
        dim = 2

    x = np.linspace(x_limits[0], x_limits[1], resolution)
    sigma = 1.0
    h = 50.
    if dim == 1:
        chrg_x = x
        csd = csd_profile(chrg_x)
        pots = integrate_1D(x_positions, chrg_x, csd, h)
        pots /= 2. * sigma  # eq.: 26 from Potworowski et al
        ele_pos = x_positions
    elif dim == 2:
        y = np.linspace(y_limits[0], y_limits[1], resolution)
        chrg_x = np.expand_dims(x, axis=1)
        chrg_y = np.expand_dims(y, axis=0)
        csd = csd_profile(chrg_x, chrg_y)
        pots = integrate_2D(x_positions, y_positions,
                            x, y,
                            csd, h,
                            chrg_x, chrg_y)
        pots /= 2 * np.pi * sigma
        ele_pos = np.vstack((x_positions, y_positions)).T
    elif dim == 3:
        y = np.linspace(y_limits[0], y_limits[1], resolution)
        z = np.linspace(z_limits[0], z_limits[1], resolution)
        chrg_x, chrg_y, chrg_z = np.mgrid[
            x_limits[0]: x_limits[1]: np.complex(0, resolution),
            y_limits[0]: y_limits[1]: np.complex(0, resolution),
            z_limits[0]: z_limits[1]: np.complex(0, resolution)
        ]

        csd = csd_profile(chrg_x, chrg_y, chrg_z)

        pots = np.zeros(len(x_positions))
        for ii in range(len(x_positions)):
            pots[ii] = integrate_3D(x_positions[ii], y_positions[ii],
                                    z_positions[ii],
                                    csd,
                                    x, y, z,
                                    chrg_x, chrg_y, chrg_z)
        pots /= 4 * np.pi * sigma
        ele_pos = np.vstack((x_positions, y_positions, z_positions)).T
    ele_pos = ele_pos * pq.mm
    ch = neo.ChannelIndex(index=range(len(pots)))
    asig = neo.AnalogSignal(np.expand_dims(pots, axis=0),
                            sampling_rate=pq.kHz, units='mV')
    ch.coordinates = ele_pos
    ch.analogsignals.append(asig)
    ch.create_relationship()
    return asig
