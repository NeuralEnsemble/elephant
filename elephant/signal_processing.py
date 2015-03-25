'''
signal_proc module

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
'''

from __future__ import division, print_function
import numpy as np
import quantities as pq


def zscore(signal, inplace=True):
    '''
    Apply a z-score operation to one or several AnalogSignalArray objects.

    The z-score operation subtracts the mean :math:`\\mu` of the signal, and
    divides by its standard deviation :math:`\\sigma`:

    .. math::
         Z(x(t))= \\frac{x(t)-\\mu}{\\sigma}

    If an AnalogSignalArray containing multiple signals is provided, the
    z-transform is always calculated for each signal individually.

    If a list of AnalogSignalArray objects is supplied, the mean and standard
    deviation are calculated across all objects of the list. Thus, all list
    elements are z-transformed by the same values of :math:`\\mu` and
    :math:`\\sigma`. For AnalogSignalArrays, each signal of the array is
    treated separately across list elements. Therefore, the number of signals
    must be identical for each AnalogSignalArray of the list.

    Parameters
    ----------
    signal : neo.AnalogSignalArray or list of neo.AnalogSignalArray
        Signals for which to calculate the z-score.
    inplace : bool
        If True, the contents of the input signal(s) is replaced by the
        z-transformed signal. Otherwise, a copy of the original
        AnalogSignalArray(s) is returned. Default: True

    Returns
    -------
    neo.AnalogSignalArray or list of neo.AnalogSignalArray
        The output format matches the input format: for each supplied
        AnalogSignalArray object a corresponding object is returned containing
        the z-transformed signal with the unit dimensionless.

    Use Case
    --------
    You may supply a list of AnalogSignalArray objects, where each object in
    the list contains the data of one trial of the experiment, and each signal
    of the AnalogSignalArray corresponds to the recordings from one specific
    electrode in a particular trial. In this scenario, you will z-transform the
    signal of each electrode separately, but transform all trials of a given
    electrode in the same way.

    Example
    -------
    >>> a=neo.AnalogSignalArray(
            [1,2,3,4,5,6]*mV,
            t_start=0*s, sampling_rate=1000*Hz)

    >>> b=neo.AnalogSignalArray(
            np.transpose([[1,2,3,4,5,6],[11,12,13,14,15,16]])*mV,
            t_start=0*s, sampling_rate=1000*Hz)

    >>> c=neo.AnalogSignalArray(
            np.transpose([[21,22,23,24,25,26],[31,32,33,34,35,36]])*mV,
            t_start=0*s, sampling_rate=1000*Hz)

    >>> print zscore(a)
    <AnalogSignalArray(array([-1.46385011, -0.87831007, -0.29277002,
        0.29277002, 0.87831007, 1.46385011]) * mV, [0.0 s, 0.006 s],
        sampling rate: 1000.0 Hz)>

    >>> print zscore(b)
    <AnalogSignalArray(array([[-1.46385011, -1.46385011],
       [-0.87831007, -0.87831007],
       [-0.29277002, -0.29277002],
       [ 0.29277002,  0.29277002],
       [ 0.87831007,  0.87831007],
       [ 1.46385011,  1.46385011]]) * mV, [0.0 s, 0.006 s],
       sampling rate: 1000.0 Hz)>

    >>> print zscore([b,c])
    [<AnalogSignalArray(array([[-1.11669108, -1.08361877],
       [-1.0672076 , -1.04878252],
       [-1.01772411, -1.01394628],
       [-0.96824063, -0.97911003],
       [-0.91875714, -0.94427378],
       [-0.86927366, -0.90943753]]) * mV, [0.0 s, 0.006 s],
       sampling rate: 1000.0 Hz)>,
       <AnalogSignalArray(array([[ 0.78170952,  0.84779261],
       [ 0.86621866,  0.90728682],
       [ 0.9507278 ,  0.96678104],
       [ 1.03523694,  1.02627526],
       [ 1.11974608,  1.08576948],
       [ 1.20425521,  1.1452637 ]]) * mV, [0.0 s, 0.006 s],
       sampling rate: 1000.0 Hz)>]
    '''
    # Transform input to a list
    if type(signal) is not list:
        signal = [signal]

    # Calculate mean and standard deviation
    m = np.mean(np.concatenate(signal), axis=0, keepdims=True)
    s = np.std(np.concatenate(signal), axis=0, keepdims=True)

    if not inplace:
        # Create new signal instance
        result = [sig.duplicate_with_new_array(
            (sig.magnitude - m.magnitude) / s.magnitude) for sig in signal]
        for sig in result:
            sig /= sig.units
    else:
        # Overwrite signal
        for sig in signal:
            sig[:] = pq.Quantity(
                (sig.magnitude - m.magnitude) / s.magnitude,
                units=sig.units)
            sig /= sig.units
        result = signal

    # Return single object, or list of objects
    if len(result) == 1:
        return result[0]
    else:
        return result
