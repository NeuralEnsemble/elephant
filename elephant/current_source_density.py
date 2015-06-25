import quantities as pq
import numpy as np
import neo
from current_source_density.KCSD2D import KCSD2D
from current_source_density.KCSD2D_Helpers import KCSD2D_params

def CSD(analog_signals, coords=None, method='KCSD2D', params={}):
    if coords==None:
        coords = []
        for ii in analog_signals:
            coords.append(ii.recordingchannel.coordinate)
    
    if len(coords) != len(analog_signals):
        raise ValueError('Number of signals and coords is not same')
    print coords

    for ii in coords: # CHECK for Dimensionality of electrodes
        if len(ii) != 2:
            raise ValueError('Invalid number of coordinate positions')

    if params == {}:
        params = KCSD2D_params
    print analog_signals[0]

    input_array=np.zeros((len(analog_signals),analog_signals[0].magnitude.shape[0]))
    for ii,jj in enumerate(analog_signals):
        input_array[ii,:]=jj.magnitude

    k = KCSD2D(np.array(coords), input_array, params=params)
    print k.values()
    

if __name__ == '__main__':
    ele_pos = np.array([[-0.2, -0.2],[0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5],
                        [1.2, 1.2]])
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])    

    an_sigs=[]
    for ii in range(len(pots)):
        rc=neo.RecordingChannel()
        rc.coordinate=ele_pos[ii]*pq.mm
        asig=neo.AnalogSignal(pots[ii]*pq.mV,sampling_rate=1000*pq.ms)
        rc.analogsignals=[asig]
        rc.create_relationship()
        an_sigs.append(asig)
   
    CSD(an_sigs)
