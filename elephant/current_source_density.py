import neo
import quantities as pq
import numpy as np

from scipy.integrate import simps 
from numpy import exp


def CSD(analog_signals, coords=None, method='KCSD2D', params={}):
    if coords==None:
        coords = []
        for ii in analog_signals:
            coords.append(ii.recordingchannel.coordinate)
    
    if len(coords) != len(analog_signals):
        raise ValueError('Number of signals and coords is not same')

    for ii in coords: # CHECK for Dimensionality of electrodes
        if len(ii) != 2:
            raise ValueError('Invalid number of coordinate positions')

    input_array=np.zeros((len(analog_signals),analog_signals[0].magnitude.shape[0]))
    for ii,jj in enumerate(analog_signals):
        input_array[ii,:]=jj.magnitude

    if method == 'KCSD2D':
        from current_source_density.KCSD2D import KCSD2D
        from current_source_density.KCSD2D_Helpers import KCSD2D_params

        if params == {}:
            params = KCSD2D_params
        k = KCSD2D(np.array(coords), input_array, params=params)
        k.cross_validate(Rs=np.array((0.2,0.22,0.24)))
        csd = k.values()
        csd = np.rollaxis(csd, -1, 0)
        output= neo.AnalogSignalArray(csd*pq.uA/pq.mm**3, t_start=analog_signals[0].t_start,
                                     sampling_rate=analog_signals[0].sampling_rate)
        output.annotate(x_coords=k.space_X, y_coords=k.space_Y)
    else:
        raise KeyError('Method not available')
    return output

def FWD(csd_profile, ele_xx, ele_yy, xlims=[0.,1.], ylims=[0.,1.], zlims=50):
    '''Forward modelling for the 2D case'''
    def integrate_2D(x, y, xlin, ylin, csd, h):
        """
        X,Y - parts of meshgrid - Mihav's implementation
        """
        X, Y = np.meshgrid(xlin, ylin)
        Ny = ylin.shape[0]
        # construct 2-D integrand
        m = np.sqrt((x - X)**2 + (y - Y)**2)
        m[m < 0.0000001] = 0.0000001 # I increased acuracy
        y = np.arcsinh(2*h / m) * csd  # corrected
        # do a 1-D integral over every row
        I = np.zeros(Ny)
        for i in xrange(Ny):
            I[i] = simps(y[:, i], ylin)  # I changed the integral
        # then an integral over the result
        F = simps(I, xlin)
        return F 
    x = np.linspace(xlims[0], xlims[1], zlims)
    y = np.linspace(ylims[0], ylims[1], zlims)
    chrg_x, chrg_y = np.mgrid[xlims[0]:xlims[1]:np.complex(0,zlims), 
                              xlims[0]:ylims[1]:np.complex(0,zlims)]
    sigma = 1.0
    h = 50.
    pots = np.zeros(len(ele_xx))
    csd = csd_profile(chrg_x, chrg_y, np.zeros(len(chrg_y))) 
    for ii in range(len(ele_xx)):
        pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii], x, y, csd, h)
    pots /= 2*np.pi*sigma
    return pots
    
def generate_electrodes(dim='2D', xlims=[0.1,0.9], ylims=[0.1,0.9], res=5):
    '''Generates electrodes'''
    if dim == '2D':
        ele_x, ele_y = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                ylims[0]:ylims[1]:np.complex(0,res)]
        ele_x = ele_x.flatten()
        ele_y = ele_y.flatten()
        return ele_x, ele_y
    elif dim == '3D' or dim == '1D':
        print 'Not implemented for this case'
        return None

def large_source_2D(x,y,z=0):
    '''Same as 'large source' profile in 2012 paper'''
    zz = [0.4, -0.3, -0.1, 0.6] 
    zs = [0.2, 0.3, 0.4, 0.2] 
    f1 = 0.5965*exp( (-1*(x-0.1350)**2 - (y-0.8628)**2) /0.4464)* exp(-(z-zz[0])**2 / zs[0]) /exp(-(zz[0])**2/zs[0])
    f2 = -0.9269*exp( (-2*(x-0.1848)**2 - (y-0.0897)**2) /0.2046)* exp(-(z-zz[1])**2 / zs[1]) /exp(-(zz[1])**2/zs[1]);
    f3 = 0.5910*exp( (-3*(x-1.3189)**2 - (y-0.3522)**2) /0.2129)* exp(-(z-zz[2])**2 / zs[2]) /exp(-(zz[2])**2/zs[2]);
    f4 = -0.1963*exp( (-4*(x-1.3386)**2 - (y-0.5297)**2) /0.2507)* exp(-(z-zz[3])**2 / zs[3]) /exp(-(zz[3])**2/zs[3]);
    f = f1+f2+f3+f4
    return f

def small_source_2D(x,y,z=0):
    def gauss2d(x,y,p):
        """
         p:     list of parameters of the Gauss-function
                [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
                SIGMA = FWHM / (2*sqrt(2*log(2)))
                ANGLE = rotation of the X,Y direction of the Gaussian in radians

        Returns
        -------
        the value of the Gaussian described by the parameters p
        at position (x,y)
        """
        rcen_x = p[0] * np.cos(p[5]) - p[1] * np.sin(p[5])
        rcen_y = p[0] * np.sin(p[5]) + p[1] * np.cos(p[5])
        xp = x * np.cos(p[5]) - y * np.sin(p[5])
        yp = x * np.sin(p[5]) + y * np.cos(p[5])

        g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2+
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    f1 = gauss2d(x,y,[0.3,0.7,0.038,0.058,0.5,0.])
    f2 = gauss2d(x,y,[0.3,0.6,0.038,0.058,-0.5,0.])
    f3 = gauss2d(x,y,[0.45,0.7,0.038,0.058,0.5,0.])
    f4 = gauss2d(x,y,[0.45,0.6,0.038,0.058,-0.5,0.])
    f = f1+f2+f3+f4
    return f

if __name__ == '__main__':
    #tests
    #e < CS - csd

    xx_ele, yy_ele = generate_electrodes()
    pots = FWD(large_source_2D, xx_ele, yy_ele) 
    pots = np.reshape(pots, (-1,1)) #into a 2D - shady
    ele_pos = np.vstack((xx_ele, yy_ele)).T

    an_sigs=[]
    for ii in range(len(pots)):
        rc=neo.RecordingChannel()
        rc.coordinate=ele_pos[ii]*pq.mm
        asig=neo.AnalogSignal(pots[ii]*pq.mV,sampling_rate=1000*pq.Hz)
        rc.analogsignals=[asig]
        rc.create_relationship()
        an_sigs.append(asig)
   
    result = CSD(an_sigs)
    print result
    print result.t_start
    print result.sampling_rate
    print result.times
