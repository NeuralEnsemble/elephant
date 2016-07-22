import neo
import quantities as pq
import numpy as np

from scipy.integrate import simps 
from numpy import exp
from current_source_density import KCSD

available_1d = ['KCSD1D']
available_2d = ['KCSD2D', 'MoIKCSD']
available_3d = ['KCSD3D']
all_kernel_methods = ['KCSD1D', 'KCSD2D', 'KCSD3D', 'MoIKCSD']

def CSD(analog_signals, coords=None, method=None, params={}, cv_params={}):
    if coords == None:
        coords = []
        for ii in analog_signals:
            coords.append(ii.recordingchannel.coordinate)
    if method == None:
        raise ValueError('Must specify a method of CSD implementation')
    if len(coords) != len(analog_signals):
        raise ValueError('Number of signals and coords is not same')
    for ii in coords: # CHECK for Dimensionality of electrodes
        if len(ii) > 3:
            raise ValueError('Invalid number of coordinate positions')
    dim = len(coords[0])
    #print 'Dimensionality of the electrode setup is: ', dim
    if dim==1 and (method not in available_1d):
        raise ValueError('Invalid method, Available options are:', available_1d)
    if dim==2 and (method not in available_2d):
        raise ValueError('Invalid method, Available options are:', available_2d)
    if dim==3 and (method not in available_3d):
        raise ValueError('Invalid method, Available options are:', available_3d)

    input_array = np.zeros((len(analog_signals),analog_signals[0].magnitude.shape[0]))
    for ii,jj in enumerate(analog_signals):
        input_array[ii,:] = jj.magnitude
    
    if method in all_kernel_methods:
        kernel_method = getattr(KCSD, method) #fetch the class 'KCSD1D'
        k = kernel_method(np.array(coords), input_array, **params)
        if (method in all_kernel_methods) and bool(cv_params): #not empty then
            #print 'Performing Cross Validation'
            if len(cv_params.keys() and ['Rs', 'lambdas']) != 2:
                raise TypeError('Invalid cv_params argument passed')
            k.cross_validate(**cv_params)
        estm_csd = k.values()
        estm_csd = np.rollaxis(estm_csd, -1, 0)
        output= neo.AnalogSignalArray(estm_csd*pq.uA/pq.mm**dim,
                                      t_start=analog_signals[0].t_start,
                                      sampling_rate=analog_signals[0].sampling_rate)
        if dim == 1:
            output.annotate(x_coords=k.estm_x)
        elif dim == 2:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y)
        elif dim == 3:
            output.annotate(x_coords=k.estm_x, y_coords=k.estm_y, z_coords=k.estm_z)
    return output

def FWD(csd_profile, ele_xx, ele_yy=None, ele_zz=None, xlims=[0.,1.], ylims=[0.,1.], zlims=[0.,1.], res=50):
    '''Forward modelling for the getting the potentials'''
    def integrate_1D(x0, csd_x, csd, h):
        m = np.sqrt((csd_x-x0)**2 + h**2) - abs(csd_x-x0)
        y = csd * m 
        I = simps(y, csd_x)
        return I

    def integrate_2D(x, y, xlin, ylin, csd, h):
        X, Y = np.meshgrid(xlin, ylin)
        Ny = ylin.shape[0]
        m = np.sqrt((x - X)**2 + (y - Y)**2)
        m[m < 0.0000001] = 0.0000001
        y = np.arcsinh(2*h / m) * csd 
        I = np.zeros(Ny)
        for i in xrange(Ny):
            I[i] = simps(y[:, i], ylin)
        F = simps(I, xlin)
        return F 
        
    def integrate_3D(x, y, z, xlim, ylim, zlim, csd, xlin, ylin, zlin, X, Y, Z):
        Nz = zlin.shape[0]
        Ny = ylin.shape[0]
        m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
        m[m < 0.0000001] = 0.0000001
        z = csd / m
        Iy = np.zeros(Ny)
        for j in xrange(Ny):
            Iz = np.zeros(Nz)                        
            for i in xrange(Nz):
                Iz[i] = simps(z[:,j,i], zlin)
            Iy[j] = simps(Iz, ylin)
        F = simps(Iy, xlin)
        return F
    dim = 1
    if ele_zz is not None:
        dim = 3
    elif ele_yy is not None:
        dim = 2
    x = np.linspace(xlims[0], xlims[1], res)
    if dim >= 2 :
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
        pots /= 2.*sigma #eq.: 26 from Potworowski et al
    elif dim == 2:
        chrg_x, chrg_y = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                  ylims[0]:ylims[1]:np.complex(0,res)]
        csd = csd_profile(chrg_x, chrg_y) 
        for ii in range(len(ele_xx)):
            pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii], x, y, csd, h)
        pots /= 2*np.pi*sigma
    elif dim == 3:
        chrg_x, chrg_y, chrg_z = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                          ylims[0]:ylims[1]:np.complex(0,res),
                                          zlims[0]:zlims[1]:np.complex(0,res)]
        csd = csd_profile(chrg_x, chrg_y, chrg_z)
        xlin = chrg_x[:,0,0]
        ylin = chrg_y[0,:,0]
        zlin = chrg_z[0,0,:]
        for ii in range(len(ele_xx)):
            pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                    xlims, ylims, zlims, csd, 
                                    xlin, ylin, zlin, 
                                    chrg_x, chrg_y, chrg_z)
        pots /= 4*np.pi*sigma
    return pots

def generate_electrodes(dim, xlims=[0.1,0.9], ylims=[0.1,0.9], zlims=[0.1,0.9], res=5):
    '''Generates electrodes'''
    if dim == 1:
        ele_x = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res)]
        ele_x = ele_x.flatten()
        return ele_x
    elif dim == 2:
        ele_x, ele_y = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                ylims[0]:ylims[1]:np.complex(0,res)]
        ele_x = ele_x.flatten()
        ele_y = ele_y.flatten()
        return ele_x, ele_y
    elif dim == 3:
        ele_x, ele_y, ele_z = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                       ylims[0]:ylims[1]:np.complex(0,res),
                                       zlims[0]:zlims[1]:np.complex(0,res)]
        ele_x = ele_x.flatten()
        ele_y = ele_y.flatten()
        ele_z = ele_z.flatten()
        return ele_x, ele_y, ele_z

def gauss_1d_dipole(x):
    src = 0.5*exp(-((x-0.7)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    snk = -0.5*exp(-((x-0.3)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    f = src+snk
    return f

def large_source_2D(x, y):
    zz = [0.4, -0.3, -0.1, 0.6] 
    zs = [0.2, 0.3, 0.4, 0.2] 
    f1 = 0.5965*exp( (-1*(x-0.1350)**2 - (y-0.8628)**2) /0.4464)* exp(-(-zz[0])**2 / zs[0]) /exp(-(zz[0])**2/zs[0])
    f2 = -0.9269*exp( (-2*(x-0.1848)**2 - (y-0.0897)**2) /0.2046)* exp(-(-zz[1])**2 / zs[1]) /exp(-(zz[1])**2/zs[1]);
    f3 = 0.5910*exp( (-3*(x-1.3189)**2 - (y-0.3522)**2) /0.2129)* exp(-(-zz[2])**2 / zs[2]) /exp(-(zz[2])**2/zs[2]);
    f4 = -0.1963*exp( (-4*(x-1.3386)**2 - (y-0.5297)**2) /0.2507)* exp(-(-zz[3])**2 / zs[3]) /exp(-(zz[3])**2/zs[3]);
    f = f1+f2+f3+f4
    return f

def small_source_2D(x, y):
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
    x0, y0, z0 = 0.3, 0.7, 0.3
    x1, y1, z1 = 0.6, 0.5, 0.7
    sig_2 = 0.023
    A = (2*np.pi*sig_2)**-1
    f1 = A*exp( (-(x-x0)**2 -(y-y0)**2 -(z-z0)**2) / (2*sig_2) )
    f2 = -1*A*exp( (-(x-x1)**2 -(y-y1)**2 -(z-z1)**2) / (2*sig_2) )
    f = f1+f2
    return f
    
if __name__ == '__main__':
    #tests
    #e < CS - csd
    dim = 1
    if dim==1 :
        ele_pos = generate_electrodes(dim=1).reshape(5,1)
        pots = FWD(gauss_1d_dipole, ele_pos) 
        pots = np.reshape(pots, (-1,1))
        test_method = 'KCSD1D'
        test_params = {'h':50.}
    elif dim==2:
        xx_ele, yy_ele = generate_electrodes(dim=2)
        pots = FWD(large_source_2D, xx_ele, yy_ele) 
        pots = np.reshape(pots, (-1,1)) #into a 2D - shady
        ele_pos = np.vstack((xx_ele, yy_ele)).T
        test_method = 'KCSD2D'
        test_params = {'sigma':1.}
    elif dim==3:
        xx_ele, yy_ele, zz_ele = generate_electrodes(dim=3, res=3)
        pots = FWD(gauss_3d_dipole, xx_ele, yy_ele, zz_ele) 
        pots = np.reshape(pots, (-1,1))
        ele_pos = np.vstack((xx_ele, yy_ele, zz_ele)).T
        test_method = 'KCSD3D'
        test_params = {'gdx':0.1, 'gdy':0.1, 'gdz':0.1, 'src_type':'step'}
        
    an_sigs=[]
    for ii in range(len(pots)):
        rc = neo.RecordingChannel()
        rc.coordinate = ele_pos[ii]*pq.mm
        asig = neo.AnalogSignal(pots[ii]*pq.mV,sampling_rate=1000*pq.Hz)
        rc.analogsignals = [asig]
        rc.create_relationship()
        an_sigs.append(asig)
    result = CSD(an_sigs, method=test_method, params=test_params, cv_params={'Rs':np.array((0.1,0.25,0.5))})

    # from matplotlib import pyplot as plt
    # plt.plot(result.annotations['x_coords'], result)
    # plt.show()


    print result
    print result.t_start
    print result.sampling_rate
    print result.times
