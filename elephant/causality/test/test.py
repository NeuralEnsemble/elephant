import numpy as np
from scipy.signal import csd as csd
from elephant.spectral import segmented_multitaper_cross_spectrum, multitaper_psd
from elephant.causality.granger import _spectral_factorization, pairwise_spectral_granger, _dagger
from matplotlib import pyplot as plt
from scipy.signal import csd
'''
Test spectral factorization
'''

# Generate data
np.random.seed(12321)
length_2d = 2**12
signal = np.zeros((2, length_2d))

order = 2
weights_1 = np.array([[0.9, 0], [0.16, 0.8]]).T
weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]]).T

weights = np.stack((weights_1, weights_2))

noise_covariance = np.array([[1., 0.4], [0.4, 0.7]])

for i in range(length_2d):
    for lag in range(order):
        signal[:, i] += np.dot(weights[lag],
                               signal[:, i - lag - 1])
    rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
    signal[0, i] += rnd_var[0]
    signal[1, i] += rnd_var[1]

signal = signal[:, 100:]
length_2d -= 100

x = signal[0]
y = signal[1]


# Estimate power spectra
f, psd_1 = multitaper_psd(x, num_tapers=5, len_segment=int(2**9))
f, psd_2 = multitaper_psd(y, num_tapers=5, len_segment=int(2**9))

# Estimate cross spectra
_,  cross_spectrum = segmented_multitaper_cross_spectrum(signal,
                                                         len_segment=int(2**9),
                                                         num_tapers=5,
                                                         return_onesided=True)

# Apply spectral factorization
cov_matrix, transfer_function = _spectral_factorization(cross_spectrum,
                                                        num_iterations=100)

# Get cross spectrum back from spectral factorization
cross_spec_fact = np.matmul(np.matmul(transfer_function, cov_matrix),
                            _dagger(transfer_function))


n = length_2d

# First compare power spectra
plt.figure()
plt.title('Power spectra')
plt.plot(f, psd_1/2, label='True PSD X / 2')
plt.plot(f, cross_spectrum[0,0, :(n+2)//2], label='True PSD Cross Spec X')
plt.plot(f, cross_spec_fact[:(n+2)//2, 0, 0], label='Dec PSD X')

plt.plot(f, psd_2/2, label='True PSD Y / 2')
plt.plot(f, cross_spectrum[1,1,:(n+2)//2], label='True PSD Y')
plt.plot(f, cross_spec_fact[:(n+2)//2, 1, 1], label='Dec PSD Y')
plt.legend()
plt.show()

plt.show()

# Next compare cross spectr
fcs, cs_xy = csd(x, y, nperseg=int(2**7))
fcs, cs_yx = csd(y, x, nperseg=int(2**7))

plt.figure()
plt.plot(f, np.real(cross_spectrum[0, 1, :(n+2)//2]),
         label='True Cross spec xy real')
plt.plot(f, np.real(cross_spec_fact[:(n+2)//2, 0, 1]),
         label='Dec Cross spec xy real')
plt.plot(f, np.imag(cross_spectrum[0,1, :(n+2)//2]),
         label='True Cross spec xy real')
plt.plot(f, np.imag(cross_spec_fact[:(n+2)//2, 0, 1]),
         label='Dec Cross spec xy real')
plt.plot(fcs, cs_xy.real / 2, label='True Cross spec xy scipy real / 2')
plt.plot(fcs, cs_xy.imag / 2, label='True Cross spec xy scipy imag / 2')
plt.legend()
plt.show()

plt.figure()
plt.plot(f, np.real(cross_spectrum[1, 0, :(n+2)//2]),
         label='True Cross spec yx real')
plt.plot(f, np.real(cross_spec_fact[:(n+2)//2, 1, 0]),
         label='Dec Cross spec yx real')
plt.plot(f, np.imag(cross_spectrum[1,0, :(n+2)//2]),
         label='True Cross spec yx real')
plt.plot(f, np.imag(cross_spec_fact[:(n+2)//2, 1, 0]),
         label='Dec Cross spec yx real')
plt.plot(fcs, cs_yx.real / 2, label='True Cross spec yx scipy real / 2')
plt.plot(fcs, cs_yx.imag / 2, label='True Cross spec yx scipy imag / 2')
plt.legend()
plt.show()


'''
Test spectral granger
'''
length_2d = 2**16
signal = np.zeros((2, length_2d))

order = 2
weights_1 = np.array([[0.9, 0], [0.16, 0.8]])
weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

weights = np.stack((weights_1, weights_2))

noise_covariance = np.array([[1., 0.4], [0.4, 0.7]])
for i in range(length_2d):
    for lag in range(order):
        signal[:, i] += np.dot(weights[lag],
                               signal[:, i - lag - 1])
    rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
    signal[0, i] += rnd_var[0]
    signal[1, i] += rnd_var[1]


# from elephant.test.test_causality import ConditionalGrangerTestCase
# signal3 = ConditionalGrangerTestCase._generate_ground_truth(length_2d=2**10, causality_type='indirect').T
# temp_signal = signal
# signal = signal3[:2, :]

#np.save('/home/jurkus/repositories/r_spectral_granger/signal.npy', signal)
freq, cross_spectrum = segmented_multitaper_cross_spectrum(signal, nw=5, return_onesided=True)

# from spectral_connectivity import Connectivity, Multitaper


# Good choices
# length_2d=2 ** 16; len_segment=2**8, num_tapers=25
# length_2d=2 ** 14; len_segment=2**8, num_tapers=25
# length_2d=2 ** 16; frequency_resolution=0.01
f, spectral_causality = pairwise_spectral_granger(signal[0], signal[1],
                                                  len_segment=2**8,
                                                  num_tapers=4, fs=1, num_iterations=50)
# f, spectral_causality = pairwise_spectral_granger(signal[0], signal[1], frequency_resolution=0.01, fs=1, num_iterations=50)

# Spectral connectivity package

##connectivity = Connectivity(
##    fourier_coefficients=multitaper.fft(),
##    expectation_type="tapers",
##    frequencies=multitaper.frequencies,
##    time=multitaper.time,
##    blocks=1,)
#fcs, cross_spectrum = segmented_multitaper_cross_spectrum(signal,
#                                                     len_segment=2**9,
#                                                     nw=5,
#                                                     return_onesided=True)
#cov_matrix, transfer_function = _spectral_factorization(cross_spectrum,
#                                                        num_iterations=100)

#c_lfp = Connectivity.from_multitaper(multitaper)

fn = np.arange(0, np.pi , 0.01)



# Theoretical prediction for granger causality
freqs_for_theo = np.array([1, 2])[:, np.newaxis] * fn
A_theo = (np.identity(2)[np.newaxis]
          - weights_1 * np.exp(
              - 1j * freqs_for_theo[0][:, np.newaxis, np.newaxis]))
A_theo -= weights_2 * np.exp(
    - 1j * freqs_for_theo[1][:, np.newaxis, np.newaxis])

H_theo = np.array([[A_theo[:, 1, 1], -A_theo[:, 0, 1]],
                  [-A_theo[:, 1, 0], A_theo[:, 0, 0]]])
H_theo /= np.linalg.det(A_theo)
H_theo = np.moveaxis(H_theo, 2, 0)

S_theo = np.matmul(np.matmul(H_theo, noise_covariance), _dagger(H_theo))

H_tilde_xx = (H_theo[:, 0, 0]
              + noise_covariance[0, 1]/noise_covariance[0, 0]*H_theo[:, 0,
                                                                     1])
H_tilde_yy = (H_theo[:, 1, 1]
              + noise_covariance[0, 1]/noise_covariance[1, 1]*H_theo[:, 1,
                                                                     0])

directional_causality_y_x = np.log(S_theo[:, 0, 0].real /
                                   (H_tilde_xx
                                    * noise_covariance[0, 0]
                                    * H_tilde_xx.conj()).real)

directional_causality_x_y = np.log(S_theo[:, 1, 1].real /
                                   (H_tilde_yy
                                    * noise_covariance[1, 1]
                                    * H_tilde_yy.conj()).real)

instantaneous_causality = np.log(
    (H_tilde_xx * noise_covariance[0, 0] * H_tilde_xx.conj()).real
    * (H_tilde_yy * noise_covariance[1, 1] * H_tilde_yy.conj()).real)
instantaneous_causality -= np.linalg.slogdet(S_theo)[1]

from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(fn, directional_causality_x_y, label='x->y')
# plt.plot(fn, directional_causality_y_x, label='y->x')
# plt.plot(fn, instantaneous_causality, label='inst')
# plt.legend()
# plt.title('Theoretical prediction')
# plt.show()
#
# from matplotlib import pyplot as plt
#
# plt.figure()
# plt.plot(f, spectral_causality[0], label='x->y')
# plt.plot(f, spectral_causality[1], label='y->x')
# plt.plot(f, spectral_causality[2], label='inst')
# plt.plot(f, spectral_causality[3], label='tot')
# plt.legend()
# plt.title('Granger (Elephant) estimate')
# plt.show()
plt.figure()
plt.plot(fn * f.max() / fn.max(), directional_causality_x_y, 'r:', label='Theoretical x->y')
plt.plot(fn * f.max() / fn.max(), directional_causality_y_x, 'b:', label='Theoretical y->x')
plt.plot(fn * f.max() / fn.max(), instantaneous_causality, 'k:', label='Theoretical inst')
plt.plot(f, spectral_causality[0], 'r', label='Estimated x->y')
plt.plot(f, spectral_causality[1], 'b', label='Estimated y->x')
plt.plot(f, spectral_causality[2], 'k', label='Estimated inst')
plt.legend()
plt.title('Granger (Elephant) vs Theoretical estimate')
plt.show()


from elephant.spectral import multitaper_coherence
freqs, coh, phase_lag = multitaper_coherence(signal[0, :], signal[1, :], len_segment=2**8, num_tapers=25, fs=1,)

tot = spectral_causality[3]
der_tot = -np.log(1 - coh)[:len(tot)]
print(f"Max error of total interdependence values: "
      f"{(tot - der_tot)[np.argmax(np.abs(tot - der_tot))]}")

# Signal used: length_2d = 2 ** 14
#gc_mat = np.load('/home/jurkus/repositories/r_spectral_granger/gc_matrix.npy')

#fig = plt.figure()
#ax = fig.gca()
#plt.plot(fn * f.max() / fn.max(), directional_causality_x_y, 'r--', label='Theoretical: x->y')
#plt.plot(fn * f.max() / fn.max(), directional_causality_y_x, 'g--', label='Theoretical: y->x')
#plt.plot(f, spectral_causality[0], 'r', label='Elephant: x->y')
#plt.plot(f, spectral_causality[1], 'g', label='Elephant: y->x')
#plt.plot(gc_mat[:, 0], gc_mat[:, 1], 'r:', label='R: x->y')
#plt.plot(gc_mat[:, 0], gc_mat[:, 2], 'g:', label='R: y->x')
#info_str = f'length_2d={length_2d}\nlen_segment=2**8\nnum_tapers=25'
#plt.text(0, 1, info_str, ha='left', va='top', transform=ax.transAxes)
#plt.legend()
#plt.title('Theoretical vs Elephant vs R')
#plt.show()
