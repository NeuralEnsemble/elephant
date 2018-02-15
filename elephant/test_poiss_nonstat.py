import elephant.spike_train_generation as stg
import elephant.statistics as stat
import numpy as np
import neo
import matplotlib.pyplot as plt
import quantities as pq
import time

sampling_period = 0.001*pq.s
samples = 10000
cosine = np.cos(np.arange(5,10,0.001))+2
rate = neo.AnalogSignal((([2]*cosine).T)*pq.Hz, sampling_period=sampling_period)
t_0_thin = time.time()
sts_thin = stg.inhomogeneous_poisson_process(rate)
t_thin = time.time() - t_0_thin
rate_thin = stat.time_histogram(
    sts_thin,sampling_period)/(sampling_period.magnitude*np.array(samples))
# print(rate[-3:])
# print(rate[:3])
# print(rate_thin[-3:])
plt.plot(rate.times, rate)
plt.plot(rate.times, rate_thin, 'r')

