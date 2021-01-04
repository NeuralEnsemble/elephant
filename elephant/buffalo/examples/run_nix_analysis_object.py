"""
This example shows basic functionality of the Buffalo analysis objects,
and compatibility with existing code that uses Elephant functions.
"""

import numpy as np
import quantities as pq
import neo
import elephant.buffalo.objects as buf_obj

import matplotlib.pyplot as plt



def main(sampling_rate=30000*pq.Hz, sampling_time=10*pq.s):

    n_samples = (sampling_rate * sampling_time).simplified.magnitude.item()
    n_samples = int(n_samples)

    # Generates analog data
    signal_array = np.random.normal(0, 5, n_samples)

    sampling_period = (1/sampling_rate).simplified

    test = buf_obj.nix.base.BuffaloAnalogSignal(signal_array,
                                                units=pq.mV,
                                                sampling_period=sampling_period,
                                                test=True)
    print(isinstance(test, neo.AnalogSignal))

    print("\n\nInitial data")
    print(test[:5], test.annotations)

    test[0:2] = [[0.06], [0.05]] * pq.V
    test[3] = 4 * pq.uV

    print("\n\nAfter setting data")
    print(test[:5], test.annotations)

    test.annotations['test'] = False

    print("\n\nAfter setting annotations")
    print(test[:5], test.annotations)

    nix_ticks = np.array(test.dimensions[0].axis(len(test)))

    print("\n\nTime axis information")
    print(test.times[:5], test.sampling_period)
    print(np.all(test.times.magnitude == nix_ticks))

    non_matched = np.where(test.times.magnitude != nix_ticks)
    print(test.times[non_matched])
    print(nix_ticks[non_matched])
    print(len(non_matched[0]))

    plt.plot(test.dimensions[0].axis(len(test)), test)
    plt.xlabel(f"{test.dimensions[0].label} ({test.dimensions[0].unit})")
    plt.ylabel(test.unit)
    plt.title(test.name)
    plt.show()


if __name__ == "__main__":
    main()
