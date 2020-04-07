import neo
from neo.io.blackrockio_v4 import BlackrockIO
import numpy as np
import quantities as pq
from os.path import splitext
from scipy import signal
from multiprocessing import Pool

import matplotlib.pyplot as plt

session_file = "/home/koehler/datafiles/jazz/j200121-test-001/j200121-v-test-001.ns6"
thresh = 128 - 1

# session_file = "/home/koehler/datafiles/blackrock/realdata-67ch.ns6"
# thresh = 67 - 1


def load_data(inputfile, sortedfile=None, return_reader=False):
    """ given a Blackrock-file (.nev, .ns2, .ns6) returns either the loaded block or the data-reader class
    :param inputfile: full path including filename and extension of .nev file to load
    :param sortedfile: full path including filename and extension of sorted .nev file to load (optional)
    :return:
    """
    data_full_path, file_ending = splitext(inputfile)
    if file_ending == '.nev':
        nsx_to_load = 2
    elif file_ending == '.ns2':
        nsx_to_load = 2
    elif file_ending == '.ns6':
        nsx_to_load = 6
    else:
        raise ValueError('The inputfile is not of the following formats:.nev,.ns2 or .ns6')

    if sortedfile is None:
        data_io_class = BlackrockIO(filename=data_full_path)
    else:
        data_io_class = BlackrockIO(filename=data_full_path,
                                           nev_override=splitext(sortedfile)[0])

    if return_reader:
        return data_io_class
    block = data_io_class.read_block(load_waveforms=True, scaling='raw')
    return block


def load_and_filter_raw_signal(channel_id, anasigs,
                               lo_order=4, lo_corner=7500*pq.Hz,
                               hi_order=4, hi_corner=250*pq.Hz,
                               sampling_rate=30000*pq.Hz,
                               time_slice=None):

    # if time_slice is None:
    #     sig = anasigs.load(channel_indexes=[channel_id
    #                                         - 1]).magnitude.flatten().astype('float32')
    # else:
    #     sig = anasigs.load(channel_indexes=[channel_id - 1],
    #                        time_slice=time_slice, strict_slicing=False
    #                        ).magnitude.flatten().astype('float32')

    sig = anasigs[channel_id - 1].magnitude.flatten()
    # lowpass
    if lo_order is not None:
        b, a = signal.butter(N=int(lo_order),
                             Wn=(lo_corner*2./sampling_rate).simplified.magnitude.flatten()[0],
                             btype='lowpass', output='ba')
        sig = signal.filtfilt(b, a, sig).astype('float32')

    # highpass
    if hi_order is not None:
        b, a = signal.butter(N=int(hi_order),
                             Wn=(hi_corner*2./sampling_rate).simplified.magnitude.flatten()[0],
                             btype='highpass', output='ba')
        sig = signal.filtfilt(b, a, sig).astype('float32')

    return sig


if __name__ == '__main__':
    channel_ids = list(range(96))
    segment_to_load = -1
    num_threads = 4
    hi_order = None # 4
    hi_corner = None # pq.Quantity(45, 'Hz')
    lo_order = None
    lo_corner = None #pq.Quantity(10, 'Hz')
    sampling_rate = pq.Quantity(30000,
                                    'Hz')
    reader = load_data(session_file, return_reader=True)
    block = reader.read_block(lazy=False, nsx_to_load=6, scaling='raw',
                              channels=channel_ids)
    anasigs = block.segments[segment_to_load].analogsignals[0]

    t_start = pq.Quantity(5, 's')
    slice_start = t_start + anasigs.t_start
    slice_stop = pq.Quantity(15, 's') + anasigs.t_start

    time_slice = (slice_start, slice_stop)

    def pool_func(channel_id):
        return load_and_filter_raw_signal(channel_id, anasigs,
                                          sampling_rate=sampling_rate,
                                          time_slice=time_slice,
                                          lo_order=lo_order,
                                          lo_corner=lo_corner,
                                          hi_order=hi_order,
                                          hi_corner=hi_corner)


    p = Pool(num_threads)
    filtered_sigs = p.map(pool_func, channel_ids)
    filtered_sigs = np.array(filtered_sigs, dtype='int16')

    print(len(filtered_sigs))

    times = pq.Quantity(np.linspace(0, filtered_sigs.shape[1]/30000,
                        filtered_sigs.shape[1]), pq.s)
    times = times + t_start

    plt.plot(filtered_sigs)
    plt.show()
    exit()

    # V2-DP
    block_1 = [24,26,28,30,32,97,98,99,100,101,102,103,104,105,106,107,108,109,
               110,111,112,113,115,117,119,121,123,124,125,126,127,128,
               65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,
               86,87,88,89,90,91,92,93,94,96,114]

    # V1-A7
    block_2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,
               27,29,31,61,63,64,120,122,33,34,35,36,37,38,39,40,41,42,43,44,
               45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,95,116,118]

    block_1_ids = np.array(block_1) - 1
    block_2_ids = np.array(block_2) - 1

    plt.subplot(311)
    # block_1_ts = np.sum(filtered_sigs[block_1_ids[block_1_ids<=thresh],:], axis=0)
    block_1_ts = np.mean(filtered_sigs[block_1_ids[0:32], :],
                        axis=0)
    block_1_ts_2 = np.mean(filtered_sigs[block_1_ids[32:64], :],
                        axis=0)

    plt.plot(times, block_1_ts, label="V2")
    plt.plot(times, block_1_ts_2, label="DP")
    plt.legend()
    plt.title("V2-DP")

    plt.subplot(312)
    block_2_ts = np.mean(filtered_sigs[block_2_ids[0:32], :],
                        axis=0)
    block_2_ts_2 = np.mean(filtered_sigs[block_2_ids[32:64], :],
                        axis=0)

    plt.plot(times, block_2_ts, label="V1")
    plt.plot(times, block_2_ts_2, label="A7")
    plt.title("V1-A7")
    plt.legend()

    mixed = np.vstack((block_1_ids, block_2_ids)).flatten()

    plt.subplot(313)
    all_blocks = np.mean(filtered_sigs[mixed[mixed <= thresh],:], axis=0)
    # all_blocks = block_1_ts - block_2_ts
    plt.plot(times, all_blocks)
    plt.title("All")

    # fig2 = plt.figure()
    # ts = [block_1_ts, block_2_ts, all_blocks]
    # for idx, plot in enumerate(ts):
    #     plt.subplot(3,1,idx+1)
    #     f, Pxx_den = signal.periodogram(plot, 30000)
    #     plt.semilogy(f, Pxx_den)
    #     # plt.ylim([1e-7, 1e5])
    #     plt.xlim([0,100])
    #     plt.xlabel('frequency [Hz]')
    #     plt.ylabel('PSD [V**2/Hz]')
    #
    # def subplot(f, P, n):
    #     if n is not None:
    #         plt.subplot(2, 1, n)
    #     plt.semilogy(f, np.mean(P, axis=0))
    #     plt.xlim([0,100])
    #     plt.xlabel('frequency [Hz]')
    #     plt.ylabel('PSD [V**2/Hz]')


    # fig3 = plt.figure()
    # f, P = signal.periodogram(filtered_sigs[block_1_ids[0:32],:], 30000)
    # subplot(f, P, 1)
    # f, P = signal.periodogram(filtered_sigs[block_1_ids[32:64],:], 30000)
    # subplot(f, P, None)
    # f, P = signal.periodogram(filtered_sigs[mixed[mixed <= thresh],:], 30000)
    # subplot(f, P, 2)

    plt.show()
