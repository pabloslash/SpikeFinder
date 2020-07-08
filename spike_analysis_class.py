
# Imports
import numpy as np
import filtering_helper

'''
THRESHOLD CROSSINGS
'''


def find_threshold_crossings_1d(neural_channel, fs, th=3.5, ap_time=1):
    """"
     Converts a 1D-list (1 channel) of raw extracellurar continuous neural recordings into a binary matrix indicating
       where threshold crossings occur.

     Input: [1 x Samples]

     fs = Sampling frequency of input data
     th = Number of standard deviations below RMS of each channel considered to trigger an AP
     ap_time (ms) = 'Artificial' refractory period. Time between 2 found AP to consider them independent.
       Typycal depolarization time is ~1ms.

     Output: [1 x Samples]
    """
    samples = len(neural_channel)
    th_crossings = np.zeros(samples)

    # Find RMS and threshold idxs
    rms = np.sqrt(np.mean([i ** 2 for i in neural_channel]))
    th_idx = [idx for idx, val in enumerate(neural_channel) if val < -th * rms]

    # Delete idx if they belong to the same AP
    ap_samples = int(ap_time / 1000 * fs)

    clean_th_idx = []
    last_idx = -ap_samples  # For first AP

    for i in th_idx:
        if i > last_idx + ap_samples:
            clean_th_idx.append(i)
            last_idx = i

    th_crossings[clean_th_idx] = 1

    return th_crossings


def find_threshold_crossings_2d(neural_data, fs, th=3.5, ap_time=1, verbose=False):
    """"
     Converts a 2D-list (n-channels) of raw extracellurar continuous neural recordings into a binary matrix indicating
       where threshold crossings occur.

     Input: [Channels x Samples]

     fs = Sampling frequency of input data
     th = Number of standard deviations below RMS of each channel considered to trigger an AP
     ap_time (ms) = 'Artificial' refractory period. Time between 2 found AP to consider them independent.
       Typycal depolarization time is ~1ms.

     Output: [Channels x Samples]
    """
    channels = len(neural_data)
    samples = len(neural_data[0])
    spike_raster = np.zeros([channels, samples])

    # Populate spike raster matrix
    for ch in range(channels):
        if verbose: print('Finding threshold crossings in channel {}'.format(ch))
        spike_raster[ch] = find_threshold_crossings_1d(neural_data[ch], fs, th=th, ap_time=ap_time)

    return spike_raster


# Return binned data according to bin size (1D or 2D)
# Input: 1d or 2D list
def bin_matrix(dat, bin_size):
    if len(np.array(dat).shape) > 1:
        bin_mat = []
        for i in range(np.array(dat).shape[0]):
            bin_mat.append(
                np.sum(np.array(dat[i][:(len(dat[i]) // bin_samples) * bin_samples]).reshape(-1, bin_samples),
                       axis=1).tolist())
        return bin_mat
    else:
        return np.sum(np.array(dat[:(len(dat) // bin_samples) * bin_samples]).reshape(-1, bin_samples), axis=1).tolist()