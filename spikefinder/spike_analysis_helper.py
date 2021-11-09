
# Imports
import numpy as np
# import filtering_helper

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


# Return binned vector according to bin size (Input: 1D list)
def downsample_list_1d(dat, number_bin_samples):
    """"
     Downsamples (bins) a 1D-list acording to selected number of bin samples.

     Input: [1 x Samples]
     number_bin_samples = Number of samples in each bin (bin size in samples).

     Output: [1 x Samples]
    """
    return np.sum(np.array(dat[:(len(dat) // number_bin_samples) * number_bin_samples]).reshape(-1, number_bin_samples), axis=1).tolist()
    
# Return binned matrix along dimension 2 according to bin size (Input: 2D list)
def downsample_list_2d(dat, number_bin_samples):
    """"
     Downsamples (bins) a 2D-list acording to selected number of bin samples.

     Input: [n x Samples]
     number_bin_samples = Number of samples in each bin (bin size in samples).

     Output: [n x Samples]
    """
    downsampled_dat = []
    for i in range(np.array(dat).shape[0]):
        downsampled_dat.append(downsample_list_1d(dat[i], number_bin_samples))

    return downsampled_dat
