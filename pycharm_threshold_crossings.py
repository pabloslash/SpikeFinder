"""
This Module contains a Class used for importing Chunk'd Behavioral and Neural Data and its relevant meta data
"""

import matplotlib
matplotlib.use('Agg')
from BirdSongToolbox.config.utils import update_config_path, get_spec_config_path
from BirdSongToolbox.import_data import ImportData

from BirdSongToolbox.config.settings import CHUNKED_DATA_PATH
from BirdSongToolbox.file_utility_functions import _load_pckl_data,_load_numpy_data, _load_json_data

from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, freqz
import numpy as np
import os
import random

'''
PLOTTING FUNCTIONS
'''


# PLOT AUDIO SAMPLE
def plot_audio_sample(data, fs, save=False, bird='', day='', chunk='', savepath='', showfig=True):
    plt.figure()
    plt.plot(np.linspace(0, len(data), len(data)) / fs,
             data)
    plt.title('Sample Audio - Chunk {} in bird {}, sess {}'.format(chunk, bird, day))
    plt.xlabel('Time (s) - Sampled @ 30kHz')
    plt.ylabel('Amplitude (a.u)')
    if save: plt.savefig(savepath + 'audio_sample_' + bird + '_' + day + '_chunk' + str(chunk) + '.jpeg')  # Save figure
    if showfig: plt.show(block=False)


# PLOT NEURAL CHANNEL
def plot_neural_sample(data, fs, data_filt=None, th=None, time_secs=1, save=False, bird=None,
                       day=None, chunk=None, channel=None, savepath=''):

    samples = int(time_secs*fs)
    plt.figure()
    plt.plot(np.linspace(0, len(data[0:samples]), len(data[0:samples])) / fs,
             data[0:samples], label='Neural signal')
    if data_filt is not None:  # If there is any filtered signal to plot along with the raw signal
        plt.plot(np.linspace(0, len(data_filt[0:samples]), len(data_filt[0:samples])) / fs,
                 data_filt[0:samples], label='Filtered signal')
    if th is not None:  # If there is any threshold to plot along with the data
        plt.plot(np.linspace(0, len(data_filt[0:samples]), len(data_filt[0:samples])) / fs,
                 [th]*len(data_filt[0:samples]), color='r', label='Threshold')
    plt.title('Neural Data - Channel {}, Chunk {}, bird {}, sess {}'.format(channel, chunk, bird, day))
    plt.xlabel('Time (s) - Sampled @ 30kHz')
    plt.ylabel('Amplitude (a.u)')
    plt.legend()
    plt.grid(True)
    plt.axis('tight')
    if save: plt.savefig(savepath + 'Neural_sample_' + bird + '_' + day + '_chunk' + str(chunk) + '.jpeg')  # Save figure
    plt.show(block=False)


# PLOT BUTTERWORTH FILTER FREQ RESPONSE
def plot_filter_freq_response(fs, lowcut=None, highcut=None, btype='band'):
    # Plot the frequency response for a few different orders.
    plt.figure()
    plt.clf()
    for order in [3, 6, 9, 10]:
        b, a = butter_filt(fs, lowcut=lowcut, highcut=highcut, btype=btype, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.title('Frequency Response of Butterworth ' + btype + 'pass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show(block=False)


'''
BUTTERWORTH FILTER FUNCTIONS
'''


# Design Butterworth Filter with Scipy
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_filt(fs, lowcut=None, highcut=None, btype='band', order=5):
    assert btype in ['band', 'low', 'high'], "Filter type must be 'low', 'high', or 'band'"
    nyq = 0.5 * fs
    if lowcut: low = lowcut / nyq
    if highcut: high = highcut / nyq

    if btype == 'band':
        b, a = butter(order, [low, high], btype)
    elif btype == 'low':
        b, a = butter(order, low, btype)
    elif btype == 'high':
        b, a = butter(order, high, btype)
    return b, a


# Filter non-causally (i.e. avoid phase distortion)
def noncausal_filter(data, b, a=1):
    y = filtfilt(b, a, data)
    return y


# Load Filter coefficients designed somewhere else (e.g. Matlab)
def load_filter_coefficients(file_path):
    coefficients = loadmat(file_path)
    a = coefficients['a'][0]  # bc arrays from Matlab are imported as an array of arays
    b = coefficients['b'][0]
    return b, a


'''''''''''''''''''''
MAIN
'''''''''''''''''''''

###################################
# Birds & Sessions
###################################
z020_days = ['day-2016-06-03', 'day-2016-06-05']
z007_days = ['day-2016-09-09', 'day-2016-09-10', 'day-2016-09-11','day-2016-09-13']
z017_days = ['day-2016-06-21']


birds = ['z007', 'z017', 'z020']

birds_sess = {'z007': ['day-2016-09-09', 'day-2016-09-10', 'day-2016-09-11', 'day-2016-09-13'],
              'z017': ['day-2016-06-21'], 'z020': ['day-2016-06-03', 'day-2016-06-05']}

# Desired filter to load.
fs = 30000  # Sampling freq
filter_folder = 'filters/'
# filter_file = 'butter_bp_500hz-6000hz_order4.mat'
filter_file = 'butter_bp_250hz-8000hz_order4.mat'
b, a = load_filter_coefficients(filter_folder + filter_file)

# for bird in birds:

bird = 'z007'
for sess in birds_sess[bird]:

    ###################################
    # IMPORT DATA
    ###################################
    # Bird & Session of Interest
    bird_id = bird
    session = sess

    z_data_chunk = ImportData(bird_id=bird_id, session=session, data_type='Raw')

    print(z_data_chunk.bird_id, z_data_chunk.date)
    print('Number of chunks in session:', len(z_data_chunk.song_neural))

    for c in range (len(z_data_chunk.song_neural)):
        print('[Channels x Samples]:', z_data_chunk.song_neural[c].shape)

    ###################################
    # VARIABLES
    ###################################
    # Chunk & Channels of Interest
    chunks = random.sample(range(1, len(z_data_chunk.song_neural)), 3)  # generate 5 random chunks
    channels = random.sample(range(1, len(z_data_chunk.song_neural[0])), 3)  # generate 5 random channels

    #########################################
    # MAIN: FILTERING & THRESHOLD CROSSINGS
    #########################################
    savepath = 'images/' + str(bird_id) + '_' + str(session) + '/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    for chunk in chunks:
        for channel in channels:
            print('Chunk: ', chunk, 'Channel: ', channel)

            # Filter Channel
            neural = z_data_chunk.song_neural[chunk][channel]
            filt_neural = noncausal_filter(neural, b, a=a)

            # Substract mean signal
            filt_neural = filt_neural - np.mean(filt_neural)

            # Find RMS and set Threshold
            threshold = 3.5  # *RMS
            RMS = np.sqrt(np.mean([i ** 2 for i in filt_neural]))

            # Find threshold idxs
            th_idx = [idx for idx, val in enumerate(filt_neural) if val < -threshold * RMS]

            # Delete idx if they belong to the same AP
            AP_time = 1  # ms. Typycal depolarization time is ~3ms
            AP_samples = int(AP_time / 1000 * fs)

            clean_th_idx = []
            last_idx = -1000  # For first bin

            for i in th_idx:
                if i > last_idx + AP_samples:
                    clean_th_idx.append(i)
                    last_idx = i

            # Typical length of whole AP is ~1ms.
            # For each spike detected, take 1ms before and after (3ms total), and plot all snippets together

            t_before = 0.5  # ms
            t_AP = 0.5  # ms
            t_after = 0.5  # ms

            s_before, s_AP, s_after = int(t_before / 1000 * fs), int(t_AP / 1000 * fs), int(
                t_after / 1000 * fs)  # In samples

            # Delete spikes that do not have enough samples before / after to capture the whole depolarization
            clean_th_idx = np.array(clean_th_idx)
            clean_th_idx = clean_th_idx[
                np.logical_and(clean_th_idx > s_before, clean_th_idx < len(filt_neural) - s_AP - s_after)]

            # Take AP snippets
            AP_list = np.zeros((len(clean_th_idx), s_before + s_AP + s_after))

            for ap in range(0, len(clean_th_idx)):
                AP_list[ap] = filt_neural[clean_th_idx[ap] - s_before: clean_th_idx[ap] + s_AP + s_after]

            #########################################

            # Plot Threshold Crossings Snnipets
            # Plot all snippets
            plt.figure()
            plt.title('{} Spikes in chunk {}, channel {} - Bird {} sess {}'
                      .format(len(AP_list), chunk, channel, bird_id, session))
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude a.u')
            for ap in range(0, len(AP_list[0:100])):
                plt.plot(np.linspace(0, t_before+t_AP+t_after, s_before+s_AP+s_after), AP_list[ap])
            plt.savefig(savepath + 'threshold-Snippets_' + bird_id + '_' + session
                        + '_chunk' + str(chunk) + '_channel' + str(channel) + '.jpeg')   # Save figure

            # #########################################
            # # Plot Inter-Spike-interval Histogram
            # isi = np.diff(clean_th_idx)  # Find samples between spikes
            # isi = [int(i/fs*1000) for i in isi]  # ISI in Integer Miliseconds
            #
            # plt.figure()
            # plt.hist(isi, bins=1000)
            # plt.title('ISI - Channel {}, Chunk {}, bird {}, sess {}'.format(channel, chunk, bird_id, session))
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Spiking Count')
            # plt.xlim([0, 50])
            # plt.xticks(ticks=np.linspace(0, 50, 11))
            # plt.savefig(savepath + 'ISI_' + bird_id + '_' + session
            #                  + '_chunk' + str(chunk) + '_channel' + str(channel) + '.jpeg')   # Save figure

            # save = True
            # showfig = False
            #
            # # Plot Audio Sample
            # plot_audio_sample(z_data_chunk.song_audio[chunk], fs, save=save, bird=bird_id, day=session, chunk=chunk,
            #                   savepath=savepath, showfig=showfig)
            #
            # # # Plot Raw Neural Channel
            # # plot_raw_neural_sample(neural, fs, time_secs=0.5, save=save, bird=bird_id, day=session, chunk=chunk,
            # #                        channel=channel, savepath=savepath, showfig=showfig)
            #
            # # Plot Filtered Neural Channel
            # plot_neural_sample(neural, filt_neural, fs, time_secs=0.5, save=save, bird=bird_id, day=session,
            #                             chunk=chunk, channel=channel, savepath=savepath, showfig=showfig)


## Save Raster Plots

for sess in birds_sess[bird]:

    ###################################
    # IMPORT DATA
    ###################################
    # Bird & Session of Interest
    bird_id = bird
    session = sess

    z_data_chunk = ImportData(bird_id=bird_id, session=session, data_type='Raw')

    print(z_data_chunk.bird_id, z_data_chunk.date)
    print('Number of chunks in session:', len(z_data_chunk.song_neural))

    for c in range(len(z_data_chunk.song_neural)):
        print('[Channels x Samples]:', z_data_chunk.song_neural[c].shape)

    for chunk in range(len(z_data_chunk.song_neural)):
        # Create spike raster matrix
        neural_mat = z_data_chunk.song_neural[chunk]
        channels = len(neural_mat)
        samples = len(neural_mat[0])
        spike_raster = np.empty([channels, samples])
        spike_raster[:] = None
        spike_raster_plot = spike_raster

        # Populate spike raster matrix
        for ch in range(0, channels):
            neural = neural_mat[ch]

            # Substract mean signal
            filt_neural = noncausal_filter(neural, b, a=a)

            # Substract mean signal
            filt_neural = filt_neural - np.mean(filt_neural)

            # Find RMS and set Threshold
            threshold = 3.5  # *RMS
            RMS = np.sqrt(np.mean([i ** 2 for i in filt_neural]))

            # Plot Filtered Neural Channel
            save = True
            savepath = 'images/' + str(bird_id) + '_' + str(session) + '/'
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            plot_neural_sample(neural, fs, data_filt=filt_neural, th=-threshold * RMS, time_secs=0.5, save=save,
                               bird=bird_id, day=session, chunk=chunk, channel=channel, savepath=savepath)

            # Find threshold idxs
            th_idx = [idx for idx, val in enumerate(filt_neural) if val < -threshold * RMS]

            # Delete idx if they belong to the same AP
            AP_time = 1  # ms. Typycal depolarization time is ~3ms
            AP_samples = int(AP_time / 1000 * fs)

            clean_th_idx = []
            last_idx = -1000  # For first bin

            for i in th_idx:
                if i > last_idx + AP_samples:
                    clean_th_idx.append(i)
                    last_idx = i

            spike_raster[ch][clean_th_idx] = 0
            spike_raster_plot[ch][clean_th_idx] = 0 + ch

        # PLOT AND SAVE RASTER MAT

        savepath = 'raster_mats/' + str(bird_id) + '_' + str(session) + '/'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        savemat(savepath + 'RasterMat_' + bird_id + '_' + session + '_chunk' + str(chunk+1) + '.mat',
                    {'raster_mat': spike_raster, 'audio': z_data_chunk.song_audio[chunk]})

        # PLOT AND SAVE RASTER PLOT FIG

        savepath = 'images/' + str(bird_id) + '_' + str(session) + '/'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        init_plot = 180000
        end_plot = 195000
        # init_plot = 0
        # end_plot = len(z_data_chunk.song_audio[chunk])

        x_len = end_plot - init_plot
        x = np.linspace(init_plot, end_plot, num=end_plot - init_plot)

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
        ax1.plot(x / fs, z_data_chunk.song_audio[chunk][init_plot:end_plot])
        ax1.set_title('Sample Audio & Raster Plot - Chunk {} in bird {}, sess {}'.format(chunk, bird_id, session))
        ax1.set_ylabel('Amplitude (a.u)')
        ax1.set_xticks([])
        for channel in range(0, channels):
            ax2.scatter(x / fs, spike_raster_plot[channel][init_plot:end_plot], linewidths=0.2, marker=3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Channel #')
        plt.savefig(savepath + 'RasterPlot_' + bird_id + '_' + session
                    + '_chunk' + str(chunk+1) + '.jpeg')   # Save figure

