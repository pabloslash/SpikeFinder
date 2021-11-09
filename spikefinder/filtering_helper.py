
# Imports
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, freqz
import numpy as np


'''
FILTER FUNCTIONS
'''


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# Build Butterworth filters with scipy.
def butter_filt_coefficients(fs, lowcut=[], highcut=[], btype='band', order=5):
    assert btype in ['band', 'low', 'high'], "Filter type must be 'low', 'high', or 'band'"
    if btype == 'low': assert lowcut, "Low cut frequency must be specified to build lowpass filter"
    elif btype == 'high': assert highcut, "High cut frequency must be specified to build a high filter"
    elif btype == 'low': assert lowcut, "Low and High cut frequencies must be specified to build a band filter"

    nyq = 0.5 * fs
    if lowcut: low = lowcut / nyq
    if highcut: high = highcut / nyq

    a, b = [], []
    if btype == 'band':
        b, a = butter(order, [low, high], btype)
    elif btype == 'low':
        b, a = butter(order, low, btype)
    elif btype == 'high':
        b, a = butter(order, high, btype)
    return b, a


# Load Butterworth filter coefficients from a file
def load_filter_coefficients_matlab(filter_file_path):
    coefficients = loadmat(filter_file_path)
    a = coefficients['a'][0]
    b = coefficients['b'][0]
    return b, a  # The output is a double list after loading .mat file


# Filter non-causally (forward & backwards) given filter coefficients
def noncausal_filter_1d(signal, b, a=1):
    y = filtfilt(b, a, signal)
    return y


def discard_channels_3d(neural_data, bird):
    """"
     Function for 'epoched data'.
     Deletes bad channels from a 3D array [epochs x channels x samples].

     Input: [Epochs x Channels x Samples]
     bird = 'z007', 'z017' or 'z020' have pre-selected bad channels

     Output: [ Epochs x (Channels - Bad_Channels) x Samples]
    """
    
    assert bird in ['z007', 'z017', 'z020'], "Bird must be 'z007', 'z017' or 'z020'"
    clean_neural_data = []
    for ep in range(len(neural_data)):
        clean_neural_data.append(discard_channels_2d(neural_data[ep], bird))
        
    return clean_neural_data

def discard_channels_2d(neural_data, bird):
    """"
     Deletes bad channels from a 2D array [channels x samples].

     Input: [Channels x Samples]
     bird = 'z007', 'z017' or 'z020' have pre-selected bad channels

     Output: [ (Channels - Bad_Channels) x Samples]
    """
    
    assert bird in ['z007', 'z017', 'z020'], "Bird must be 'z007', 'z017' or 'z020'"

    bad_channels = {'z007': [0,24], 
                    'z017': [], 
                    'z020': [1]}
    
    print('Deleting channels {}'.format(bad_channels[bird]))
    
    return np.delete(neural_data, bad_channels[bird], 0)
    
