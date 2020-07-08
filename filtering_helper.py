
# Imports
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, freqz


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
