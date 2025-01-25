'''
resample_fft.py

Description
-----------
This module contains the implementation of FFT-based resampling for EEG data. 
The fft_resample function uses the Fast Fourier Transform (FFT) to resample the input signals to a new sampling frequency.

Dependencies
------------
numpy, scipy, modules.core.eegdata, modules.utils.trial_transform

'''

import numpy as np
from scipy.signal import resample

def fft_resample(eegdata, new_sfreq):
    
    '''
    Resamples the input EEG data to a new sampling frequency using FFT.

    Parameters
    ----------
    X : np.ndarray
        Input EEG data. Expected shape is (n_trials, n_bands, n_electrodes, n_samples).
    sfreq : float
        Original sampling frequency of the input data.
    new_sfreq : float
        New sampling frequency to resample the data.

    Returns
    -------
    dict
        A dictionary containing the resampled data and the new sampling frequency.
        {'data': np.ndarray, 'sfreq': float}

    '''
    
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
    sfreq = eegdata['sfreq']
    divisor = sfreq//new_sfreq
    duration = X.shape[-1]/sfreq
    old_times = np.arange(0, duration, 1./sfreq)
    new_times = np.arange(0, duration, 1./new_sfreq)

    X_ = []
    for signal_ in range(X.shape[0]):
        new_signal = resample(X[signal_], int(new_times.shape[0]))
        X_.append(new_signal)

    X_ = np.array(X_)
    X_ = X_.reshape(*eegdata['X'].shape[:-1],eegdata['X'].shape[-1]//divisor )

    eegdata['X'] = X_
    eegdata['sfreq'] = new_sfreq

    return eegdata
