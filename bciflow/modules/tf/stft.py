'''
stft.py

Description
-----------
This module contains the implementation of the Short-Time Fourier Transform (STFT) for EEG data. 
The STFT function transforms the input signals into the time-frequency domain.

Dependencies
------------
numpy, scipy, modules.core.eegdata, modules.utils.trial_transform

'''

import numpy as np
from scipy.signal import stft

def STFT(eegdata, nperseg=None, noverlap=None, nfft=None, window='hann', return_onesided=True, freqs_per_bands='auto'):
    
    X = eegdata['X'].copy()
    if X.shape[1] != 1:
        raise ValueError('The input data must have only one band.')
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
    
    if nperseg is None:
        nperseg = eegdata["sfreq"]

    if nfft is None:
        nfft = nperseg
    
    if noverlap is None:
        noverlap = nperseg-1
        
    if freqs_per_bands == 'auto':
        freqs_per_bands = 4

    total_bands = ((eegdata['sfreq']//2 + 1) //2) // freqs_per_bands + 1

    X_ = []
    for signal_ in range(X.shape[0]):
        frequencies, time_points, Zxx = stft(X[signal_], fs=eegdata['sfreq'], nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, return_onesided=return_onesided)
        Zxx = np.abs(Zxx).astype(float)
        if freqs_per_bands == 'auto':
            Zxx = [Zxx[2*i] + Zxx[2*i+1]/2 for i in range(len(Zxx)//2)]
        else:
            Zxx = [np.mean(Zxx[i:i+freqs_per_bands], axis=0) for i in range(0, len(Zxx)//2, freqs_per_bands)]
        X_.append(Zxx)
    X_ = np.array(X_)
    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape[0], eegdata['X'].shape[2], total_bands, eegdata['X'].shape[3]+1)
    X_ = np.transpose(X_, (0, 2, 1, 3))
    X_ = X_[:,:,:, 1:]
    eegdata['X'] = X_
    return eegdata
