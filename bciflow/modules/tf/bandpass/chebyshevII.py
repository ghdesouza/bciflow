

import numpy as np
from scipy.signal import cheby2, filtfilt

def chebyshevII(eegdata, low_cut=4, high_cut=40, btype='bandpass', order=4, rs='auto'):
    ''' Bandpass filter using Chebyshev type II filter.

        Description
        -----------
        This function implements a bandpass filter using Chebyshev type II filter.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        sfreq : int
            The sampling frequency.
        low_cut : int
            The low cut frequency.
        high_cut : int
            The high cut frequency.
        btype : str
            The type of filter. It can be 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
        order : int
            The order of the filter.
            For Chebyshev type II filter, the order must be even.
        rs : int
            The minimum attenuation in the stop band.
            If 'auto', the value is set to 40 for bandpass filters and 20 for other filters.

        returns
        -------
        np.ndarray
            The filtered data.
        '''

    Wn = [low_cut, high_cut]
    
    if rs == 'auto':
        if btype == 'bandpass':
            rs = 40
        else:
            rs = 20

    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = filtfilt(*cheby2(order, rs, Wn, btype, fs=eegdata['sfreq']), X[signal_])
        X_.append(filtered)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape)

    eegdata['X'] = X_

    return eegdata
