import numpy as np
from scipy.signal import cheby2, filtfilt

def chebyshevII(eegdata, low_cut=4, high_cut=40, btype='bandpass', order=4, rs='auto'):
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
