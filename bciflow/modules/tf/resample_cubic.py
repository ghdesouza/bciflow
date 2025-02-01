import numpy as np
from scipy.interpolate import CubicSpline

def cubic_resample(eegdata, new_sfreq):
    
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
    sfreq = eegdata['sfreq']
    divisor = sfreq//new_sfreq
    duration = X.shape[-1]/sfreq
    old_times = np.arange(0, duration, 1./sfreq)
    new_times = np.arange(0, duration, 1./new_sfreq)
    X_ = []
    for signal_ in range(X.shape[0]):
                cubic_spline = CubicSpline(old_times, X[signal_])
                new_signal = cubic_spline(new_times)
                X_.append(new_signal)

    X_ = np.array(X_)
    X_ = X_.reshape(*eegdata['X'].shape[:-1],eegdata['X'].shape[-1]//divisor )

    eegdata['X'] = X_
    eegdata['sfreq'] = new_sfreq

    return eegdata
