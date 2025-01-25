import numpy as np
import pywt

def wavelet(eegdata, levels=5):
    ''' Wavelet transform.

    Description
    -----------
    This function implements the wavelet transform.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    levels : int
        The number of levels.

    returns
    -------
    np.ndarray
        The transformed data.

    '''
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    widths = np.arange(1, levels+1)
    X_ = []
    for signal_ in range(X.shape[0]):
        coef_, freqs_ = pywt.cwt(X[signal_], widths, 'morl')
        X_.append(coef_)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape[0], eegdata['X'].shape[2], eegdata['X'].shape[1]*levels , eegdata['X'].shape[3])
    X_ = np.transpose(X_, (0, 2, 1, 3))
    eegdata['X'] = X_

    return eegdata



