'''
emd_sift.py

Description
-----------
This module contains the implementation of the Empirical Mode Decomposition (EMD) 
applied to EEG data. The EMD function decomposes the input signals into Intrinsic Mode Functions (IMFs).

Dependencies
------------
numpy
emd

'''

import numpy as np
import emd

def EMD(eegdata, n_imfs=5):

    '''
    Empirical Mode Decomposition (EMD) function to decompose signals into Intrinsic Mode Functions (IMFs).

    Parameters
    ----------
    eegdata : dict
        Input EEG data.
    n_imfs : int, optional
        Number of IMFs to extract, by default 5.

    Returns
    -------
    output : dict
        The transformed data.
        Value stored in key X is decomposed IMFs with shape (n_trials, n_imfs, n_electrodes, n_samples).

    Raises
    ------
    ValueError
        If the input data does not have exactly one band (shape[1] != 1).
    '''
    X = eegdata['X'].copy()
    # verify if the data has only one band
    if X.shape[1] != 1:
        raise ValueError('The input data must have only one band.')
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []

    for signal_ in range(X.shape[0]):
        try:
            imfs_ = emd.sift.sift(X[signal_], max_imfs=None).T
        except:
            imfs_ = np.random.rand(n_imfs, X.shape[-1]) * 1e-6

        if len(imfs_) > n_imfs:
            imfs_[n_imfs-1] = np.sum(imfs_[n_imfs-1:], axis=0)
            imfs_ = imfs_[:n_imfs]
        elif len(imfs_) < n_imfs:
            imfs_ = np.concatenate((imfs_, np.zeros((n_imfs-len(imfs_), X.shape[-1]))), axis=0)
        
        X_.append(imfs_)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape[0], eegdata['X'].shape[2],n_imfs*eegdata['X'].shape[1], eegdata['X'].shape[3])
    X_ = np.transpose(X_, (0, 2, 1, 3))
    eegdata['X'] = X_

    return eegdata
