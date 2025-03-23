'''
logpower.py

Description
-----------
This module contains the implementation of the logpower feature extractor.

Dependencies
------------
numpy

'''
import numpy as np

def logpower(eegdata: dict, flating: bool = False) -> dict:
    ''' Logpower feature extractor
    
    Description
    -----------
    This method implements the logpower feature extractor.
    It transforms the input data into the logpower feature space. 
    It returns a dictionary with the transformed data.

    Attributes
    ----------
    eegdata : dict
        The input data.
    flating : bool
        Decides whether to return the data in a flat format.
        Default is False.

    Returns
    -------
    output : dict
        The transformed data.

    '''
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = np.log(np.mean(X[signal_]**2))
        X_.append(filtered)

    X_ = np.array(X_)
    shape = eegdata['X'].shape
    if flating:
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))
    else:
        X_ = X_.reshape((shape[0], shape[1], np.prod(shape[2:-1])))

    eegdata['X'] = X_

    return eegdata