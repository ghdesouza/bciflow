'''
welch_period.py

Description
-----------
This module contains the implementation of the welch_periodogram feature extractor.

Dependencies
------------
eegdata on modules/core
typing
numpy

'''

import numpy as np
from typing import Union, List, Optional
from scipy.signal import welch

class welch_period():
    def __init__(self, flating: Optional[bool] = False):
        if type(flating) != bool:
            raise ValueError ("Has to be a boolean type value")
        else:
            self.flating = flating

    def fit(self, eegdata):
        if type(eegdata) != dict:
            raise ValueError ("Has to be a dict type")         
        return self

    def transform(self, eegdata, sfreq: int) -> dict:
        if type(eegdata) != dict:
            raise ValueError ("Has to be a dict type")                
        X = eegdata['X'].copy()
            
        many_trials = len(X.shape) == 4
        if not many_trials:
            X = X[np.newaxis, :, :, :]

        output = []
        trials_, bands_, channels_, _ = X.shape

        for trial_ in range(trials_):
            output.append([])
            for band_ in range(bands_):
                output[trial_].append([])
                for channel_ in range(channels_):
                    if X[trial_, band_, channel_, :].std() == 0:
                        output[trial_][band_].append(0)
                    else:
                        output[trial_][band_].append(welch(X[trial_, band_, channel_, :], sfreq))    

        output = np.array(output)
        
        if self.flating:
            output = output.reshape(output.shape[0], -1)

        if not many_trials:
            output = output[0]

        eegdata['X'] = output
        return eegdata
    
    def fit_transform(self, eegdata, sfreq: int) -> dict:
        return self.fit(eegdata).transform(eegdata, sfreq)
