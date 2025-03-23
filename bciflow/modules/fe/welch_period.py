'''
welch_period.py

Description
-----------
This module contains the implementation of the welch_periodogram feature extractor.

Dependencies
------------
numpy
scipy

'''

import numpy as np
from scipy.signal import welch

class welch_period():
    ''' Welch periodogram feature extractor
    
    Description
    -----------
    This class implements welch's periodgram feature extractor.
    
    Attributes
    ----------
    None
    
    Methods
    -------
    transform(data):
        Transforms the input data into welch's periodgram feature space.
        
    '''
    def __init__(self, flating: bool = False):
        ''' Initializes the class.
        
        Description
        -----------
        This method initializes the class. Here you decide whether to return the data in a flat format.
        The default is False. It does not return anything.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        '''
        if type(flating) != bool:
            raise ValueError ("Has to be a boolean type value")
        else:
            self.flating = flating

    def fit(self, eegdata):
        ''' That method does nothing.
        
        Description
        -----------
        This method does nothing.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        Returns
        -------
        self
        
        '''
        if type(eegdata) != dict:
            raise ValueError ("Has to be a dict type")         
        return self

    def transform(self, eegdata, sfreq: int) -> dict:
        ''' Transforms the input data into welch's periodgram feature space.
        
        Description
        -----------
        This method transforms the input data into welch's periodogram feature space. It returns a
        dictionary with the transformed data.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        Returns
        -------
        output : dict
            The transformed data.
            
        '''
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
        ''' Fits the model to the input data and transforms it into welch's periodogram feature space.
        
        Description
        -----------
        This method fits the model to the input data and transforms it into welch´s periodogram feature
        space. It returns a dictionary with the transformed data.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        Returns
        -------
        output : dict
            The transformed data.
            
        '''
        return self.fit(eegdata).transform(eegdata, sfreq)
