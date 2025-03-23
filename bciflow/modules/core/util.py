'''
util.py

Description
-----------
This module contains the `util` class, which implements several utility methods for EEG data manipulation.
The class is designed to assist in k-fold operations and streamline preprocessing tasks such as cropping,
timestamp generation, and trial selection for EEG data stored in dictionary format.

Dependencies
------------
numpy
pandas

'''

import numpy as np
import pandas as pd
from numpy import mean, sqrt, square, arange
import numpy as np

class util():
    ''' Utility functions for EEG data manipulation.
    
    Description
    -----------
    This class implements various utility methods to facilitate the manipulation of EEG data
    stored in dictionary format, including functions for timestamp creation, data cropping, trial extraction,
    function application on trials, and data concatenation.
    
    Attributes
    ----------
    None
    
    Methods
    -------
    timestamp(data):
        Calculates the timestamps for the EEG data based on its starting time (tmin), sampling frequency (sfreq),
        and the number of time samples.
    
    crop(data, tmin, window_size, inplace):
        Crops the EEG data to a specified time window.
        
    get_trial(data, ids):
        Extracts specified trials from the EEG data based on given indices.
    
    apply_to_trials(data, func, func_param, inplace=False):
        Applies a specified function to each trial in the EEG data. Parameter inplace has an default value of False.
    
    concatenate(data_collection):
        Concatenates multiple EEG data dictionaries into a single one.
    
    '''
    def timestamp(data):
        ''' Calculates timestamps for the EEG data.
        
        Description
        -----------
        This method generates an array of timestamps based on the EEG data's starting time (tmin),
        sampling frequency (sfreq), and number of time samples.
        
        Parameters
        ----------
        data : dict
            EEG data dictionary.
        
        Returns
        -------
        np.array
            Array of timestamps corresponding to each time sample.
        
        '''
        tmin = data["tmin"]
        sfreq = data["sfreq"]
        size = data["X"].shape[-1]
        return np.array([tmin + i/sfreq for i in range(size)])

    def crop(data, tmin, window_size, inplace):
        ''' Crops the EEG data to a specified time window.
        
        Description
        -----------
        This method crops the EEG data, retaining a window of specified length starting from `tmin`.
        If `inplace` is set to False, it returns a new cropped EEG dictionary without modifying the input data.
        
        Parameters
        ----------
        data : dict
            EEG data dictionary.
        tmin : float
            Starting time for the cropping.
        window_size : float
            Duration (in seconds) of the time window to keep.
        inplace : bool, optional
            If True, modifies the input data dictionary. If False, returns a new dictionary.
        
        Returns
        -------
        dict (optional)
            Cropped EEG data (only if `inplace=False`).
        
        Raises
        ------
        ValueError
            If `tmin + window_size` exceeds the maximum time in the original data.
        
        '''
        data = data if inplace else data.copy()

        X = data['X'].copy()
        X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

        indice = int((tmin - data["tmin"]) * data["sfreq"])
        max_indice = indice + int(window_size * data["sfreq"])
        if np.any(indice + int(window_size * data["sfreq"]) > X.shape[-1]):
            raise ValueError("tmin + window_size must be less than or equal to the tmax of the original data")

        X = X[:, indice:max_indice]
        X = X.reshape((*data['X'].shape[:-1], max_indice - indice))

        data["X"] = X
        data['tmin'] = tmin

        if not inplace:
            return data

    def get_trial(data, ids):
        ''' Extracts specified trials from the EEG data.
        
        Description
        -----------
        This method extracts the specified trials from the EEG data, based on the indices provided in `ids`.
        
        Parameters
        ----------
        data : dict
            EEG data dictionary.
        ids : list[int] or np.ndarray
            Indices of the trials to extract.
        
        Returns
        -------
        dict
            New EEG data dictionary containing only the selected trials.
        
        '''
        data = data.copy()

        if type(ids) != np.ndarray:
            if type(ids) == int:
                ids = [ids]
            ids = np.array(ids)

        data["X"] = data["X"][ids]
        data["y"] = data["y"][ids]
 
        return data

    def apply_to_trials(data, func, func_param={}, inplace = False):
        ''' Applies a specified function to each trial in the EEG data.
        
        Description
        -----------
        This method applies a given function to each trial in the EEG data. The function should
        accept a single-trial EEG dictionary as input. If `inplace` is set to False, it returns
        a new EEG dictionary with the function applied to each trial.
        
        Parameters
        ----------
        data : dict
            EEG data dictionary.
        func : callable
            Function to apply to each trial.
        func_param : dict, optional
            Additional keyword arguments to pass to `func`.
        inplace : bool, optional
            If True, modifies the input data dictionary. If False, returns a new dictionary. Parameter is set as False by default.
        
        Returns
        -------
        dict (optional)
            EEG data dictionary with the function applied to each trial (only if `inplace=False`).
        
        '''
        data = data if inplace else data.copy()

        temp_X = []
        for trial_ in range(len(data["X"])):
            temp_X.append(func(util.get_trial(data, trial_), **func_param))

        data = util.concatenate(temp_X)
        
        if not inplace:
            return data
    
    def concatenate(data_colection):
        ''' Concatenates multiple EEG data dictionaries.
        
        Description
        -----------
        This method concatenates a list of EEG data dictionaries into a single EEG data dictionary.
        
        Parameters
        ----------
        data_collection : list[dict]
            A list of EEG data dictionaries to concatenate.
        
        Returns
        -------
        dict
            A new EEG data dictionary containing the concatenated data from all input dictionaries.
        
        '''
        data = data_colection[0].copy()
        for data_ in data_colection[1:]:
            data["X"] = np.concatenate([data["X"], data_["X"]])
            data["y"] = np.concatenate([data["y"], data_["y"]])

        return data