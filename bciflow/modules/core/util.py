import numpy as np
import pandas as pd
from numpy import mean, sqrt, square, arange
import sys
import os
import numpy as np
from typing import Union, List, Optional

class util():

    def timestamp(data):
        tmin = data["tmin"]
        sfreq = data["sfreq"]
        size = data["X"].shape[-1]
        return np.array([tmin + i/sfreq for i in range(size)])

    def crop(data, tmin, window_size, inplace):

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

        data = data.copy()

        if type(ids) != np.ndarray:
            if type(ids) == int:
                ids = [ids]
            ids = np.array(ids)

        data["X"] = data["X"][ids]
        data["y"] = data["y"][ids]
 
        return data

    def apply_to_trials(data, func, func_param={}, inplace = False):

        data = data if inplace else data.copy()

        temp_X = []
        for trial_ in range(len(data["X"])):
            temp_X.append(func(util.get_trial(data, trial_), **func_param))

        data = util.concatenate(temp_X)
        
        if not inplace:
            return data
    
    def concatenate(data_colection):
        
        data = data_colection[0].copy()
        for data_ in data_colection[1:]:
            data["X"] = np.concatenate([data["X"], data_["X"]])
            data["y"] = np.concatenate([data["y"], data_["y"]])

        return data