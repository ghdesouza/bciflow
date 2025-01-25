'''
euclideanalignment.py

Description
-----------
This module contains the implementation of the Euclidean Alignment method for Spatial filter.

Dependencies
------------
numpy
scipy

'''



from scipy.linalg import fractional_matrix_power
import numpy as np

class ea:
    def __init__(self):   
        self.target_transformation = None

    def calc_r(self, data):
        list_r = []
        for band in range(data.shape[1]):
            r = np.zeros((data.shape[2], data.shape[2]))
            for trial in range(data.shape[0]):
                product = np.dot(data[trial][band], data[trial][band].T)
                r += product
            r = r / data.shape[0]
            list_r.append(r)
        return np.array(list_r)
    
    def full_r(self, data):
        list_r = self.calc_r(data)
        list_r_inv = [fractional_matrix_power(r, -0.5) for r in list_r]
        return np.array(list_r_inv)

    def fit(self, eegdata):
        data = eegdata['X'].copy()
        self.target_transformation = self.full_r(data)
        return self

    def transform(self, eegdata):
        X = eegdata['X'].copy()

        for band in range(X.shape[1]):
            for trial in range(X.shape[0]):
                X[trial][band] = np.dot(self.target_transformation[band], X[trial][band])

        eegdata['X'] = X
        return eegdata

    def fit_transform(self, eegdata):
         return self.fit(eegdata).transform(eegdata)
