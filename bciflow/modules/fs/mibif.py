'''
mibif.py

Description
-----------
This module contains the implementation of the Mutual Information Best Individual Features feature extractor.

Dependencies
------------
numpy
sklearn

'''

import numpy as np
from sklearn.metrics import mutual_info_score

class MIBIF:
    ''' Mutual Information Best Individual Features feature extractor

    Description
    -----------
    This class implements the Mutual Information Best Individual Features feature extractor.

    Attributes
    ----------
    n_features : int
        The number of features to be selected.
    original_n_features : int
        The original total number of features available.
    paired : bool
        Whether the features are paired or not.
    order : list
        The order of the features.
    clf : nbpw
        The classifier used to calculate the mutual information.
    pairs : np.ndarray
        The pairs of features.

    Methods
    -------
    find_pair(u, max_col):
        Finds the pair of a feature.
    fit(eegdata):  
        Fits the feature extractor to the data.
    transform(eegdata):
        Transforms the input data into the selected feature space.
    fit_transform(eegdata):
        Fits the feature extractor to the data and transforms the input data into the selected feature space.

    '''
    def __init__(self, n_features, clf, paired=True):
        ''' Initializes the class.
        
        Description
        -----------
        This method initializes the class. It receives the number of features to be selected, the total number of features
        available, and whether the features are paired or not. It does not return anything.
        
        Parameters
        ----------
        n_features : int
            The number of features to be selected.
        paired : bool
            Whether the features are paired or not.
            Just for CSP, use paired=True.

        Returns
        -------
        None
        
        '''

        self.original_n_features = n_features
        self.n_features = self.original_n_features
        self.paired = paired
        self.order = [0]
        self.clf = clf

    def find_pair(self, u, max_col):
        ''' Finds the pair of a feature.

        Description
        -----------
        This method finds the pair of a feature. 
        It receives the feature index and the maximum number of columns. 
        It returns the index of the pair of the feature.

        Warning
        -------
        This method is only used when the features are paired in CSP.

        Parameters
        ----------
        u : int
            The feature index.
        max_col : int
            The maximum number of columns.

        Returns
        -------
        int
            The index of the pair of the feature.

        '''
        i = int(u / max_col)
        j = u % max_col
        j_pair = max_col - 1 - j
        u = max_col * i + j_pair
        return u

    def fit(self, eegdata):
        ''' Fits the feature extractor to the data.

        Description
        -----------
        This method fits the feature extractor to the data.
        It receives the input data and the labels.
        It does return itself.

        Parameters
        ----------
        eegdata : dict
            The input data.
        
        Returns
        -------
        self

        '''

        self.n_features = self.original_n_features

        X = eegdata['X'].copy()
        y = eegdata['y'].copy()

        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]

        X_ = [X[i].reshape(-1) for i in range(len(X))]
        mi = []
        for i in range(len(X_[0])):
            X__ = []
            for j in range(len(X_)):
                X__.append([X_[j][i]])
            X__ = np.array(X__)
            #self.clf.fit(np.array(X__), y)
            try:
                self.clf.fit(X__, y)
            except:
                X__ = np.nan_to_num(X__)
                X__ += np.random.random(X__.shape) * 1e-6
                self.clf.fit(X__, y)
            y_pred = self.clf.predict(np.array(X__))
            mi.append([i, mutual_info_score(y, y_pred)])                

        mi = sorted(mi, key=lambda x: x[1], reverse=True)
        mi = np.array(mi)
        self.order = mi[:, 0].astype(int)

        if self.paired:
            self.pairs = np.zeros((int(len(X[0])/2), 2))
            max_col = X.shape[-1]
            new_order = []
            n_features = self.n_features
            for i in range(len(self.order)):
                order_ = self.order[i]
                new_order.append(order_)
                order_pair = self.find_pair(order_, max_col)
                new_order.append(order_pair)
                if order_pair not in self.order[:self.n_features] and i < self.n_features:
                    n_features += 1
            self.order = new_order
            self.n_features = n_features

        return self

    def transform(self, eegdata):
        ''' Transforms the input data into the selected feature space.
        
        Description
        -----------
        This method transforms the input data into the selected feature space.
        It receives the input data.
        It returns the transformed data.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        Returns
        -------
        output : dict
            The transformed data.
            
        '''

        X = eegdata['X'].copy()

        X_ = [X[i].reshape(-1) for i in range(len(X))]
        X_ = np.array(X_)
        X_ = X_[:, self.order][:, :self.n_features]

        eegdata['X'] = X_
        return eegdata

    def fit_transform(self, eegdata):
        ''' Fits the feature extractor to the data and transforms the input data into the selected feature space.

        Description
        -----------
        This method fits the feature extractor to the data and transforms the input data into the selected feature space.
        It receives the input data and the labels.
        It returns the transformed data.

        Parameters
        ----------
        eegdata : dict
            The input data.

        Returns
        -------
        output : dict
            The transformed data.

        '''
        
        return self.fit(eegdata).transform(eegdata)

