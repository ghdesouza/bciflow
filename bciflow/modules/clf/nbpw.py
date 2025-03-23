'''
nbpw.py

Description
-----------
This module contains the implementation of the Naive Bayes Parzen Window classifier.

Dependencies
------------
numpy

'''

import numpy as np

class nbpw():
    ''' Naive Bayes Parzen Window classifier
    
    Description
    -----------
    This class implements the Naive Bayes Parzen Window classifier. It is a non-parametric
    classifier that uses the Parzen Window method to estimate the probability density function
    of the features given the class. The class is implemented in a way that it can be used
    as a drop-in replacement for the classifiers in the scikit-learn library.
    
    Attributes
    ----------
    None

    Methods
    -------
    __init__():
        Initializes the class.
    predict_proba(X):
        Predicts the probability of each class given the features.
    predict(X):
        Predicts the class of the input features.
    fit(X, y):
        Fits the model to the input features and labels.
    soothing_kernel(y, h):
        Returns the value of the soothing kernel for the input parameters.
    PXij(Xij, j):
        Returns the probability of the feature Xij given the class.
    PXij_w(Xij, w, j):
        Returns the probability of the feature Xij given the class w.
    Pw_Xi(w, Xi):
        Returns the probability of the class w given the features Xi.

    '''
    def __init__(self):
        ''' Initializes the class.
        
        Description
        -----------
        This method initializes the class. It does not receive any parameters and does not
        return anything.
        
        Parameters
        ----------
        flating : bool, optional
        
        Returns
        -------
        None
        
        '''
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ''' Predicts the probability of each class given the features.
        
        Description
        -----------
        This method predicts the probability of each class given the input features. It
        returns a matrix with the probabilities of each class for each input feature.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input features.
            
        Returns
        -------
        proba : array-like, shape (n_samples, n_classes)
            The probabilities of each class for each input feature.
        
        '''
        proba = []
        for Xi in X:
            proba.append( [ self.Pw_Xi(w, Xi) for w in self.labels] )
            # if some value in proba is zero or nan, all values in the vector will be replaced by 1/n_classes
            if np.sum(proba[-1]) == 0 or np.isnan(np.sum(proba[-1])):
                proba[-1] = [1./len(proba[-1])]*len(proba[-1])
            else:
                proba[-1] /= np.sum(proba[-1])

        proba = np.array(proba) 
        nan_idx = np.isnan(proba)
        proba[nan_idx] = 1./len(proba[0])

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:

        ''' Predicts the class of the input features.
        
        Description
        -----------
        This method predicts the class of the input features. It returns the class with the
        highest probability for each input feature.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input features.
            
        Returns
        -------
        pred : array-like, shape (n_samples,)
            The predicted classes for the input features.
            
        '''
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        
        ''' Fits the model to the input features and labels.

        Description
        -----------
        This method fits the model to the input features and labels. It calculates the
        smoothing parameters and the probability of each class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input features.
        y : array-like, shape (n_samples,)
            The labels.
        verbose : bool, optional
            If True, prints the progress of the fitting process.

        Returns
        -------
        self : object
            Returns the instance itself.

        '''

        self.X, self.y = X.copy(), y.copy()

        self.sqrt_2pi = 1./np.sqrt(2*np.pi)

        self.N = len(self.y)
        self.labels = np.unique(np.array(self.y))
        self.Nw = {}
        self.Pw = {}
        for i in self.labels:
            self.Nw[i] = np.count_nonzero(self.y==i)
            self.Pw[i] = self.Nw[i]/len(self.y)


        self.hj = [ ((4./(3.*self.N))**0.2)*np.std(self.X[:, j]) for j in range(len(self.X[0])) ]
        self.hwj = {}
        for w in self.labels:
            X_w = self.X[self.y==w]
            self.hwj[w] = [ ((4./(3.*len(X_w)))**0.2)*np.std(X_w[:, j]) for j in range(len(X_w[0])) ]

        return self
    
    def soothing_kernel(self, y: float, h: float) -> float:
        ''' Returns the value of the soothing kernel for the input parameters.

        Description
        -----------
        This method returns the value of the soothing kernel for the input parameters.

        Parameters
        ----------
        y : float
            The input value.
        h : float
            The smoothing parameter.

        Returns
        -------
        result : float
            The value of the soothing kernel for the input parameters.
        
        '''
        return self.sqrt_2pi * np.exp(-((y*y)/(2*h*h)))
        
    def PXij(self, Xij: float, j: int) -> float:
        ''' Returns the probability of the feature Xij given the class.
        
        Description
        -----------
        This method returns the probability of the feature Xij given the class. It calculates
        the probability using the Parzen Window method.
        
        Parameters
        ----------
        Xij : float
            The input feature.
        j : int
            The index of the feature.

        Returns
        -------
        result : float
            The probability of the feature Xij given the class.

        '''
        return np.sum(np.array([ self.soothing_kernel( Xij-Xj, self.hj[j] ) for Xj in self.X[:, j] ])) / self.N

    def PXij_w(self, Xij: float, w: int, j: int) -> float:
        ''' Returns the probability of the feature Xij given the class w.
        
        Description
        -----------
        This method returns the probability of the feature Xij given the class w. It calculates
        the probability using the Parzen Window method.
        
        Parameters
        ----------
        Xij : float
            The input feature.
        w : int
            The class.
        j : int
            The index of the feature.
            
        Returns
        -------
        result : float
            The probability of the feature Xij given the class w.
            
        '''
        return np.sum(np.array([ self.soothing_kernel( Xij-Xkj, self.hwj[w][j] ) for Xkj in self.X[self.y==w][:, j] ])) / self.Nw[w]
    
    def Pw_Xi(self, w: int, Xi: np.ndarray) -> float:
        ''' Returns the probability of the class w given the features Xi.
        
        Description
        -----------
        This method returns the probability of the class w given the features Xi. It calculates
        the probability using the Parzen Window method.
        
        Parameters
        ----------
        w : int
            The class.
        Xi : array-like, shape (n_features,)
            The input features.
        
        Returns
        -------
        result : float
            The probability of the class w given the features Xi.

        '''
        result = self.Pw[w]
        for j in range(len(Xi)):
            if self.PXij(Xi[j], j) == 0:
                result *= self.PXij_w(Xi[j], w, j)/1e-8
            else:
                result *= self.PXij_w(Xi[j], w, j)/self.PXij(Xi[j], j)
        return result
