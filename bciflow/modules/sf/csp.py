import numpy as np
import scipy as sp

class csp:
    def __init__(self, m_pairs: int = 2):
        if type(m_pairs) != int or m_pairs <= 0:
            raise ValueError("Must be a positive integer")
        else:
            self.m_pairs = m_pairs

    def fit(self, eegdata: dict) -> np.ndarray:
        X = None
        y = None
        if type(eegdata['X']) != np.ndarray:
            raise ValueError("data must be an array-like object")
        else:
            X = eegdata['X'].copy()

        if type(eegdata['y']) != np.ndarray:
            raise ValueError("data must be an array-like object")
        else:
            y = eegdata['y'].copy()

        self.bands = X.shape[1]
        self.n_electrodes = X.shape[2]

        self.W = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))

        # unique values of y
        y_unique = np.unique(y)
        if len(y_unique) != 2:
            raise ValueError("y must have exactly two unique classes.")

        sigma = np.zeros((len(y_unique), self.bands, self.n_electrodes, self.n_electrodes))

        for i in range(len(y)):
            for band_ in range(self.bands):
                if y[i] == y_unique[0]:
                    sigma[0, band_] += (X[i, band_] @ X[i, band_].T)
                else:
                    sigma[1, band_] += (X[i, band_] @ X[i, band_].T)

        for band_ in range(self.bands):
            sigma[0, band_] /= np.sum(y == y_unique[0])
            sigma[1, band_] /= np.sum(y == y_unique[1])

        sigma_tot = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))
        for band_ in range(self.bands):
            sigma_tot[band_] = sigma[0, band_] + sigma[1, band_]

        W = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))
        for band_ in range(self.bands):
            try:
                _, W[band_] = sp.linalg.eigh(sigma[0, band_], sigma_tot[band_])
            except:
                W[band_] = np.eye(self.n_electrodes)
                

        W = np.array(W)
        first_aux = W[:, :, :self.m_pairs]
        last_aux = W[:, :, -self.m_pairs:]

        self.W = np.concatenate((first_aux, last_aux), axis=2)

        return self

    def transform(self, eegdata: dict) -> dict:
        X = None
        y = None
        if type(eegdata['X']) != np.ndarray:
            raise ValueError("data must be an array-like object")
        else:
            X = eegdata['X'].copy()


        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=1)

        if X.shape[1] != self.bands:
            raise ValueError("The number of bands in the input data is different from the number of bands in the fitted data.")
        
        X = [np.transpose(self.W[band_]) @ X[:, band_] for band_ in range(self.bands)]
        X = np.swapaxes(np.array(X), 0, 1)

        eegdata['X'] = X
        return eegdata
    
        #X_ = [np.transpose(self.W) @ X_[i] for i in range(len(X_))]

    def fit_transform(self, eegdata: dict) -> dict:
        X = None
        y = None
        if type(eegdata['X']) != np.ndarray:
            raise ValueError("data must be an array-like object")
        else:
            X = eegdata['X'].copy()

        if type(eegdata['y']) != np.ndarray:
            raise ValueError("data must be an array-like object")
        else:
            y = eegdata['y'].copy()


        return self.fit(eegdata).transform(eegdata)
