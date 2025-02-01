import numpy as np

class nbpw():

    def __init__(self):
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

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
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> object:

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
        return self.sqrt_2pi * np.exp(-((y*y)/(2*h*h)))
        
    def PXij(self, Xij: float, j: int) -> float:
        return np.sum(np.array([ self.soothing_kernel( Xij-Xj, self.hj[j] ) for Xj in self.X[:, j] ])) / self.N

    def PXij_w(self, Xij: float, w: int, j: int) -> float:
        return np.sum(np.array([ self.soothing_kernel( Xij-Xkj, self.hwj[w][j] ) for Xkj in self.X[self.y==w][:, j] ])) / self.Nw[w]
    
    def Pw_Xi(self, w: int, Xi: np.ndarray) -> float:
        result = self.Pw[w]
        for j in range(len(Xi)):
            if self.PXij(Xi[j], j) == 0:
                result *= self.PXij_w(Xi[j], w, j)/1e-8
            else:
                result *= self.PXij_w(Xi[j], w, j)/self.PXij(Xi[j], j)
        return result
