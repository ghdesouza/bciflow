import numpy as np

class nonlinearenergy():
    def __init__(self, flating: bool = False): 
        if type(flating) != bool:
            raise ValueError ("Has to be a boolean type value")
        else:
            self.flating = flating

    def fit(self, eegdata):
        if type(eegdata) != dict:
            raise ValueError ("Has to be a dict type")         
        return self

    def transform(self, eegdata) -> dict:
        if type(eegdata) != dict:
            raise ValueError ("Has to be a dict type")                
        X = eegdata['X'].copy()

        many_trials = len(X.shape) == 4
        if not many_trials:
            X = X[np.newaxis, :, :, :]

        output = []
        trials_, bands_, channels_, times_ = X.shape

        for trial_ in range(trials_):
            output.append([])
            for band_ in range(bands_):
                output[trial_].append([]) 
                for channel_ in range(channels_):
                    diff = X[trial_, band_, channel_, 1:-1]**2 - (X[trial_, band_, channel_, 2:] * X[trial_, band_, channel_, :-2])
                    output[trial_][band_].append(np.sum(diff))
                
        output = np.array(output)
        
        if self.flating:
            output = output.reshape(output.shape[0], -1)

        if not many_trials:
            output = output[0]
        eegdata['X'] = output
        return eegdata
    
    def fit_transform(self, eegdata) -> dict:
        return self.fit(eegdata).transform(eegdata)
