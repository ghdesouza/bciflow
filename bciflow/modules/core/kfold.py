'''
kfold.py

Description
-----------

This module contains the method kfold, which is used to perform a stratified k-fold cross-validation. 
The method is designed to work with eegdata dictionary.

Dependencies
------------
util on modules/core
sklearn
numpy
pandas
inspect

'''

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import inspect
from ..core.util import util

def kfold(target, start_window=0, start_test_window=None, window_size=2, pre_folding={}, pos_folding={}):
    ''' kfold object
    
    Description
    -----------
    This method is used to perform a stratified k-fold cross-validation. 
    The method is designed to work with eegdata dictionary.

    Parameters
    ----------
    target : dict
        Input EEG data.
    start_window : int

    start_test_window : int

    pre_folding : dict
        A dictionary containing the preprocessing functions to be applied to the data before the cross-validation.
        The keys are the names of the preprocessing functions, and the values are tuples containing the function and its parameters.
    pos_folding : dict
        A dictionary containing the postprocessing functions to be applied to the data before the cross-validation.
        The keys are the names of the postprocessing functions, and the values are the functions.
    window_size : float 
            The size of the window to be used in the crop method of eegdata.
    Returns
    -------
    results : pandas.DataFrame
        A pandas dataframe containing the results of the cross-validation. 
        The columns are 'fold', 'tmin', 'true_label', and the labels of the events in the target object.

    '''
    if type(start_window) is float:
        start_window = [start_window]

    if start_test_window is None:
        start_test_window = start_window

    target_dict = {}
    for tmin_ in start_test_window:
        target_dict[tmin_] = util.crop(data=target, tmin=tmin_, window_size=window_size, inplace=False)
        
    for tmin_ in start_test_window:
        for name, pre_func in pre_folding.items():

            if inspect.isfunction(pre_func[0]):
                target_dict[tmin_] = util.apply_to_trials(data=target_dict[tmin_], func=pre_func[0], func_param=pre_func[1], inplace=False)
            else:
                target_dict[tmin_] = util.apply_to_trials(data=target_dict[tmin_], func=pre_func[0].transform, func_param=pre_func[1], inplace=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_id = 0
    results = []
    for train_index, test_index in skf.split(target["y"], target["y"]):
        fold_id += 1

        target_train = []
        for tmin_ in start_window:
            target_train.append(util.get_trial(data=target, ids=train_index))
        target_train = util.concatenate(target_train)

        target_test = {}
        for tmin_ in start_test_window:
            target_test[tmin_] = util.get_trial(data=target, ids=test_index)

        for name, pos_func in pos_folding.items():
            
            if name != 'clf':
                if inspect.isfunction(pos_func[0]):
                    target_train = pos_func[0](target_train, **pos_func[1])
                else:
                    target_train = pos_func[0].fit_transform(target_train, **pos_func[1])

                for tmin_ in start_test_window:
                    if inspect.isfunction(pos_func[0]):
                        target_test[tmin_] = pos_func[0](target_test[tmin_], **pos_func[1])
                    else:
                        target_test[tmin_] = pos_func[0].transform(target_test[tmin_])


        clf, clf_param = pos_folding['clf']
        if not inspect.isfunction(clf):
            clf = clf.fit(target_train['X'], target_train['y'], **clf_param)
                
        for tmin_ in start_test_window:
            try:
                y_pred = clf.predict_proba(target_test[tmin_]['X'])
            except:
                y_pred = np.zeros((len(target_test[tmin_]['y']), len(target['y_dict'])))
            y_pred = np.round(y_pred, 4)
            for trial_ in range(len(y_pred)):
                results.append([fold_id, tmin_, find_key_with_value(target['y_dict'], target_test[tmin_]['y'][trial_]), *y_pred[trial_]])


    results = np.array(results)
    results = pd.DataFrame(results, columns=['fold', 'tmin', 'true_label', *target['y_dict'].keys()])

    return results


def find_key_with_value(dictionary, i):
    '''find_key_with_value
    
    Description
    -----------
    This function returns the key of a dictionary given a value.
    
    Parameters
    ----------
    dictionary : dict
        The dictionary to be searched.
    i : any
        The value to be searched for.

    Returns
    -------
    key : any
        The key of the dictionary that contains the value i. If the value is not found, returns None.

    '''
    for key, value in dictionary.items():
        if value == i:
            return key
    return None