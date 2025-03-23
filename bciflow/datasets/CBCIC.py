'''
CBCIC.py

Description
-----------
This code is used to load EEG data from the CBCIC dataset. It modifies the data to fit the requirements of the eegdata dictionary, which is used to store and process EEG data.

Dependencies
------------
numpy
pandas
scipy 
'''


import numpy as np
import scipy

def cbcic(subject: int=1, 
          session_list: list=None,
          labels=['left-hand', 'right-hand'],
          path='data/cbcic/'):
    
    """
    Description
    -----------
    
    Load EEG data from the CBCIC dataset. 
    It modifies the data to fit the requirements of the eegdata dictionary, which is used to store and process EEG data. 

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from.
        session_list : list, optional
            list of session codes. 
            default state is None, which results on the collection of all session. 
        labels : list
            list of labels used in the dataset.
            default state is to use all labels.
        path : str
            path to the foldar that contains all dataset files.


    Returns:
    ----------
        eegdata: A dictionary containing the loaded EEG data.

    """

    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("subject has to be between 1 and 9")
    
    # Check if the session_list input is valid
    if type(session_list) != list and session_list != None:
        raise ValueError("session_list has to be an List or None type")
    if session_list != None:
        for i in session_list:
            if i not in ['T', 'E']:
                raise ValueError("session_list has to be a sublist of ['T', 'E']")
    else:
        session_list = ['T', 'E']

    # Check if the labels input is valid
    if type(labels) != list:
        raise ValueError("labels has to be a list type value")
    for i in labels:
        if i not in ['left-hand', 'right-hand']:
            raise ValueError("labels has to be a sublist of ['left-hand', 'right-hand']")
    
    # Check if the path input is valid
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'
    
    # Set basic parameters of the clinical BCI challenge dataset
    sfreq = 512.
    events = {'get_start': [0, 3],
            'beep_sound': [2],
            'cue': [3, 8],
            'task_exec': [3, 8]}
    ch_names = np.array(["F3", "FC3", "C3", "CP3", "P3", "FCz", "CPz", "F4", "FC4", "C4", "CP4", "P4"])
    tmin = 0.

    rawData, rawLabels = [], []

    for sec in session_list:
        
        file_name = 'parsed_P%02d%s.mat'%(subject, sec)
        try:
            raw=scipy.io.loadmat(path+file_name)
        except:
            raise ValueError("The file %s does not exist in the path %s"%(file_name, path))

        rawData_ = raw['RawEEGData']
        rawLabels_ = np.reshape(raw['Labels'], -1)
        rawData_ = np.reshape(rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2]))
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)
    
    X, y = np.concatenate(rawData), np.concatenate(rawLabels)
    labels_dict = {1: 'left-hand', 2: 'right-hand'}
    y = np.array([labels_dict[i] for i in y])

    selected_labels = np.isin(y, labels)
    X, y = X[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])

    return {'X': X, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin}
