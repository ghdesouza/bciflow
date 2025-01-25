'''
CBCIC.py

Description
-----------
This code is used to load EEG data from the BCICIV2a dataset. 
It modifies the data to fit the requirements of the eegdata class, 
which is used to store and process EEG data. 

Dependencies
------------
numpy
pandas
scipy
mne 

'''

#https://sites.google.com/view/bci-comp-wcci/

import numpy as np
import pandas as pd
import scipy
import mne

import os
file_directory = os.path.dirname(os.path.abspath(__file__))

def cbcic(subject: int=1, 
          session_list: list=None, 
          run_list: list=None, 
          labels=['left-hand', 'right-hand']):
    """
        Description
        -----------
        
        Load EEG data from the CBCIC dataset. 
        It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data. 

        Parameters
        ----------
            subject : int
                index of the subject to retrieve the data from
            session_list : list, optional
                list of session codes
            run_list : list, optional
                list of run numbers
            events_dict : dict
                dictionary mapping event names to event codes
            verbose : str
                verbosity level


        Returns:
        ----------
            eegdata: An instance of the eegdata class containing the loaded EEG data.

        """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")

    sfreq = 512.
    events = {'get_start': [0, 3],
            'beep_sound': [2],
            'cue': [3, 8],
            'task_exec': [3, 8]}
    ch_names = ["F3", "FC3", "C3", "CP3", "P3", "FCz", "CPz", "F4", "FC4", "C4", "CP4", "P4"]
    ch_names = np.array(ch_names)
    tmin = 0.

    if session_list is None:
        session_list = ['T', 'E']

    rawData, rawLabels = [], []

    for sec in session_list:
        raw=scipy.io.loadmat(file_directory+'/data/CBCIC/parsed_P%02d%s.mat'%(subject, sec))
        rawData_ = raw['RawEEGData']
        rawLabels_ = np.reshape(raw['Labels'], -1)
        rawData_ = np.reshape(rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2]))
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)
    
    data, labels = np.concatenate(rawData), np.concatenate(rawLabels)
    labels_dict = {'left-hand': 1, 'right-hand': 2}

    return {'X': data, 
            'y': labels, 
            'sfreq': sfreq, 
            'y_dict': labels_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin}