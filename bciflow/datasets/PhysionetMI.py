'''
PhysionetMI.py

Description
-----------
This code is used to load EEG data from the PhysionetMI dataset. 
It modifies the data to fit the requirements of the eegdata class, 
which is used to store and process EEG data. 

Dependencies
------------
numpy
pandas
scipy
mne 

'''

import numpy as np
import pandas as pd
import scipy
import mne

import os
file_directory = os.path.dirname(os.path.abspath(__file__))

def physionetmi(subject: int=1, 
                session_list: list=None, 
                run_list: list=None, 
                labels=['left-hand', 'right-hand', 'both-feet', 'both-hand']):
    """
        Description
        -----------
        
        Load EEG data from the PhysionetMI dataset. 
        The data is loaded for a specific subject, session, and run.
        The data is filtered based on the event codes specified in the 'labels_dict'.

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

    sfreq = 160.
    events = {'cue': [0, 1],
            'task_exec': [1, 4]}
    ch_names = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
                'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 
                'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
                'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
                
    labels_dict = {'left-hand': 1, 'right-hand': 2, 'both-feet': 3, 'both-hand': 4}

    ch_names = np.array(ch_names)
    tmin = 0.

    if session_list is None:
        session_list = [i+1 for i in range(2,14)]

    rawData, rawLabels = [], []

    for sec in session_list:
        raw=mne.io.read_raw_edf(file_directory+'/data/PhysionetMI/S%03d/S%03dR%02d.edf'%(subject, subject, sec), preload=True, verbose=False)
        raw_data = raw.get_data()

        description = raw.annotations.to_data_frame()
        description = np.array(description[description['description'] != "T0"]["description"])
        annotations_final = [label_name(key_, sec) for key_ in description]
        annotations = raw.annotations.to_data_frame()
        annotations = annotations[annotations['description'] != "T0"]
        annotations['description'] = annotations_final
        
        first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
        annotations['onset'] = (pd.to_datetime(annotations['onset']) - first_timestamp).dt.total_seconds()
        annotations['description'] = annotations['description'].astype(int)
        new_trial_time = np.array(annotations['onset'])

        times_ = np.array(raw.times)
        rawData_ = []
        for trial_ in new_trial_time:
            idx_ = np.where(times_ == trial_)[0][0]
            rawData_.append(raw_data[:, idx_:idx_+640])
        rawData_ = np.array(rawData_)
        rawData_ = np.reshape(rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2]))
        rawLabels_ = np.array(annotations['description'])

        rawData.append(rawData_)
        rawLabels.append(rawLabels_)

    rawLabels = np.concatenate(rawLabels)
    rawData = np.concatenate(rawData)

    inside = np.isin(rawLabels, list(map(labels_dict.get, labels_dict.keys())))
    rawData, rawLabels = rawData[inside], rawLabels[inside]

    data, labels = rawData, rawLabels
    
    return {'X': data, 
            'y': labels, 
            'sfreq': sfreq, 
            'y_dict': labels_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin}

dic = {
    'T1': { 1 : [3,4,7,8,11,12], 4 : [5,6,9,10,13,14]},
    'T2': { 2 : [3,4,7,8,11,12], 3 : [5,6,9,10,13,14]},
}



def label_name(key, sec):

    ''' 
        Description
        -----------
        This function returns the event name given the event code and session number.

        Parameters
        ----------
        key : int
            event code
        sec : int
            session number
        Returns
        -------
        membro : str
        a string representing the event name
        
        '''

    for membro, numero in dic[key].items():
        if sec in numero:
            return membro
