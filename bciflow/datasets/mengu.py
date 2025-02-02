import numpy as np
import h5py

def mengu(subject: int=1, 
          session_list: list=None,
          labels: list=None,
          depth: list=None,
          path='data/mengu/'):
    
    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be a int type value")
    if subject > 30 or subject < 1:
        raise ValueError("subject has to be between 1 and 30")
    
    # Check if the session_list input is valid
    _available_sessions = ['s%02d'%i for i in range(1, 12+1)]
    if session_list == None:
        session_list = _available_sessions
    elif type(session_list) != list:
        raise ValueError("session_list has to be an List or None type")
    else:
        for i in session_list:
            if i not in _available_sessions:
                raise ValueError("session_list has to be a sublist of ['s1', 's2', ..., 's12']")

    # Check if the labels input is valid
    _available_labels = ['f%02d'%i for i in range(1, 60+1)]
    if labels == None:
        labels = _available_labels
    elif type(labels) != list:
        raise ValueError("labels has to be a list type value")
    else:
        for i in labels:
            if i not in _available_labels:
                raise ValueError("labels has to be a sublist of ['f1', 'f2', ..., 'f60']")
    
    # Check if the depth input is valid
    _available_depths = ['low', 'high']
    if depth == None:
        depth = _available_depths
    elif type(depth) != list:
        raise ValueError("depth has to be a list type value")
    else:
        for i in depth:
            if i not in _available_depths:
                raise ValueError("depth has to be a sublist of ['low', 'high']")

    # Check if the path input is valid
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'
    
    # Set basic parameters of the clinical BCI challenge dataset
    sfreq = 1000.
    events = {'task_exec': [0, 5]}
    ch_names = np.array(["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", 
                         "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", 
                         "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", 
                         "CZ", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", 
                         "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", 
                         "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", 
                         "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",])
    tmin = 0.

    with h5py.File(path+'data_s%d_64.mat'%subject, "r") as f:
        data = np.asarray(f["datas"])

    session_id = np.where(np.isin(_available_sessions, session_list))[0]
    labels_id = np.where(np.isin(_available_labels, labels))[0]
    depth_id = np.where(np.isin(_available_depths, depth))[0]
    data = data[session_id, :, :, :, :]
    data = data[:, labels_id, :, :, :]
    data = data[:, :, :, :, depth_id]

    X, y = [], []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[4]):
                X.append(data[i, j, :, :, k])
                y.append(_available_labels[labels_id[j]])
    X, y = np.array(X), np.array(y)

    y_dict = {label: i for i, label in enumerate(labels)}
    y = np.array([y_dict[i] for i in y])

    print(X.shape, y.shape, y)
    print(y_dict)

    return {'X': X, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin}
