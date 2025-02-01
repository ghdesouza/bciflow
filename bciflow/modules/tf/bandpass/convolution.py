import numpy as np

def bandpass_conv(eegdata, low_cut=4, high_cut=40, transition=None, window_type='hamming', kind='same'):
    
    X = eegdata['X'].copy()
    sfreq = eegdata['sfreq']
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    if transition is None:
        transition = (high_cut - low_cut) / 2
    if isinstance(transition, int):
        transition = float(transition)
    if isinstance(transition, float):
        transition = [transition, transition]

    NL = int(4 * sfreq / transition[0])
    NH = int(4 * sfreq / transition[1])


    hlpf = np.sinc(2 * high_cut / sfreq * (np.arange(NH) - (NH - 1) / 2))
    if window_type=='hamming':
        hlpf *= np.hamming(NH)
    elif window_type=='blackman':
        hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)

    hhpf = np.sinc(2 * low_cut / sfreq * (np.arange(NL) - (NL - 1) / 2))
    if window_type=='hamming':
        hhpf *= np.hamming(NL)
    elif window_type=='blackman':
        hhpf *= np.blackman(NL)
    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    kernel = np.convolve(hlpf, hhpf)
    if len(kernel) > X.shape[-1] and kind == 'same':
        kind = 'valid'

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = np.convolve(X[signal_], kernel, mode=kind)
        X_.append(filtered)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape)
    eegdata['X'] = X_
    return eegdata

