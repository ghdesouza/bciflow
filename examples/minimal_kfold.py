import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.datasets.BCICIV2a import bciciv2a as eeg_dataset
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

dataset = eeg_dataset(subject = 1)

pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'fe': (logpower, {'flating': True}),
               'clf': (lda(), {})}

results = kfold(target=dataset, start_window=2.5, pre_folding=pre_folding, pos_folding=pos_folding)
print(results)