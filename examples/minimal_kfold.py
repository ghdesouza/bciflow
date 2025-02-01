import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

dataset = cbcic(subject = 1)

pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'fe': (logpower, {'flating': True}),
               'clf': (lda(), {})}

results = kfold(target=dataset, 
                start_window=dataset['events']['cue'][0]+0.5, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding)

print(results)