import sys

sys.path.append(r'../')
import numpy as np
# from functions import *
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet'

# Load Data (and excluding behavioural neurons)
for worm_num in [1]:
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL', ]

    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.behaviour
    state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing',
                   'Ventral turn']
c
    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    print(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}')
    np.savetxt(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}', Y0_tr)
    np.savetxt(f'data/generated/saved_Y/Y1_tr__{algorithm}_worm_{worm_num}', Y1_tr)
    np.savetxt(f'data/generated/saved_Y/Y0_tst__{algorithm}_worm_{worm_num}', Y0_tst)
    np.savetxt(f'data/generated/saved_Y/Y1_tst__{algorithm}_worm_{worm_num}', Y1_tst)
    np.savetxt(f'data/generated/saved_Y/B_train_1__{algorithm}_worm_{worm_num}', B_train_1)
    np.savetxt(f'data/generated/saved_Y/B_test_1__{algorithm}_worm_{worm_num}', B_test_1)
