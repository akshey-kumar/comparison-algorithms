import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'BunDLeNet_behaviour_shuffling'
os.makedirs('data/generated/shuffling_experiments', exist_ok=True)
for worm_num in range(5):
    # Load Data (and excluding behavioural neurons)
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

    # Shuffling data to destroy dynamical information
    rand_idx = np.random.permutation(B.shape[0])
    B = B[rand_idx]

    # prepare data for BunDLe Net
    X_, B_ = prep_data(X, B, win=15)
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)


    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    train_history, test_history = train_model(
        X_train,
        B_train_1,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=1000,
        validation_data=(X_test, B_test_1),
        initialisation='best_of_5_init',
    )
    history = {
        "markov_train_loss": train_history[:, 0],
        "markov_test_loss": test_history[:, 0],
        "behaviour_train_loss": train_history[:, 1],
        "behaviour_test_loss": test_history[:, 1],
        "total_train_loss": train_history[:, -1],
        "total_test_loss": test_history[:, -1]
    }
    np.save(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', history)

    ### Projecting into latent space
    Y0_tr = model.tau(X_train[:, 0]).numpy()
    Y1_tr = model.tau(X_train[:, 1]).numpy()
    Y0_tst = model.tau(X_test[:, 0]).numpy()
    Y1_tst = model.tau(X_test[:, 1]).numpy()

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}', Y0_tr)
    np.savetxt(f'data/generated/saved_Y/Y1_tr__{algorithm}_worm_{worm_num}', Y1_tr)
    np.savetxt(f'data/generated/saved_Y/Y0_tst__{algorithm}_worm_{worm_num}', Y0_tst)
    np.savetxt(f'data/generated/saved_Y/Y1_tst__{algorithm}_worm_{worm_num}', Y1_tst)
    np.savetxt(f'data/generated/saved_Y/B_train_1__{algorithm}_worm_{worm_num}', B_train_1)
    np.savetxt(f'data/generated/saved_Y/B_test_1__{algorithm}_worm_{worm_num}', B_test_1)

