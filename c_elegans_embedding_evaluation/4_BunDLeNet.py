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

    # Preprocess and prepare data for BunDLe Net
    # time, X = preprocess_data(X, data.fps)
    X_, B_ = prep_data(X, B, win=15)

    # Train test split
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

    # Deploy BunDLe Net
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
        # best_of_5_init=True

    )

    plt.figure()
    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Train loss $\mathcal{L}$"
    ]):
        plt.plot(train_history[:, i], label=label)
    plt.legend()
    plt.show()

    # Projecting into latent space
    Y0_tr = model.tau(X_train[:, 0]).numpy()
    Y1_tr = model.tau(X_train[:, 1]).numpy()

    Y0_tst = model.tau(X_test[:, 0]).numpy()
    Y1_tst = model.tau(X_test[:, 1]).numpy()

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    print(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}')
    np.savetxt(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}', Y0_tr)
    np.savetxt(f'data/generated/saved_Y/Y1_tr__{algorithm}_worm_{worm_num}', Y1_tr)
    np.savetxt(f'data/generated/saved_Y/Y0_tst__{algorithm}_worm_{worm_num}', Y0_tst)
    np.savetxt(f'data/generated/saved_Y/Y1_tst__{algorithm}_worm_{worm_num}', Y1_tst)
    np.savetxt(f'data/generated/saved_Y/B_train_1__{algorithm}_worm_{worm_num}', B_train_1)
    np.savetxt(f'data/generated/saved_Y/B_test_1__{algorithm}_worm_{worm_num}', B_test_1)
