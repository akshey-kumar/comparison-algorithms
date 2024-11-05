import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

algorithm = 'BunDLeNet'
worm_num = 0
print(algorithm, ' worm_num: ', worm_num)

data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
B_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)


def extract_bouts(B, b):
    bouts = []
    current_bout = []
    for i, val in enumerate(B):
        if val == b:
            current_bout.append(i)  # Add index to the current bout
        else:
            if current_bout:  # If the current bout is not empty, save it
                bouts.append(current_bout)
                current_bout = []  # Reset for the next bout

    if current_bout:  # Append the last bout if the B ends with a bout
        bouts.append(current_bout)

    return sorted(bouts, key=len)


def plot_behaviours_in_normalised_time(Y, B, b, behaviour_name):
    # extracting behavioural bouts
    bout_indices = extract_bouts(B, b)
    Y_bouts = [Y[idx] for idx in bout_indices]

    # plotting
    # cmap = plt.get_cmap('tab10')
    colors = ["#008080", "#FF6F61", "#FFD700"]
    # plt.figure(figsize=(3.5, 3))
    plt.figure(figsize=(6, 5))

    for bout_idx, y_bout in enumerate(Y_bouts):
        normalised_t = np.linspace(0, 1, y_bout.shape[0])
        for i in range(3):
            darkness_factor = 0.3 + 0.7 * (bout_idx / len(Y_bouts))
            # color = cmap(i)
            # plt.plot(normalised_t, y_bout[:, i], c=colors[i], alpha=darkness_factor, marker='o')
            plt.plot(normalised_t, y_bout[:, i], c=colors[i], alpha=darkness_factor, marker='o')

    plt.xlabel('normalized time')
    plt.yticks([])
    plt.title(f"Behaviour : {behaviour_name}")
    plt.tight_layout()


vis = LatentSpaceVisualiser(Y0_tr, B_train_1, data.behaviour_names)
vis.plot_phase_space(axis_view=(0,0,), arrow_length_ratio=0.3)
plt.show()


for b in np.unique(B_train_1):
    plot_behaviours_in_normalised_time(Y1_tr, B_train_1, b, data.behaviour_names[b])
plt.show()