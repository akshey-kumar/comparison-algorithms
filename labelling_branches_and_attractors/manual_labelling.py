import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

import os

def manual_labeling(Y, b, b_names, plot_type='points', save_path='labelling_branches_and_attractors/label_progress.npy'):
    """
    Manually label dataset points with undo support ('z') and minimal save/resume.

    Parameters:
    -----------
    Y : numpy.ndarray
        The latent space coordinates (N x 3).
    b : numpy.ndarray
        The initial behavior labels for each data point (N,).
    b_names : list
        List of behavior names for the labels.
    save_path : str
        File to save/load progress.
    """
    if os.path.exists(save_path):
        manually_assigned_labels = np.load(save_path, allow_pickle=True)
        print(f"Loaded progress from {save_path}")
        print(manually_assigned_labels)
    else:
        #manually_assigned_labels = np.zeros_like(b)
        manually_assigned_labels = np.full_like(b, '', dtype='object')

    colors = sns.color_palette('deep', len(np.unique(b)))
    if plot_type == 'points':
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            Y[:, 0], Y[:, 1], Y[:, 2],
            c=b,
            cmap=ListedColormap(colors),
            s=8,
            alpha=0.3
        )
    elif plot_type == 'trajectories':
        vis = LatentSpaceVisualiser(
            y=Y,
            b=b.astype(int),
            b_names={i: name for i, name in enumerate(b_names)},
        )
        fig, ax = vis.plot_phase_space(arrow_length_ratio=0.5, show_fig=False, alpha=0.2)

    star_line, = ax.plot([Y[0, 0]], [Y[0, 1]], [Y[0, 2]],
                         marker='*', color='red', markersize=10,
                         markeredgecolor='black', linestyle='None')
    plt.ion()
    plt.show()

    # Find first unlabeled index (assuming label 0 means "unlabeled")
    idx = np.argmax(manually_assigned_labels == '')
    while idx < len(Y):
        star_line.set_data([Y[idx, 0]], [Y[idx, 1]])
        star_line.set_3d_properties([Y[idx, 2]])
        plt.draw()
        plt.pause(0.01)

        key = input(f"Label point {idx + 1}/{len(Y)} (an alphabet or press '1' to undo or press '2' for trajectory view): ")

        if key.isalpha():
            manually_assigned_labels[idx] = key
            idx += 1
        elif key == '1' and idx > 0:
            idx -= 1
            print(f"Undo! Back to point {idx + 1}")
        elif key == '2':
            vis = LatentSpaceVisualiser(
                y=Y,
                b=b.astype(int),
                b_names={i: name for i, name in enumerate(b_names)},
            )
            fig, ax = vis.plot_phase_space(arrow_length_ratio=0.5, show_fig=False, alpha=0.4)
            ax.plot(Y[idx,0], Y[idx,1], Y[idx,2],
                       marker='*', color='red', markersize=10,
                       markeredgecolor='black', linestyle='None')
            plt.draw()
        else:
            print("Invalid input.")

        if idx % 5 == 0:
            print('saved')
            np.save(save_path, manually_assigned_labels)

    np.save(save_path, manually_assigned_labels)
    plt.ioff()
    print(f"Saved final labels to {save_path}")
    return manually_assigned_labels

from ncmcm.data_loaders.matlab_dataset import Database

algorithm = 'BunDLeNet'
worm_num = 0

data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
b_names = data.behaviour_names

# loading embedding
file_pattern = f'data/generated/comparable_embeddings/{{}}__BunDLeNet_worm_{worm_num}'
y = np.loadtxt(file_pattern.format('Y'))
b = np.loadtxt(file_pattern.format('B')).astype(int)

Y, B = y[:], b[:]
# Start the manual labeling
branch_labels = manual_labeling(Y,
                                B,
                                b_names,
                                plot_type='points',
                                save_path = f'labelling_branches_and_attractors/label_progress_worm_{worm_num}.npy'
                                )

print(branch_labels)

vis = LatentSpaceVisualiser(
            y=Y,
            b=branch_labels.astype(int),
            b_names=['1','2','3']        )
vis.plot_phase_space(arrow_length_ratio=0.5)

