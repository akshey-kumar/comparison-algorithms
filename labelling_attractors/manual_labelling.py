import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

def manual_labeling(Y, b, b_names, plot_type='points'):
    """
    Manually label the dataset points one by one.

    Parameters:
    -----------
    Y : numpy.ndarray
        The latent space coordinates of the data points (N x 3).
    b : numpy.ndarray
        The initial behavior labels for each data point (N,).
    b_names : list
        List of behavior names for the labels.
    """
    manually_assigned_labels = np.copy(b)

    colors = sns.color_palette('deep', len(np.unique(b)))
    color_dict = {name: colors[i] for i, name in enumerate(np.unique(b))}

    if plot_type=='points':
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            Y[:, 0], Y[:, 1], Y[:, 2],
            c=manually_assigned_labels,
            cmap=ListedColormap(colors),
            s=8, alpha=0.4
        )
    elif plot_type=='trajectories':
        vis = LatentSpaceVisualiser(
            y=Y,
            b=manually_assigned_labels.astype(int),
            b_names={i: name for i, name in enumerate(b_names)},
        )
        fig, ax = vis.plot_phase_space(arrow_length_ratio=0.5, show_fig=False, alpha=0.2)

    # Plot current point as a star
    star_line, = ax.plot([Y[0, 0]], [Y[0, 1]], [Y[0, 2]],
                         marker='*', color='red', markersize=10,
                         markeredgecolor='black', linestyle='None')
    plt.ion()
    plt.show()

    idx = 0
    while idx < len(Y):
        # Update star marker position
        star_line.set_data([Y[idx, 0]], [Y[idx, 1]])
        star_line.set_3d_properties([Y[idx, 2]])

        plt.draw()
        plt.pause(0.01)

        key = input(f"Label point {idx + 1}/{len(Y)} with a number (or press 'u' for 3D view): ")

        if key.isdigit():
            label = int(key)
            manually_assigned_labels[idx] = label
            print(f"Label for point {idx + 1} set to {label}.")
        else:
            print("Invalid input, skipping label.")

        idx += 1

    plt.ioff()
    return manually_assigned_labels


from ncmcm.data_loaders.matlab_dataset import Database

algorithm = 'BunDLeNet'
worm_num = 1

data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
b_names = data.behaviour_names

# loading embedding
file_pattern = f'data/generated/comparable_embeddings/{{}}__BunDLeNet_worm_{worm_num}'
y = np.loadtxt(file_pattern.format('Y0'))
b = np.loadtxt(file_pattern.format('B')).astype(int)

Y, B = y[:4], b[:4]
# Start the manual labeling
branch_labels = manual_labeling(Y,B, b_names, plot_type='points')
print(branch_labels)

vis = LatentSpaceVisualiser(
            y=Y,
            b=branch_labels.astype(int),
            b_names=['1','2','3']        )
vis.plot_phase_space(arrow_length_ratio=0.5)

