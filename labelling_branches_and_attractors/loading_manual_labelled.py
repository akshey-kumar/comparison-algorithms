import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

import os
from ncmcm.data_loaders.matlab_dataset import Database

worm_num = 0
'''
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
b_names = data.behaviour_names
'''
# loading embedding
file_pattern = f'data/generated/comparable_embeddings/{{}}__BunDLeNet_worm_{worm_num}'
y = np.loadtxt(file_pattern.format('Y'))
b = np.loadtxt(file_pattern.format('B')).astype(int)

# Start the manual labeling
manually_assigned_labels = np.load(f'labelling_branches_and_attractors/label_progress_worm_{worm_num}.npy', allow_pickle=True)
branch_labels = np.array([s[0] for s in manually_assigned_labels])

print(branch_labels)
print(np.unique(branch_labels))
label_dict = {i: str(label) for i, label in enumerate(np.unique(branch_labels))}
translated_labels = np.array([k for s in branch_labels for k, v in label_dict.items() if v == s])
print(translated_labels)
plt.plot(translated_labels)
vis = LatentSpaceVisualiser(
            y=y,
            b=translated_labels,
            b_names={i:f'branch {i+1}' for i in label_dict},
)
vis.plot_phase_space(arrow_length_ratio=0.5)

