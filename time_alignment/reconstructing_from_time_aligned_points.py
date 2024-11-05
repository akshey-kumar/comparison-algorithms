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
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from inverse_embedding import fit_inverse_embedder

algorithm = 'BunDLeNet'
worm_num = 0

data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL', ]
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour
x_, b_ = prep_data(x, b, win=15)
x_train, x_test, _, _ = timeseries_train_test_split(x_, b_)
x0_tr = x_train[:, 0, -1, :]
x0_tst = x_test[:, 0, -1, :]
print(x_train.shape)

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
b_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
b_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)


plotting_neuronal_behavioural(x0_tr, b_train_1, b_names=data.behaviour_names)

# fitting inverse embedder
inverse_embedder = fit_inverse_embedder(x0_tr, y0_tr, x0_tst, y0_tst)

# points to inverse embedc(ventral turn)
y0_inv_embed = np.array([
    [-0.368, 1.104, -1.196],
    [-0.441,0.406,0.009],
    [-0.191, 0.020, 0.673],
    [-1.067, -0.309, 0.673]
])

vis = LatentSpaceVisualiser(y0_tr, b_train_1, data.behaviour_names)
fig, ax = vis.plot_phase_space(axis_view=(0,0,),  arrow_length_ratio=0.2, show_fig=False)
ax.scatter(y0_inv_embed[:, 0], y0_inv_embed[:, 1], y0_inv_embed[:, 2], c='k', s=25, marker='x')
plt.show()

exit()

x0_pred = inverse_embedder(y0_inv_embed)
for x0 in x0_pred:
    plt.plot(x0)
plt.show()




exit()
# points to inverse embedc(sus rev)
y0_inv_embed = np.array([
    [1.671, 0.801, -0.212],
    [1.671, 0.801, -0.682],
    [3.006, -0.028, -0.984],
    [1.766, 0.789, -0.421],
    [1.766, 0.789, -1.784]
])