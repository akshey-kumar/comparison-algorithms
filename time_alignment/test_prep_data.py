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
import tensorflow as tf

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
print(x[15:].shape, b[15:].shape)
x_, b_ = prep_data(x, b, win=15)
print(x_.shape, b_.shape)

print((b[15:] == b_).sum())

print((x[15:] == x_[:,1,-1,:]).sum(), x[15:].size)

x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
x1_tr = x_train[:, 1, -1, :]
x1_tst = x_test[:, 1, -1, :]
