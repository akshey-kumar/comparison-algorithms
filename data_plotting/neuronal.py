import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database

algorithm = 'BunDLeNet'
# Load Data (and excluding behavioural neurons)
worm_num = 0
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

# Set up the figure
plt.figure(figsize=(4, 3))  # Smaller width, moderate height for readability

# Plot each trace with an offset
for i in range(35, 45):
    plt.plot(X[500:2000, i] + i / 2)

# Customize the plot
plt.xlabel('Time', fontsize=12)  # Label the x-axis
plt.ylabel('Neurons', fontsize=12)  # Label the y-axis
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.box(False)  # Remove the border frame

# Display the plot
plt.show()
