import sys
sys.path.append(r'../')
import numpy as np
from scipy import signal
import mat73
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.animation as animation
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class Database:
    def __init__(self):
        data_set_no = 0
        data_dict = mat73.loadmat('NoStim_Data.mat')
        data  = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][data_set_no]
        derivatives = data['derivs'][data_set_no]
        NeuronNames = data['NeuronNames'][data_set_no]
        fps = data['fps'][data_set_no]
        States = data['States'][data_set_no]


        self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
        self.state_names = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.derivative_traces = derivatives['traces'].T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

flat_partial = lambda x: x.reshape(x.shape[0],-1)

def r2_single(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return 1 - mse(y_pred, y_true)/tf.math.reduce_variance(y_true)

def r2(Y_true, Y_pred):
    r2_list=[]
    for i in range(Y_true.shape[-1]):
        R2 = r2_single(Y_true[:,i], Y_pred[:,i])
        r2_list.append(R2)
    r2_list = tf.stack(r2_list)
    return tf.math.reduce_mean(r2_list)

def bandpass(traces, f_l, f_h, sampling_freq):
    
    
    cut_off_h = f_h*sampling_freq/2 ## in units of sampling_freq/2
    cut_off_l= f_l*sampling_freq/2 ## in units of sampling_freq/2
        #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.
    
    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

def hits_at_rank(rank, Y_test, Y_pred):
    nbrs = NearestNeighbors(n_neighbors=rank, algorithm='ball_tree').fit(Y_test)
    distances, indices = nbrs.kneighbors(Y_test)
    return np.mean(np.linalg.norm(Y_pred - Y_test, axis=1) < distances[:,-1])

def plot_phase_space(pca_neurons, states, show_points=False):
    if pca_neurons.shape[1]==3:
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        X = pca_neurons.T
        points = np.array(X).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = [*mcolors.TABLEAU_COLORS.keys()][:8] ###
        #cmap = cm.get_cmap('Pastel1')
        #colors = cmap(np.arange(8))
        for segment, state in zip(segments, states[:-1]):
            p = ax.plot3D(segment.T[0], segment.T[1], segment.T[2], color=colors[state] )
        # Create legend
        state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reverse', 'Slowing', 'Ventral turn']
        legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
        ax.legend(handles=legend_elements)
        plt.show()
    elif pca_neurons.shape[1]==2:
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes()
        X = pca_neurons.T
        points = np.array(X).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = [*mcolors.TABLEAU_COLORS.keys()][:8]
        #cmap = cm.get_cmap('Pastel1')
        #colors = cmap(np.arange(8))
        for segment, state in zip(segments, states[:-1]):
            p = ax.plot(segment.T[0], segment.T[1], color=colors[state])
        # Create legend
        state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reverse', 'Slowing', 'Ventral turn']
        legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
        ax.legend(handles=legend_elements)
    else:
        print("Error: Dimension of input array is neither 2 or 3")
     
    if show_points==True:
    	ax.scatter(X[0], X[1], X[2], c='k',s=0.2)
    return ax
        
def rotating_plot(Y, B, show_points=False):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    def rotate(angle):
        ax.view_init(azim=angle)

    points = np.array(Y.T).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [*mcolors.TABLEAU_COLORS.keys()][:8] ###
    #cmap = cm.get_cmap('Pastel1')
    #colors = cmap(np.arange(8))
    for segment, state in zip(segments, B[:-1]):
        p = ax.plot3D(segment.T[0], segment.T[1], segment.T[2], color=colors[state] )
    # Create legend
    state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reverse', 'Slowing', 'Ventral turn']
    legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
    ax.legend(handles=legend_elements)
    if show_points==True:
    	ax.scatter(Y[:,0], Y[:,1], Y[:,2], c='k',s=0.2)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 5), interval=150)
    rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
    plt.show()
    return ax
