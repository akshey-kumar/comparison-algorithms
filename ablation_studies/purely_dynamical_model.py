import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import tensorflow as tf

# Load Data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'BunDLeNet_ablation_pure_dynamical'
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL'
]
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.behaviour

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
    gamma=1.0,
    learning_rate=0.001,
    n_epochs=1000,
    validation_data=(X_test, B_test_1),
    initialisation='best_of_5_init',
)

plt.figure()
for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Train loss $\mathcal{L}$"
]):
    plt.plot(train_history[:, i], label=label)
plt.plot(test_history[:, -1], label='Test loss')
plt.legend()
plt.show()

# Projecting into latent space
Y0_ = model.tau(X_[:, 0]).numpy()

# Save the weights
save_model = False
if save_model:
    model.save_weights(f'data/generated/BunDLeNet_model_worm_{worm_num}')
    np.savetxt(f'data/generated/saved_Y/Y0__{algorithm}_worm_{worm_num}', Y0_)
    np.savetxt(f'data/generated/saved_Y/B__{algorithm}_worm_{worm_num}', B_)
    Y0_ = np.loadtxt(f'data/generated/saved_Y/Y0__{algorithm}_worm_{worm_num}')
    B_ = np.loadtxt(f'data/generated/saved_Y/B__{algorithm}_worm_{worm_num}').astype(int)

# Plotting latent space dynamics
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names, show_points=True)
vis.plot_latent_timeseries()
vis.plot_phase_space()
vis.rotating_plot(filename=f'figures/rotation_{algorithm}_worm_{worm_num}.gif')
