import sys
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'BunDLeNet'
b_function = 'trinary_behaviour_distractors'

# Load x and b
x = np.load(f"branching_simulation/data/x_{b_function}.npy")
b = np.load(f"branching_simulation/data/b_{b_function}.npy")
print(x.shape, b.shape)


x_, b_, t_ = [], [], []
for i, _ in enumerate(x):
    x_trial, b_trial = prep_data(x[i], b[i], win=1)
    x_.append(x_trial)
    b_.append(b_trial)
    t_.append(np.arange(b_trial.shape[0]))


x_ = np.concatenate(x_, axis=0)
b_ = np.concatenate(b_, axis=0)
t_ = np.concatenate(t_, axis=0)
print(x_.shape, b_.shape, t_.shape)

show_plots = False
if show_plots:
    plt.plot(x_[:,-1,0,:])
    plt.plot(b_, '--')
    plt.show()


# x_, b_ = prep_data(x, b, win=1)

# Train test split
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
x_train, x_test, t_train_1, t_test_1 = timeseries_train_test_split(x_, t_)

# Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=np.unique(b).shape[0])
print(np.unique(b).shape[0])

train_history, test_history = train_model(
    x_train,
    b_train_1,
    model,
    b_type='discrete',
    gamma=0.9,
    learning_rate=0.001,
    n_epochs=500,
    initialisation=None,
    validation_data=(x_test, b_test_1),
)

plt.figure()
for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Train loss $\mathcal{L}$"
]):
    plt.plot(train_history[:, i], label=label)
plt.plot(test_history[:, -1], label='Test loss', linestyle='--')
plt.legend()


# Projecting into latent space
y0_tr = model.tau(x_train[:, 0]).numpy()
y1_tr = model.tau(x_train[:, 1]).numpy()

y0_tst = model.tau(x_test[:, 0]).numpy()
y1_tst = model.tau(x_test[:, 1]).numpy()

# Save the weights
# model.save_weights(f'data/generated/BunDLeNet_model_branching_simulated')
print(f'data/generated/saved_Y/y0_tr__{algorithm}_branching_simulated')
np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_branching_simulated_{b_function}', y0_tr)
np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_branching_simulated_{b_function}', y1_tr)
np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_branching_simulated_{b_function}', y0_tst)
np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_branching_simulated_{b_function}', y1_tst)
np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_branching_simulated_{b_function}', b_train_1)
np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_branching_simulated_{b_function}', b_test_1)
np.savetxt(f'data/generated/saved_Y/t_train_1__{algorithm}_branching_simulated_{b_function}', t_train_1)
np.savetxt(f'data/generated/saved_Y/t_test_1__{algorithm}_branching_simulated_{b_function}', t_test_1)
plt.show()

