import os
import numpy as np

from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

algorithm = 'BunDLeNet_recon_x'
rat_name = 'achilles'

data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
np.where(x < 0)
x_, b_ = prep_data(x, b, win=20)
x_train, x_test, _, _ = timeseries_train_test_split(x_, b_)

results = []
for latent_dim in [1,2,3,4,5,6,7,8,9,10]:
    print(latent_dim)
    for i in range(5):
        model = BunDLeNet(latent_dim=latent_dim, num_behaviour=x_.shape[-1])
        model.build(input_shape=x_train.shape)
        model.tau = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='linear'),
        ])
        train_history, test_history = train_model(
            x_train,
            x_train[:,1,0,:],
            model,
            b_type='continuous',
            gamma=0.9,
            learning_rate=0.001,
            n_epochs=500,
            initialisation=None,
            validation_data=(x_test, x_test[:,1,0,:]),
        )
        results.append({
            "latent_dim": latent_dim,
            "markov_train_loss": train_history[-1, 0],
            "markov_test_loss": test_history[-1, 0],
            "behaviour_train_loss": train_history[-1, 1],
            "behaviour_test_loss": test_history[-1, 1],
            "total_train_loss": train_history[-1, -1],
            "total_test_loss": test_history[-1, -1]
        })

os.makedirs('data/generated/reconstruction_studies', exist_ok=True)
np.save(f'data/generated/reconstruction_studies/losses_vs_dim_{algorithm}.npy', results)

results = np.load(f'data/generated/reconstruction_studies/losses_vs_dim_{algorithm}.npy', allow_pickle=True)
print(results)