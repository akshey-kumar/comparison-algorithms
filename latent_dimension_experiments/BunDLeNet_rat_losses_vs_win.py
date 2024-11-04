import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'BunDLeNet'

# for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
rat_name = 'achilles'
# Load data
data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
np.where(x < 0)

results = []
for win in range(1,50):
    print(f"win: {win}")
    x_, b_ = prep_data(x, b, win=win)
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
    model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1])
    train_history, test_history = train_model(
        x_train,
        b_train_1,
        model,
        b_type='continuous',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
        validation_data=(x_test, b_test_1),
    )
    results.append({
        "win": win,
        "markov_train_loss": train_history[-1,0],
        "markov_test_loss": test_history[-1,0],
        "behaviour_train_loss": train_history[-1, 1],
        "behaviour_test_loss": test_history[-1, 1],
        "total_train_loss": train_history[-1,-1],
        "total_test_loss": test_history[-1,-1]
    })

    # Append the result to a text file
    with open(f'latent_dimension_experiments/losses_vs_win_{algorithm}.txt', 'a') as f:
        f.write(str(results[-1]) + '\n')

    # print(results)
    # np.save(f'latent_dimension_experiments/losses_vs_win_{algorithm}.npy', results)
