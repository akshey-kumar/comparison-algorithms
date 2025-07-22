import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

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

'''
results = []
for win in range(1,50):

    # prepare data for BundLe Net
    X_, B_ = prep_data(X, B, win=15)
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
    print(f"win: {win}")
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    train_history, test_history = train_model(
        X_train,
        B_train_1,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=1000,
        validation_data=(X_test, B_test_1),
        initialisation='best_of_5_init',
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

    print(results)
    np.save(f'latent_dimension_experiments/losses_vs_win_{algorithm}.npy', results)
'''
# Plotting
results = np.load(f'latent_dimension_experiments/losses_vs_win_{algorithm}.npy', allow_pickle=True)
df = pd.DataFrame.from_dict(list(results))
print(df.head())

df_melted = df.melt(id_vars='win', value_vars=['markov_train_loss', 'behaviour_train_loss', 'total_train_loss'],
                    var_name='Loss Type', value_name='Loss')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_melted, x='win', y='Loss', hue='Loss Type', style='Loss Type', s=100)
plt.xlabel('Latent Dimension')
plt.ylabel('Train Loss')
plt.title('Markov and Behaviour Loss vs Latent Dimension')
plt.show()