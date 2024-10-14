import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

algorithm = "BunDLeNet"
results = np.load(f'latent_dimension_experiments/losses_vs_dim_{algorithm}.npy', allow_pickle=True)
df = pd.DataFrame.from_dict(list(results))

min_total_loss = df.groupby('latent_dim')['total_train_loss'].min()
min_markov_loss = df.groupby('latent_dim')['markov_train_loss'].min()
min_behaviour_loss = df.groupby('latent_dim')['behaviour_train_loss'].min()

plt.figure(figsize=(10, 6))
plt.plot(min_total_loss, label='total loss')
plt.plot(min_markov_loss, label='markov loss')
plt.plot(min_behaviour_loss, label='behaviour loss')
plt.legend()
plt.xlabel('Latent Dimension')
plt.ylabel('Loss')
plt.title('Markov and Behaviour Loss vs Latent Dimension')
plt.show()


df_melted = df.melt(id_vars='latent_dim',
                    value_vars=['markov_train_loss', 'behaviour_train_loss', 'total_train_loss'],
                    var_name='Loss Type',
                    value_name='Loss')
plt.figure(figsize=(10, 6))
sns.stripplot(
    data=df_melted,
    x='latent_dim',
    y='Loss',
    hue='Loss Type',
    )

plt.xlabel('Latent Dimension')
plt.ylabel('Loss')
plt.title('Markov and Behaviour Loss vs Latent Dimension')
plt.show()