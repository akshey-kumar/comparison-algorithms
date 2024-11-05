import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

results_recon_x = np.load(f'data/generated/reconstruction_studies/losses_vs_dim_BunDLeNet_recon_x.npy', allow_pickle=True)
results_recon_b = np.load(f'data/generated/reconstruction_studies/losses_vs_dim_BunDLeNet_recon_b.npy', allow_pickle=True)

df_x = pd.DataFrame.from_dict(list(results_recon_x))
df_b = pd.DataFrame.from_dict(list(results_recon_b))

# Rename columns to ensure they match
df_x.rename(columns={'your_latent_dim_column': 'latent_dim', 'your_total_train_loss_column': 'total_train_loss'}, inplace=True)
df_b.rename(columns={'your_latent_dim_column': 'latent_dim', 'your_total_train_loss_column': 'total_train_loss'}, inplace=True)
print(df_x)
# Add a source column to differentiate between the two datasets
df_x['source'] = 'full neuronal state reconstruction'
df_b['source'] = 'reconstructing only the behaviour of interest'

# Concatenate the two DataFrames
df_combined = pd.concat([df_x, df_b])

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_combined, x='latent_dim', y='total_train_loss', hue='source', marker='o')

plt.title('Loss vs Latent Dimensions')
plt.xlabel('Latent Dimensions')
plt.ylabel('Loss')
plt.legend(title=None)
plt.grid(True)
plt.show()

