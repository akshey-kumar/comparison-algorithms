import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from behaviour_alignment import behaviour_alignment
import seaborn as sns

algorithm = 'BunDLeNet_win_1' # 'BunDLeNet' 'cebra_h' 'PCA' 'CCA' 'RRR'
rat_names = ['achilles', 'gatsby', 'cicero', 'buddy']
reg_scores = np.zeros((len(rat_names), len(rat_names)))

for i, rat_name_i in enumerate(rat_names):
    for j, rat_name_j in enumerate(rat_names):

        y_t_i = np.loadtxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name_i}')
        b_t_i = np.loadtxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name_i}')
        y_t_j = np.loadtxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name_j}')
        b_t_j = np.loadtxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name_j}')

        y_b_i = behaviour_alignment(y_t_i, b_t_i, n_bins = 1600)
        y_b_j = behaviour_alignment(y_t_j, b_t_j, n_bins = 1600)

        df = pd.merge(y_b_i, y_b_j, on='b', how = 'inner', suffixes=('_i','_j'))

        y_i = np.stack(df['y_i'].to_numpy())
        y_j = np.stack(df['y_j'].to_numpy())

        # Regressing y_j (dependent variable) on y_i (independent variable)
        reg = LinearRegression().fit(y_i, y_j)
        reg_scores[i,j] = reg.score(y_i, y_j)

# Create a heatmap with annotations
plt.figure(figsize=(8, 6))  # Adjust the figure size if necessary
sns.set(font_scale=1.2)  # Increase font size for better readability

# Plot the heatmap
sns.heatmap(reg_scores, annot=True, fmt=".2f", cmap='YlGnBu',
            cbar_kws={'label': 'Regression Score'},
            xticklabels=rat_names,
            yticklabels=rat_names,
            vmax=0, vmin=1,
            linewidths=0.5, linecolor='gray')

# Set the title and show the plot
plt.title(algorithm, fontsize=18)
plt.show()
