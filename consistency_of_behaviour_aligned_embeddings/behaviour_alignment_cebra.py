import sys
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import cebra

algorithm = 'cebra_h' # 'BunDLeNet' 'cebra_h' 'PCA' 'CCA' 'y_shuffled_BunDLeNet' 'linear_dynamics' # 'linear_dynamics' 'point_embedding_noisy' 'y_shuffled_BunDLeNet' # 'BunDLeNet' 'cebra_h' 'PCA' 'CCA'
y = []
b = []
rat_names = ['achilles', 'gatsby', 'cicero', 'buddy']
for rat_name_i in rat_names:

        y1_tr_i = np.loadtxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name_i}')
        b_train_1_i = np.loadtxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name_i}')[:,0]
        y.append(y1_tr_i)
        b.append(b_train_1_i)

# embedding1 = np.random.uniform(0, 1, (1000, 5))
# embedding2 = np.random.uniform(0, 1, (1000, 8))
# labels1 = np.random.uniform(0, 1, (1000, ))
# labels2 = np.random.uniform(0, 1, (1000, ))
# print(embedding1.shape, embedding2.shape, labels1.shape, labels2.shape)
# Between-datasets consistency, by aligning on the labels
scores, pairs, ids_datasets = cebra.sklearn.metrics.consistency_score(embeddings=y,
                                                                  labels=b,
                                                                  dataset_ids=rat_names,
                                                                  between="datasets")
print(scores)
cebra.plot_consistency(
    scores,
    pairs=pairs,
    datasets=ids_datasets,
    title="BunDLe-Net")
plt.show()