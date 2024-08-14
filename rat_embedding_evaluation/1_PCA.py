import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from sklearn.decomposition import PCA

algorithm = 'PCA'

for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=1)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy PCA
    dim = 3
    pca = PCA(n_components=dim)
    pca.fit(x_train[:, 0, 0, :])
    print('Percentage of variance explained by the first ', dim, ' PCs: ',
          pca.explained_variance_ratio_[:dim].sum().round(3))

    # Projecting into latent space
    y0_tr = pca.transform(x_train[:, 0, 0, :])
    y1_tr = pca.transform(x_train[:, 1, 0, :])

    y0_tst = pca.transform(x_test[:, 0, 0, :])
    y1_tst = pca.transform(x_test[:, 1, 0, :])

    # Save the weights
    # model.save_weights(f'data/generated/{algorithm}_model_rat_{rat_name}')
    np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}', y0_tr)
    np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}', y1_tr)
    np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}', y0_tst)
    np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}', y1_tst)
    np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
