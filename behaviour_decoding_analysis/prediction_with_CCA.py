import os

import numpy as np
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

algorithm = 'CCA'

for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=20)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy model
    dim = 3
    cca = CCA(n_components=dim)
    cca.fit(x_train[:, 0, 0, :], b_train_1)
    print('R squared of CCA on train data', cca.score(x_train[:, 0, 0, :], b_train_1))
    print('R squared of CCA on test data', cca.score(x_test[:, 0, 0, :], b_test_1))

    # Projecting into latent space
    y0_tr = cca.transform(x_train[:, 0, 0, :])
    y1_tr = cca.transform(x_train[:, 1, 0, :])
    y0_tst = cca.transform(x_test[:, 0, 0, :])
    y1_tst = cca.transform(x_test[:, 1, 0, :])

    b_train_1_pred = cca.predict(x_train[:, 1, 0, :])
    b_test_1_pred = cca.predict(x_test[:, 1, 0, :])
    """
    plt.figure()
    plt.plot(b_train_1_pred)
    plt.plot(b_train_1)

    plt.figure()
    plt.plot(y0_tr)
    plt.show()
    """

    # Save the weights
    # model.save_weights(f'data/generated/BunDLeNet_model_rat_{rat_name}')
    print(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}')
    os.makedirs('data/generated/predicted_and_true_behaviours', exist_ok=True)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1_pred__{algorithm}_rat_{rat_name}',
               b_train_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1_pred__{algorithm}_rat_{rat_name}', b_test_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
    
