import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from sklearn.metrics import mean_squared_error
import scipy


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = np.dot(X.T, X) + reg * scipy.sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        self.W = V[0:rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def project(self, X):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.asarray(np.dot(X, self.A.T))

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.asarray(np.dot(X, np.dot(self.A.T, self.W.T)))


algorithm = 'RRR'

for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=1)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :].reshape(x_train.shape[0], -1)
    x1_tr = x_train[:, 1, :, :].reshape(x_train.shape[0], -1)
    x0_tst = x_test[:, 0, :, :].reshape(x_test.shape[0], -1)
    x1_tst = x_test[:, 1, :, :].reshape(x_test.shape[0], -1)

    ### Deploy RRR
    dim = 3
    rrr = ReducedRankRegressor(x0_tr, b_train_1, dim)

    ### Projecting into latent space
    y0_tr = rrr.project(x0_tr)
    y1_tr = rrr.project(x1_tr)
    y0_tst = rrr.project(x0_tst)
    y1_tst = rrr.project(x1_tst)

    # Predicting
    b_train_1_pred = rrr.predict(x0_tr)
    b_test_1_pred = rrr.predict(x0_tst)

    print('mse of train data ', mean_squared_error(b_train_1_pred, b_train_1))
    print('mse of test data ', mean_squared_error(b_test_1_pred, b_test_1))

    # Save the weights
    # model.save_weights(f'data/generated/{algorithm}_model_rat_{rat_name}')
    np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}', y0_tr)
    np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}', y1_tr)
    np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}', y0_tst)
    np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}', y1_tst)
    np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
