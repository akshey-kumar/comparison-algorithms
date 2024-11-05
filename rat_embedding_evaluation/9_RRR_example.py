import numpy as np
from sklearn.model_selection import train_test_split
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
        print(self.W.shape, self.A.shape)

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def project(self, X):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, self.A.T)

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, np.dot(self.A.T, self.W.T))

def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = np.random.randn(num, dimX)
    W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
    Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
    return X, Y

# Generating synthetic data
n_samples = 1000
dimX = 10
dimY = 5
rrank = 2
X, Y = ideal_data(n_samples, dimX, dimY, rrank, noise=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit Reduced Rank Regression with a chosen rank (e.g., rank 2)
rank = 2
rrr = ReducedRankRegressor(X, Y, rank)
Y_pred = rrr.predict(X)
rr_space = rrr.project(X)
print(Y_pred.shape, rr_space.shape)

print("True Y (first 2 samples):")
print(Y[:2])
print("\nPredicted Y (first 2 samples):")
print(Y_pred[:2])

# Mean squared error to evaluate
mse = mean_squared_error(np.asarray(Y), np.asarray(Y_pred))
print("\nMean Squared Error:", mse)