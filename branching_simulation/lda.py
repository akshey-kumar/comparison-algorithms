
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split



algorithm = 'LDA'
b_function = 'trinary_behaviour'

# Load x and b
x = np.load(f"branching_simulation/data/x_{b_function}.npy")
b = np.load(f"branching_simulation/data/b_{b_function}.npy")
print(x.shape, b.shape)


x_, b_, t_ = [], [], []
for i, _ in enumerate(x):
    x_trial, b_trial = prep_data(x[i], b[i], win=1)
    x_.append(x_trial)
    b_.append(b_trial)
    t_.append(np.arange(b_trial.shape[0]))


x_ = np.concatenate(x_, axis=0)
b_ = np.concatenate(b_, axis=0)
t_ = np.concatenate(t_, axis=0)
print(x_.shape, b_.shape, t_.shape)

show_plots = True
if show_plots:
    plt.plot(x_[:,-1,0,:])
    plt.plot(b_, '--')
    plt.show()

# Train test split
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
x_train, x_test, t_train_1, t_test_1 = timeseries_train_test_split(x_, t_)

### Deploy LDA
dim = 3
lda = LinearDiscriminantAnalysis(n_components=dim)
lda.fit(x_train[:, 1, 0, :], b_train_1)
print('Accuracy of LDA on train data', lda.score(x_train[:, 1, 0, :], b_train_1))
print('Accuracy of LDA on test data', lda.score(x_test[:, 1, 0, :], b_test_1))

### Projecting into latent space
y0_tr = lda.transform(x_train[:, 0, 0, :])
y1_tr = lda.transform(x_train[:, 1, 0, :])
y0_tst = lda.transform(x_test[:, 0, 0, :])
y1_tst = lda.transform(x_test[:, 1, 0, :])

# Train test split
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
x_train, x_test, t_train_1, t_test_1 = timeseries_train_test_split(x_, t_)

print(f'data/generated/saved_Y/y0_tr__{algorithm}_branching_simulated')
np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_branching_simulated_{b_function}', y0_tr)
np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_branching_simulated_{b_function}', y1_tr)
np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_branching_simulated_{b_function}', y0_tst)
np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_branching_simulated_{b_function}', y1_tst)
np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_branching_simulated_{b_function}', b_train_1)
np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_branching_simulated_{b_function}', b_test_1)
np.savetxt(f'data/generated/saved_Y/t_train_1__{algorithm}_branching_simulated_{b_function}', t_train_1)
np.savetxt(f'data/generated/saved_Y/t_test_1__{algorithm}_branching_simulated_{b_function}', t_test_1)
plt.show()




