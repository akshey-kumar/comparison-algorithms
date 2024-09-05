import sys
from matplotlib import pyplot as plt
#sys.path.append(r'../')
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import sklearn
import os

# os.chdir('../..')

rat_name = sys.argv[1]

# Load data
data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
np.where(x < 0)
x_, b_ = prep_data(x, b, win=20)

# Train test split
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
# X1_tr = X_train[:,1,:,:]
# X1_tst = X_test[:,1,:,:]

# Behavioural prediction accuracy directly from neuronal data
# Behavioural decodablity from neuronal level (microvariable)
r2_list = []
for i in tqdm(np.arange(50)):
    b_predictor = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    b_predictor.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mse'])

    history = b_predictor.fit(
        x_train,
        b_train_1,
        epochs=100,
        batch_size=100,
        validation_data=(x_test, b_test_1),
        verbose=0
    )
    # Summarize history for accuracy
    # plt.plot(history.history['mse'])
    # plt.plot(history.history['val_mse'])
    # plt.show()
    b1_tst_pred = b_predictor(x_test).numpy()

    r2_score = sklearn.metrics.r2_score(b1_tst_pred, b_test_1)
    r2_list.append(r2_score)

r2_list = np.array(r2_list)
np.savetxt('data/generated/rat_evaluation_metrics/r2_list_X_rat_' + str(rat_name), r2_list)

# Estimating the chance accuracy of behaviour decoding
chance_r2 = np.zeros(500)
for i, _ in enumerate(chance_r2):
    rand_idx = np.random.choice(np.arange(b_test_1.shape[0]), size=b_test_1.shape[0])
    b_perm = b_test_1[rand_idx]

    chance_r2[i] = sklearn.metrics.r2_score(b_perm, b_test_1)

print('Chance prediction accuracy: ', chance_r2.mean().round(3), ' pm ', chance_r2.std().round(3))
np.savetxt('data/generated/rat_evaluation_metrics/r2_list_chance_rat_' + str(rat_name), chance_r2)
