import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

algorithm = sys.argv[1]
rat_name = sys.argv[2]
print(algorithm, ' rat_name: ', rat_name)

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
y0_tr = np.loadtxt(file_pattern.format('y0_tr'))
y1_tr = np.loadtxt(file_pattern.format('y1_tr'))
y0_tst = np.loadtxt(file_pattern.format('y0_tst'))
y1_tst = np.loadtxt(file_pattern.format('y1_tst'))
b_train_1 = np.loadtxt(file_pattern.format('b_train_1'))
b_test_1 = np.loadtxt(file_pattern.format('b_test_1'))

y0_tr = y0_tr.reshape(y0_tr.shape[0], -1)
y1_tr = y1_tr.reshape(y1_tr.shape[0], -1)
y0_tst = y0_tst.reshape(y0_tst.shape[0], -1)
y1_tst = y1_tst.reshape(y1_tst.shape[0], -1)
ydiff_tr = y1_tr - y0_tr
ydiff_tst = y1_tst - y0_tst

# Dynamics predictability evaluation
mse_list = []
for i in tqdm(np.arange(10)):
    # Scaling input and output data
    yinmax = (np.abs(y0_tr)).max()  # Parameters for scaling
    y0_tr, y0_tst = y0_tr / yinmax, y0_tst / yinmax
    ydmax = (np.abs(ydiff_tr)).max()  # Parameters for scaling
    ydiff_tr, ydiff_tst = ydiff_tr / ydmax, ydiff_tst / ydmax

    # Defining the model
    model_ydiff_f_yt = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='linear')])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model_ydiff_f_yt.compile(optimizer=opt,
                             loss='mse',
                             metrics=['mse'])

    history = model_ydiff_f_yt.fit(y0_tr,
                                   ydiff_tr,
                                   epochs=50,
                                   batch_size=1000,
                                   validation_data=(y0_tst, ydiff_tst),
                                   verbose=0
                                   )

    # Predictions
    ydiff_tr_pred = model_ydiff_f_yt(y0_tr).numpy()
    ydiff_tst_pred = model_ydiff_f_yt(y0_tst).numpy()

    # Inverse scaling the data
    ydiff_tr_pred, ydiff_tr, y0_tr = ydiff_tr_pred * ydmax, ydiff_tr * ydmax, y0_tr * yinmax
    ydiff_tst_pred, ydiff_tst, y0_tst = ydiff_tst_pred * ydmax, ydiff_tst * ydmax, y0_tst * yinmax

    y1_tr_pred = y0_tr + ydiff_tr_pred
    y1_tst_pred = y0_tst + ydiff_tst_pred

    # Evaluation
    flat_partial = lambda x: x.reshape(x.shape[0],-1)
    baseline_tr = mean_squared_error(flat_partial(y1_tr), flat_partial(y0_tr))
    modelmse_tr = mean_squared_error(flat_partial(y1_tr), flat_partial(y1_tr_pred))
    baseline_tst = mean_squared_error(flat_partial(y1_tst), flat_partial(y0_tst))
    modelmse_tst = mean_squared_error(flat_partial(y1_tst), flat_partial(y1_tst_pred))
    print(baseline_tr)
    print(modelmse_tr)
    print(baseline_tst)
    print(modelmse_tst)

    mse_list.append([baseline_tr, modelmse_tr, baseline_tst, modelmse_tst])

# Saving the metrics
mse_list = np.array(mse_list)
os.makedirs('data/generated/rat_evaluation_metrics', exist_ok=True)
np.savetxt(f'data/generated/rat_evaluation_metrics/mse_list_{algorithm}_rat_{rat_name}', mse_list)
