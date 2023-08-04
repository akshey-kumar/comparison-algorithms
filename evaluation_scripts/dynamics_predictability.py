import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = sys.argv[1]
worm_num = int(sys.argv[2])
print(algorithm, ' worm_num: ', worm_num)

file_pattern = f'../data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1'))
B_test_1 = np.loadtxt(file_pattern.format('B_test_1'))

Y0_tr = Y0_tr.reshape(Y0_tr.shape[0],-1)
Y1_tr = Y1_tr.reshape(Y1_tr.shape[0],-1)
Y0_tst = Y0_tst.reshape(Y0_tst.shape[0],-1)
Y1_tst = Y1_tst.reshape(Y1_tst.shape[0],-1)
Ydiff_tr = Y1_tr - Y0_tr
Ydiff_tst = Y1_tst - Y0_tst

### Dynamics predictability (with linear autoregressor)
mse_list = []
for i in tqdm(np.arange(10)):
    ### Scaling input and output data
    Yinmax = (np.abs(Y0_tr)).max() # Parameters for scaling
    Y0_tr, Y0_tst = Y0_tr/Yinmax, Y0_tst/Yinmax
    Ydmax = (np.abs(Ydiff_tr)).max() # Parameters for scaling
    Ydiff_tr, Ydiff_tst = Ydiff_tr/Ydmax, Ydiff_tst/Ydmax

    # Defining the model
    model_ydiff_f_yt = tf.keras.Sequential([tf.keras.layers.Dense(3, activation='linear')])
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    model_ydiff_f_yt.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mse'])

    history = model_ydiff_f_yt.fit(Y0_tr,
                          Ydiff_tr,
                          epochs=50,
                          batch_size=100,
                          validation_data=(Y0_tst, Ydiff_tst),
                          verbose=0
                          )

    # Preidctions
    Ydiff_tr_pred = model_ydiff_f_yt(Y0_tr).numpy()
    Ydiff_tst_pred = model_ydiff_f_yt(Y0_tst).numpy()

    # Inverse scaling the data
    Ydiff_tr_pred, Ydiff_tr, Y0_tr = Ydiff_tr_pred*Ydmax, Ydiff_tr*Ydmax, Y0_tr*Yinmax
    Ydiff_tst_pred, Ydiff_tst, Y0_tst = Ydiff_tst_pred*Ydmax, Ydiff_tst*Ydmax, Y0_tst*Yinmax

    Y1_tr_pred = Y0_tr + Ydiff_tr_pred
    Y1_tst_pred = Y0_tst + Ydiff_tst_pred

    # Evaluation
    baseline_tr  = mean_squared_error(flat_partial(Y1_tr), flat_partial(Y0_tr))
    modelmse_tr = mean_squared_error(flat_partial(Y1_tr), flat_partial(Y1_tr_pred))
    baseline_tst  = mean_squared_error(flat_partial(Y1_tst), flat_partial(Y0_tst))
    modelmse_tst = mean_squared_error(flat_partial(Y1_tst), flat_partial(Y1_tst_pred))
    
    mse_list.append([baseline_tr, modelmse_tr, baseline_tst, modelmse_tst])

# Saving the metrics
mse_list = np.array(mse_list)
np.savetxt('../data/generated/evaluation_metrics/mse_list_' + algorithm + '_worm_' +  str(worm_num), mse_list)
