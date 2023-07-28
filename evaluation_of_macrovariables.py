import sys
sys.path.append(r'../')
import numpy as np
from functions import *

for algorithm in ['PCA', 'tsne', 'autoencoder', 'ArAe', 'BunDLeNet', 'cebra_h']
	for worm_num in range(5):

		Y0_tr = np.loadtxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' +  str(worm_num))
		Y1_tr = np.loadtxt('data/generated/saved_Y/Y1_tr__' +  algorithm  + '_worm_' +  str(worm_num))
		Y0_tst = np.loadtxt('data/generated/saved_Y/Y0_tst__' +  algorithm  + '_worm_' +  str(worm_num))
		Y1_tst = np.loadtxt('data/generated/saved_Y/Y1_tst__' +  algorithm  + '_worm_' +  str(worm_num))
		B_train_1 = np.loadtxt('data/generated/saved_Y/B_train_1__' +  algorithm  + '_worm_' +  str(worm_num))
		B_test_1 = np.loadtxt('data/generated/saved_Y/B_test_1__' +  algorithm  + '_worm_' +  str(worm_num))

		Y0_tr = Y0_tr.reshape(Y0_tr.shape[0],-1)
		Y1_tr = Y1_tr.reshape(Y1_tr.shape[0],-1)
		Y0_tst = Y0_tst.reshape(Y0_tst.shape[0],-1)
		Y1_tst = Y1_tst.reshape(Y1_tst.shape[0],-1)
		Ydiff_tr = Y1_tr - Y0_tr
		Ydiff_tst = Y1_tst - Y0_tst

		### Behaviour decodability
		acc_list = []
		for i in tqdm(np.arange(20)):
		    b_predictor = tf.keras.Sequential([layers.Dense(8, activation='linear')]) 
		    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
		    b_predictor.compile(optimizer='adam',
		                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		                  metrics=['accuracy'])

		    history = b_predictor.fit(Y1_tr,
		                          B_train_1,
		                          epochs=100,
		                          batch_size=100,
		                          validation_data=(Y1_tst, B_test_1),
		                          verbose=0
		                          )

		    B1_tst_pred = b_predictor(Y1_tst).numpy().argmax(axis=1)
		    acc_list.append(accuracy_score(B1_tst_pred, B_test_1))

		# Saving the metrics
		acc_list = np.array(acc_list)
		np.savetxt('data/generated/evaluation_metrics/acc_list_' + algorithm + '_worm_' +  str(worm_num),acc_list)


		### Chance accuracy estimation
		chance_acc = np.zeros(500)
		for i, _ in enumerate(chance_acc):
		    B_perm = np.random.choice(B_test_1, size=B_test_1.shape)
		    chance_acc[i] = accuracy_score(B_perm, B_test_1)
		print('Chance prediction accuracy: ', chance_acc.mean().round(3), ' pm ', chance_acc.std().round(3))


		### Dynamics predictability (with linear autoregressor)
		mse_list = []
		for i in tqdm(np.arange(20)):
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
		np.savetxt('data/generated/evaluation_metrics/mse_list_' + algorithm + '_worm_' +  str(worm_num), mse_list)


