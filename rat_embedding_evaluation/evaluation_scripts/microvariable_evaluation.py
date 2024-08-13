import sys
sys.path.append(r'../')
import numpy as np
from functions import *
import cebra.datasets
import sklearn
import os
os.chdir('../..')

rat_name = sys.argv[1]

hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-' + rat_name)

X = hippocampus_pos.neural.numpy()
B = hippocampus_pos.continuous_index.numpy()
X = X - np.min(X) ### cebra doesnt work otherwise if there are negative values

X_, B_ = prep_data(X, B, win=15)
X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
X1_tr = X_train[:,1,:,:]
X1_tst = X_test[:,1,:,:]

### Behavioural prediction accuracy directly from neuronal data
### Behavioural decodablity from neuronal level (microvariable)
r2_list = []
for i in tqdm(np.arange(50)):

	b_predictor = tf.keras.Sequential([
		layers.Flatten(),
		layers.Dense(3, activation='linear')
	]) 
	opt = tf.keras.optimizers.Adam(learning_rate=0.01)
	b_predictor.compile(optimizer='adam',
						loss=tf.keras.losses.MeanSquaredError(),
						metrics=['mse'])

	history = b_predictor.fit(X1_tr,
						  B_train_1,
						  epochs=100,
						  batch_size=100,
						  validation_data=(X1_tst, B_test_1),
						  verbose=0
						  )
		# Summarize history for accuracy
	#plt.plot(history.history['accuracy'])
	#plt.plot(history.history['val_accuracy'])
	#plt.show()
	B1_tst_pred = b_predictor(X1_tst).numpy()


	r2_score = sklearn.metrics.r2_score(B1_tst_pred, B_test_1)
	r2_list.append(r2_score)

r2_list = np.array(r2_list)
np.savetxt('data/generated/evaluation_metrics/acc_list_X_rat_' +  str(rat_name), r2_list)



### Estimating the chance accuracy of behaviour decoding
chance_r2 = np.zeros(500)
for i, _ in enumerate(chance_r2):
	rand_idx = np.random.choice(np.arange(B_test_1.shape[0]), size=B_test_1.shape[0])
	B_perm = B_test_1[rand_idx]
	#B_perm = np.random.choice(B_test_1, size=B_test_1.shape)
	
	chance_r2[i] = sklearn.metrics.r2_score(B_perm, B_test_1)

print('Chance prediction accuracy: ', chance_r2.mean().round(3), ' pm ', chance_r2.std().round(3))
np.savetxt('data/generated/evaluation_metrics/acc_list_chance_rat_' +  str(rat_name), chance_r2)