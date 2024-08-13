import sys
sys.path.append(r'../')
import os
os.chdir('..')
import numpy as np
import sklearn
from functions import *

algorithm = sys.argv[1]
rat_name = sys.argv[2]
print(algorithm, ' rat_name: ', rat_name)

file_pattern = f'../data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
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

### Behaviour decodability evaluation
r2_list = []
for i in tqdm(np.arange(10)):
	b_predictor = tf.keras.Sequential([layers.Dense(3, activation='linear')]) 
	opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
	b_predictor.compile(optimizer='adam',
								loss=tf.keras.losses.MeanSquaredError(),
								metrics=['mse'])

	history = b_predictor.fit(Y1_tr,
							B_train_1,
							epochs=100,
							batch_size=100,
							validation_data=(Y1_tst, B_test_1),
							verbose=0
							)
	
	B1_tst_pred = b_predictor(Y1_tst).numpy()
	r2_score = sklearn.metrics.r2_score(B1_tst_pred, B_test_1)
	print(r2_score)
	r2_list.append(r2_score)

# Saving the metrics
r2_list = np.array(r2_list)
np.savetxt('../data/generated/evaluation_metrics/r2_list_' + algorithm + '_rat_' +  str(rat_name), r2_list)

