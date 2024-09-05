import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics

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

y0_tr = y0_tr.reshape(y0_tr.shape[0],-1)
y1_tr = y1_tr.reshape(y1_tr.shape[0],-1)
y0_tst = y0_tst.reshape(y0_tst.shape[0],-1)
y1_tst = y1_tst.reshape(y1_tst.shape[0],-1)
ydiff_tr = y1_tr - y0_tr
ydiff_tst = y1_tst - y0_tst

# Behaviour decodability evaluation
r2_list = []
for i in tqdm(np.arange(10)):
	b_predictor = tf.keras.Sequential([
		tf.keras.layers.Dense(3, activation='linear')
	])
	opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
	b_predictor.compile(
		optimizer='adam',
		loss=tf.keras.losses.MeanSquaredError(),
		metrics=['mse']
	)
	history = b_predictor.fit(
		y1_tr,
		b_train_1,
		epochs=100,
		batch_size=100,
		validation_data=(y1_tst, b_test_1),
		verbose=0
	)
	
	b1_tst_pred = b_predictor(y1_tst).numpy()
	r2_score = metrics.r2_score(b1_tst_pred, b_test_1)
	print(r2_score)
	r2_list.append(r2_score)

# Saving the metrics
r2_list = np.array(r2_list)
os.makedirs('data/generated/rat_evaluation_metrics', exist_ok=True)
np.savetxt(f'data/generated/rat_evaluation_metrics/r2_list_{algorithm}_rat_{rat_name}', r2_list)
