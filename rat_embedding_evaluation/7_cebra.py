import sys
sys.path.append(r'../')
import os
os.chdir('..')
import numpy as np
from functions import *
import cebra.datasets
from cebra import CEBRA

algorithm = 'cebra_h'

### Embedding with CEBRA
for rat_name in ['achilles', 'gatsby','cicero', 'buddy']:
	### Load data
	hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-' + rat_name)
	
	X = hippocampus_pos.neural.numpy()
	B = hippocampus_pos.continuous_index.numpy()

	X = X - np.min(X) ### cebra doesnt work otherwise if there are negative values
	np.where(X<0)
	
	X_, B_ = prep_data(X, B, win=1)

	## Train test split 
	X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

	### Deploy CEBRA hybrid
	cebra_hybrid_model = CEBRA(model_architecture='offset10-model',
						batch_size=512,
						learning_rate=3e-4,
						temperature=1,
						output_dimension=3,
						max_iterations=5000,
						distance='cosine',
						conditional='time_delta',
						device='cuda_if_available',
						verbose=True,
						time_offsets=10,
						hybrid = True)

	cebra_hybrid_model.fit(X_train[:,0,0,:], B_train_1.astype(float))

	### Projecting into latent space
	Y0_tr = cebra_hybrid_model.transform(X_train[:,0,0,:])
	Y1_tr = cebra_hybrid_model.transform(X_train[:,1,0,:])

	Y0_tst = cebra_hybrid_model.transform(X_test[:,0,0,:])
	Y1_tst = cebra_hybrid_model.transform(X_test[:,1,0,:])

	# Save the weights
	# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
	np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_rat_' + str(rat_name), Y0_tr)
	np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_rat_' + str(rat_name), Y1_tr)
	np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_rat_' + str(rat_name), Y0_tst)
	np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_rat_' + str(rat_name), Y1_tst)
	np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_rat_' + str(rat_name), B_train_1)
	np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_rat_' + str(rat_name), B_test_1)
	

