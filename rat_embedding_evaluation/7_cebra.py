import numpy as np
from cebra import CEBRA
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'cebra_h'

for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    ### Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=20)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy CEBRA hybrid
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
                               hybrid=True)

    cebra_hybrid_model.fit(x_train[:, 0, 0, :], b_train_1.astype(float))

    # Projecting into latent space
    y0_tr = cebra_hybrid_model.transform(x_train[:, 0, 0, :])
    y1_tr = cebra_hybrid_model.transform(x_train[:, 1, 0, :])
    y0_tst = cebra_hybrid_model.transform(x_test[:, 0, 0, :])
    y1_tst = cebra_hybrid_model.transform(x_test[:, 1, 0, :])

    # Save the weights
    # model.save_weights(f'data/generated/{algorithm}_model_rat_{rat_name}')
    np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}', y0_tr)
    np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}', y1_tr)
    np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}', y0_tst)
    np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}', y1_tst)
    np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
