import sys
# sys.path.append(r'../')
import numpy as np

from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet'

for rat_name in ['achilles', 'gatsby','cicero', 'buddy']:
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=20)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy BunDLe Net

    model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1])

    train_history, test_history = train_model(
        x_train,
        b_train_1,
        model,
        b_type='continuous',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
        initialisation=None,
        validation_data=(x_test, b_test_1),
    )

    plt.figure()
    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Train loss $\mathcal{L}$"
    ]):
        plt.plot(train_history[:, i], label=label)
    plt.plot(test_history[:, -1], label='Test loss', linestyle='--')
    plt.legend()


    # Projecting into latent space
    y0_tr = model.tau(x_train[:, 0]).numpy()
    y1_tr = model.tau(x_train[:, 1]).numpy()

    y0_tst = model.tau(x_test[:, 0]).numpy()
    y1_tst = model.tau(x_test[:, 1]).numpy()

    # Save the weights
    # model.save_weights(f'data/generated/BunDLeNet_model_rat_{rat_name}')
    print(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}')
    np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}', y0_tr)
    np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}', y1_tr)
    np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}', y0_tst)
    np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}', y1_tst)
    np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)


plt.show()

