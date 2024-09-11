import sys
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

algorithm = 'BunDLeNet_HPO'

for rat_name in ['gatsby','cicero']:
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)

    def train_bundlenet(config):
        # Extract hyperparameters from the config dictionary
        learning_rate = config["learning_rate"]
        latent_dim = int(config["latent_dim"])
        n_epochs = int(config["n_epochs"])
        win = int(config["win"])

        x_, b_ = prep_data(x, b, win=win)
        x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
        model = BunDLeNet(latent_dim=latent_dim, num_behaviour=b_train_1.shape[1])
        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model,
            b_type='continuous',
            gamma=0.9,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            initialisation=(5, 20),
            validation_data=(x_test, b_test_1),
        )
        # Report validation loss (or another metric) back to Ray Tune
        train.report({"val_loss": test_history[-1, -1]})

    search_space = {
        "win": tune.loguniform(1, 50),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "latent_dim": tune.uniform(1, 10),
        "n_epochs": tune.uniform(10, 500)
    }
    # scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=500, grace_period=20, reduction_factor=2)
    search_algo = BayesOptSearch(metric="val_loss", mode="min")
    tuner = tune.Tuner(
        tune.with_parameters(train_bundlenet),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=100,
            # scheduler=scheduler,
            max_concurrent_trials=8,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric='val_loss', mode='min')
    print("Minimum validation loss:", best_result.metrics['val_loss'])
    print("Best hyperparameters found were: ", best_result.config)
    np.savez(f'optimal_hyperparameters_{algorithm}_{rat_name}.npz', **best_result.config)

    learning_rate = best_result.config["learning_rate"]
    latent_dim = int(best_result.config["latent_dim"])
    n_epochs = int(best_result.config["n_epochs"])
    win = int(best_result.config["win"])

    x_, b_ = prep_data(x, b, win=win)

    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy BunDLe Net
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=b_.shape[1])

    train_history, test_history = train_model(
        x_train,
        b_train_1,
        model,
        b_type='continuous',
        gamma=0.9,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        initialisation=(5, 20),
        validation_data=(x_test, b_test_1),
    )
    print(f'val loss: {test_history[-1, -1]}')

    # Projecting into latent space
    y0_tr = model.tau(x_train[:, 0]).numpy()
    y1_tr = model.tau(x_train[:, 1]).numpy()
    y0_tst = model.tau(x_test[:, 0]).numpy()
    y1_tst = model.tau(x_test[:, 1]).numpy()
    y0_ = model.tau(x_[:, 0]).numpy()
    y1_ = model.tau(x_[:, 1]).numpy()

    save_data = True
    if save_data:
        # Save the weights
        # model.save_weights(f'data/generated/BunDLeNet_model_rat_{rat_name}')
        print(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}')
        np.savetxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}', y0_tr)
        np.savetxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}', y1_tr)
        np.savetxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}', y0_tst)
        np.savetxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}', y1_tst)
        np.savetxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
        np.savetxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
