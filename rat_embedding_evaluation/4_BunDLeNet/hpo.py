#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import traceback

def main():
    # Configuration
    algorithm = 'BunDLeNet_HPO'
    rat_name = 'achilles'  # ['achilles', 'gatsby', 'cicero', 'buddy']
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    def train_bundlenet(config):
        try:
            # Extract hyperparameters
            learning_rate = config["learning_rate"]
            latent_dim = int(config["latent_dim"])
            n_epochs = int(config["n_epochs"])
            win = int(config["win"])

            # Prepare data
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
                report_ray_tune=True,
            )

        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()

    max_epochs = 500
    search_space = {
        "win": tune.loguniform(1, 50),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "latent_dim": tune.uniform(1, 5),
        "n_epochs": tune.loguniform(10, max_epochs)
    }

    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=max_epochs,
        grace_period=10,
        reduction_factor=2
    )

    search_algo = BayesOptSearch(metric="val_loss", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(train_bundlenet),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=100,
            scheduler=scheduler,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    # tuner = tune.Tuner.restore("/Users/aksheykumar/ray_results/train_bundlenet_2024-09-13_16-38-41", trainable=tune.with_parameters(train_bundlenet))

    results = tuner.fit()

    # Save the best result
    best_result = results.get_best_result(metric='val_loss', mode='min')
    print("Minimum validation loss:", best_result.metrics['val_loss'])
    print("Best hyperparameters found were: ", best_result.config)

    # Save best hyperparameters to a file
    results_dir = 'data/generated/hpo/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_{rat_name}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Minimum validation loss: {best_result.metrics['val_loss']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")


if __name__ == "__main__":
    main()
