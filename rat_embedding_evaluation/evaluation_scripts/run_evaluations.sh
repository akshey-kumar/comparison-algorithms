#!/bin/bash

## Performing evaluation of the microvariable (neuronal level) for all rats
# Loop for microvariable_evaluation.py
: "
for rat_name in 'achilles' 'gatsby' 'cicero' 'buddy'
do
    python3 microvariable_evaluation.py $rat_name
done
"

## Performing evaluation of embeddings of various algorithms for all rats
# Loop for behaviour_decoding.py and dynamics_predictability.py
for rat_name in 'achilles' 'gatsby' 'cicero' 'buddy'
do
    # python3 rat_embedding_evaluation/evaluation_scripts/microvariable_evaluation.py $rat_name
    for algorithm in 'BunDLeNet' 'cebra_h' 'PCA' 'CCA' 'y_shuffled_BunDLeNet' 'linear_dynamics' # 'linear_dynamics' 'point_embedding_noisy' 'y_shuffled_BunDLeNet' # 'BunDLeNet' 'cebra_h' 'PCA' 'CCA'
    do
        # python3 rat_embedding_evaluation/evaluation_scripts/behaviour_decoding.py $algorithm $rat_name
        python3 rat_embedding_evaluation/evaluation_scripts/dynamics_predictability.py $algorithm $rat_name

    done
done
