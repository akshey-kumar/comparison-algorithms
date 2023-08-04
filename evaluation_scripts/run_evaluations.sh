#!/bin/bash

# Loop for microvariable_evaluation.py
# for worm_num in 0 1 2 3 4
# do
#     python3 microvariable_evaluation.py $worm_num
# done

# Loop for behaviour_decoding.py and dynamics_predictability.py
for worm_num in 0 1 2 3 4
do
    for algorithm in 'PCA' 'tsne' 'autoencoder' 'ArAe' 'BunDLeNet' 'cebra_h'
    do
        python3 behaviour_decoding.py $algorithm $worm_num
        python3 dynamics_predictability.py $algorithm $worm_num
    done
done
