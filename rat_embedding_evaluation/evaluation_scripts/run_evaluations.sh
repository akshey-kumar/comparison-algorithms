#!/bin/bash

## Performing evaluation of the microvariable (neuronal level) for all rats
# Loop for microvariable_evaluation.py
'''
for rat_name in 'achilles' 'gatsby' 'cicero' 'buddy'
do
    python3 microvariable_evaluation.py $rat_name
done
'''

## Performing evaluation of embeddings of various algorithms for all rats
# Loop for behaviour_decoding.py and dynamics_predictability.py
for rat_name in 'achilles' 'gatsby' 'cicero' 'buddy'
do
    for algorithm in 'BunDLeNet' 'cebra_h'
    do
        #python3 behaviour_decoding.py $algorithm $rat_name
        python3 dynamics_predictability.py $algorithm $rat_name
    done
done
