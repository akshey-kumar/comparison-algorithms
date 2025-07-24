#!/bin/bash
set -e

algorithms=(
    #1_1_PCA_time_delay_embedding.py
    #1_PCA.py
    2_1_autoencoder_time_delay_embedding.py
    3_1_ArAe_time_delay_embedding.py
    4_1_BunDLeNet_linear.py
    4_2_BunDLeNet_win_1.py
    4_BunDLeNet.py
    5_tsne.py
    6_1_LDA_time_delay_embedding.py
    6_LDA.py
    7_cebra.py
)

for algorithm in "${algorithms[@]}"; do
    echo "Running $algorithm"
    python3 c_elegans_embedding_evaluation/"$algorithm"
done