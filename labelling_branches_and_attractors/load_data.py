import numpy as np

worm_num = 0
attractor_labels = np.loadtxt(f'attractor_labels_worm_{worm_num}.csv')
B = np.loadtxt(f'B__BunDLeNet_worm_{worm_num}')
Y = np.loadtxt(f'Y__BunDLeNet_worm_{worm_num}').astype(int)
print(attractor_labels.shape)
print(B.shape)
print(Y.shape)