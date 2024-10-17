import sys
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

algorithm = sys.argv[1]
worm_num = sys.argv[2]
b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

Y0_tr = np.loadtxt(f'data/generated/saved_Y/Y0_tr__{algorithm}_worm_{worm_num}')
Y1_tr = np.loadtxt(f'data/generated/saved_Y/Y1_tr__{algorithm}_worm_{worm_num}')
Y0_tst = np.loadtxt(f'data/generated/saved_Y/Y0_tst__{algorithm}_worm_{worm_num}')
Y1_tst = np.loadtxt(f'data/generated/saved_Y/Y1_tst__{algorithm}_worm_{worm_num}')
B_train_1 = np.loadtxt(f'data/generated/saved_Y/B_train_1__{algorithm}_worm_{worm_num}').astype(int)
B_test_1 = np.loadtxt(f'data/generated/saved_Y/B_test_1__{algorithm}_worm_{worm_num}').astype(int)

# Discrete variable plotting
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

vis = LatentSpaceVisualiser(
    y=Y0_tr,
    b=B_train_1.astype(int),
    b_names=b_names,
    show_points=True

)
fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.0001)

vis.plot_latent_timeseries()

vis = LatentSpaceVisualiser(
    y=Y0_tst,
    b=B_test_1.astype(int),
    b_names=b_names,
    show_points=True
)
#fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.0001)

plt.show()

