import sys
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

algorithm = sys.argv[1]
b_function = sys.argv[2]

# Plotting
y0_tr = np.loadtxt(f'data/generated/saved_Y/y0_tr__{algorithm}_branching_simulated_{b_function}')
y0_tr += np.random.normal(loc=0, scale=0.00005, size=y0_tr.shape)
y1_tr = np.loadtxt(f'data/generated/saved_Y/y1_tr__{algorithm}_branching_simulated_{b_function}')
y1_tr += np.random.normal(loc=0, scale=0.00005, size=y1_tr.shape)
y0_tst = np.loadtxt(f'data/generated/saved_Y/y0_tst__{algorithm}_branching_simulated_{b_function}')
y0_tst  += np.random.normal(loc=0, scale=0.00005, size=y0_tst.shape)
y1_tst = np.loadtxt(f'data/generated/saved_Y/y1_tst__{algorithm}_branching_simulated_{b_function}')
y1_tst  += np.random.normal(loc=0, scale=0.00005, size=y1_tst.shape)
b_train_1 = np.loadtxt(f'data/generated/saved_Y/b_train_1__{algorithm}_branching_simulated_{b_function}')
b_test_1 = np.loadtxt(f'data/generated/saved_Y/b_test_1__{algorithm}_branching_simulated_{b_function}')
t_train_1 = np.loadtxt(f'data/generated/saved_Y/t_train_1__{algorithm}_branching_simulated_{b_function}')
t_test_1 = np.loadtxt(f'data/generated/saved_Y/t_test_1__{algorithm}_branching_simulated_{b_function}')

# b_names = ['start', '1 (True)', '2 (False)']


# plotting points coloured by behaviour
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
ax.axis('off')
tr_pts = ax.scatter(y0_tr[:, 0], y0_tr[:, 1], y0_tr[:, 2], c=b_train_1, cmap='plasma', s=0.5)
tst_pts = ax.scatter(y0_tst[:, 0], y0_tst[:, 1], y0_tst[:, 2], c=b_test_1, cmap='plasma', s=10)
plt.colorbar(tr_pts)

# plotting points coloured by time within trial
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
ax.axis('off')
tr_pts = ax.scatter(y0_tr[:, 0], y0_tr[:, 1], y0_tr[:, 2], c=t_train_1, s=0.5)
tst_pts = ax.scatter(y0_tst[:, 0], y0_tst[:, 1], y0_tst[:, 2], c=t_test_1, s=10)
plt.colorbar(tr_pts)
'''
#  plotting dynamics
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

vis = LatentSpaceVisualiser(
    y=y0_tr,
    b=b_train_1.astype(int),
    b_names=np.unique(b_train_1),
    show_points=True

)
fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.1)

vis.plot_latent_timeseries()

vis = LatentSpaceVisualiser(
    y=y0_tst,
    b=b_test_1.astype(int),
    b_names=np.unique(b_test_1),
    show_points=True
)
fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.0001)
'''
plt.show()



