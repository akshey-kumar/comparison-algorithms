import sys
from functions import *

algorithm = sys.argv[1]
rat_name = sys.argv[2]
# Plotting

y0_tr = np.loadtxt(f'data/generated/saved_Y/y0_tr__{algorithm}_rat_{rat_name}')
y1_tr = np.loadtxt(f'data/generated/saved_Y/y1_tr__{algorithm}_rat_{rat_name}')
y0_tst = np.loadtxt(f'data/generated/saved_Y/y0_tst__{algorithm}_rat_{rat_name}')
y1_tst = np.loadtxt(f'data/generated/saved_Y/y1_tst__{algorithm}_rat_{rat_name}')
b_train_1 = np.loadtxt(f'data/generated/saved_Y/b_train_1__{algorithm}_rat_{rat_name}')
b_test_1 = np.loadtxt(f'data/generated/saved_Y/b_test_1__{algorithm}_rat_{rat_name}')

### Continuous variable plotting
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
# ax.axis('off')
tr_pts = ax.scatter(y0_tr[:, 0], y0_tr[:, 1], y0_tr[:, 2], c=b_train_1[:,0], s = 0.5)
tst_pts= ax.scatter(y0_tst[:, 0], y0_tst[:, 1], y0_tst[:, 2], c=b_test_1[:,0], s = 10)
plt.colorbar(tr_pts)

### Discrete variable plotting
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
plot_ps_(fig, ax, Y=y0_tr, B=b_train_1[:,1].astype(int), state_names=['0', '1'], show_points=False)
plot_ps_(fig, ax, Y=y0_tst, B=b_test_1[:,1].astype(int), state_names=['0', '1'], show_points=True)

plt.show()