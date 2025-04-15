import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

b_function = 'AND_distractors'
n_trials = 200
t_steps_per_trial = 100
n_distractors = 1
inputs = np.random.choice([1, -1], size=(n_trials, 3 + n_distractors))

x = np.zeros((n_trials, t_steps_per_trial, 3 + n_distractors))
b = np.zeros((n_trials, t_steps_per_trial,))
'''
# trinary behaviour distractors
def b_func(x1, x2, x3):
    if x1==0:
        return 1
    else:
        if x2==1:
            return 3
        else:
            if x3==1:
                return 2
            else:
                return 1
'''
def b_func(x1, x2, x3):
    if x1==0:
        return 0
    else:
        if x2==0:
            return 0
        else:
            if x3==0:
                return 0
            else:
                return 1


for i in range(n_trials):
    t_1, t_2, t_3, t_b = 20, 40, 60, 80
    x[i, t_1:, 0] = inputs[i, 0]
    x[i, t_2:, 1] = inputs[i, 1]
    x[i, t_3:, 2] = inputs[i, 2]
    for k in range(3,3 + n_distractors): # distractors
        t_k = np.random.randint(100)
        x[i, t_k:, k] = inputs[i, k]
    b[i, t_b:] = b_func((inputs[i, 0] == 1), (inputs[i, 1] == 1), (inputs[i, 2] == 1))

# Apply Gaussian smoothing
x = gaussian_filter1d(x, sigma=1, axis=1)

# Apply Gaussian noise
print(x.shape, b.shape)
x = x + np.random.normal(loc=0, scale=0.01, size=x.shape)

# Plot
plt.figure(figsize=(10, 6))
for i in range(50):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for j in range(4):
        plt.plot(x[i,:, j], c=color_list[j], alpha=0.9)
    plt.plot(b[i], alpha=0.1, c='k')
plt.show()

# Save x and b
np.save(f"branching_simulation/data/x_{b_function}.npy", x)
np.save(f"branching_simulation/data/b_{b_function}.npy", b)

# Load x and b
x = np.load(f"branching_simulation/data/x_{b_function}.npy")
b = np.load(f"branching_simulation/data/b_{b_function}.npy")
print(np.unique(b))