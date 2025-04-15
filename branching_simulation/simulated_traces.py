import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

b_function = 'trinary_behaviour_exp'
n_trials = 50
t_steps_per_trial = 100
inputs = np.random.choice([1, -1], size=(n_trials, 3))

x = np.zeros((n_trials, t_steps_per_trial, 3))
b = np.zeros((n_trials, t_steps_per_trial,))

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



for i in range(n_trials):
    t_1, t_2, t_3, t_b = 20, 40, 60, 80
    x[i, t_1:, 0] = inputs[i, 0]
    x[i, t_2:, 1] = inputs[i, 1]
    x[i, t_3:, 2] = inputs[i, 2]
    b[i, t_b:] = b_func((inputs[i, 0] == 1), (inputs[i, 1] == 1), (inputs[i, 2] == 1))
    b[i, -1] = 4 #
    # b[i, t_b:] = 2 - ((inputs[i, 0] == 1) & (inputs[i, 1] == 1) & (inputs[i, 2] == 1))
    # b[i, t_b:] = 2 - (((inputs[i, 0] == 1) & (inputs[i, 1] == -1) & (inputs[i, 2] == 1)) |
    #                   ((inputs[i, 0] == -1) & (inputs[i, 1] == 1) & (inputs[i, 2] == -1)) |
    #                   (~(inputs[i, 0] == 1) & (inputs[i, 1] == 1) & ~(inputs[i, 2] == 1)))

# Apply Gaussian smoothing
x = gaussian_filter1d(x, sigma=3, axis=1)

# Apply Gaussian noise
print(x.shape, b.shape)
x = x + np.random.normal(loc=0, scale=0.01, size=x.shape)

# Plot the original and smoothed x for the first trial to visualize the difference
plt.figure(figsize=(10, 6))
plt.plot(x[0, :, 0], color='blue')
plt.plot(x[0, :, 1], color='red')
plt.show()

# Save x and b
np.save(f"branching_simulation/data/x_{b_function}.npy", x)
np.save(f"branching_simulation/data/b_{b_function}.npy", b)

# Load x and b
x = np.load(f"branching_simulation/data/x_{b_function}.npy")
b = np.load(f"branching_simulation/data/b_{b_function}.npy")
print(np.unique(b))