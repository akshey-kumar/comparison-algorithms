import numpy as np

# Define the algorithm and rat_name
algorithm = "BunDLeNet_HPO"
rat_name = "cicero"

# Construct the filename without the directory path
filename = f'../../optimal_hyperparameters_{algorithm}_{rat_name}.npz'

# Load the parameters from the file
with np.load(filename) as data:
    params = dict(data)  # Convert the loaded data to a dictionary

# Print the loaded parameters
print("Loaded hyperparameters:", params)
