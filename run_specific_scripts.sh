#!/bin/bash

# Loop through the command-line arguments (script filenames)
for script in "$@"; do
  echo "Running $script..."
  python3 "$script"
  echo "Finished running $script"
done
