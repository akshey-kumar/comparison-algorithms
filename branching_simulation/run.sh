#!/bin/bash

# Default values for the arguments
ALGORITHM="BunDLeNet"
B_FUNCTION="and"

# Check if the user has passed argum ents
if [ $# -ge 1 ]; then
  ALGORITHM=$1
fi

if [ $# -ge 2 ]; then
  B_FUNCTION=$2
fi

# Run the Python script with the given arguments
python3 branching_simulation/bundlenet.py "$ALGORITHM" "$B_FUNCTION"
python3 branching_simulation/visualise_embeddings.py "$ALGORITHM" "$B_FUNCTION"