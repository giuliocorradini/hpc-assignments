#!/bin/bash

# Define dataset sizes
DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET")

# Loop through each dataset size and compile the program
for DATASET in "${DATASETS[@]}"; do
    echo "Profiling with $DATASET"
    make VERSION=gramschmidt.cu NVCFLAGS="--expt-relaxed-constexpr -D$DATASET" clean all profile
done