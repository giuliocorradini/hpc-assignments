#!/bin/bash

# Define dataset sizes
DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET")

# Loop through each dataset size and compile the program
for DATASET in "${DATASETS[@]}"; do
    #echo "Compiling with $DATASET"
    make VERSION=gramschmidt.cu NVCFLAGS="--expt-relaxed-constexpr -D$DATASET" clean all > /dev/null 2> /dev/null

    # Run the program 5 times and calculate the average result
    total=0
    for i in {1..5}; do
        result=$(./gramschmidt.exe)
        total=$(echo "$total + $result" | bc)
    done
    average=$(echo "scale=3; $total / 5" | bc)
    echo "Average result for $DATASET: $average"
done
