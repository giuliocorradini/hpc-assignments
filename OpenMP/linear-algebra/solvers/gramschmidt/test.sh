#!/bin/bash

BENCHMARK_RESULTS_FILE=results.txt

# Copy the specified file in place of gramschmidt.c
function prepare_src() {
    cp $1 gramschmidt.c
    echo $1 >> $BENCHMARK_RESULTS_FILE
}

function compile() {
    make EXT_CFLAGS="-DPOLYBENCH_TIME" clean all
}

function run() {
    for r in $(seq $1); do
        ./gramschmidt_acc | tee -a $BENCHMARK_RESULTS_FILE
    done
}

echo "Testing base version"
prepare_src gramschmidt-original.c
compile > /dev/null
run 5 times

echo "Testing static-optimized version"
prepare_src gramschmidt-static-opt.c
compile > /dev/null
run 5 times

echo "Testing worker threads"
prepare_src gramschmidt-workerthreads.c
compile > /dev/null
run 5 times

echo "Testing transpose threads"
prepare_src gramschmidt-transpose.c
compile > /dev/null
run 5 times