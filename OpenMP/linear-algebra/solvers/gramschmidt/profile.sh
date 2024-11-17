#!/bin/bash

BENCHMARK_RESULTS_FILE=profile-results.txt
PERF_FLAGS=task-clock,context-switches,page-faults,cycles,cpu-migrations,branches,branch-misses,instructions,cache-misses,cache-references,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores


rm $BENCHMARK_RESULTS_FILE

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
        perf stat -r 5 -e $PERF_FLAGS -o temp_r.txt ./gramschmidt_acc
        cat temp_r.txt >> $BENCHMARK_RESULTS_FILE
        rm temp_r.txt
    done
}

echo "Testing base version"
prepare_src gramschmidt-original.c
compile > /dev/null
run 1 times

echo "Testing static-optimized version"
prepare_src gramschmidt-static-opt.c
compile > /dev/null
run 1 times

echo "Testing worker threads"
prepare_src gramschmidt-workerthreads.c
compile > /dev/null
run 1 times

echo "Testing transpose threads"
prepare_src gramschmidt-transpose.c
compile > /dev/null
run 1 times
