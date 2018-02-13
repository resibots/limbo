#!/bin/sh
set -x
echo "generating doxygen for Limbo, current path $PWD"
doxygen Doxyfile

echo "generating default params"
cd ..
./waf configure
./waf default_params


echo "getting the latest benchmark result (needs to be in $HOME/limbo_benchmarks)"
# get the last benchmark
DIR=$HOME/limbo_benchmarks
BENCHMARKS=$DIR/`ls -t $DIR | head -n 1`
cp $BENCHMARKS/bo_benchmarks.rst docs/bo_benchmarks.rst
cp -r $BENCHMARKS/fig_benchmarks docs

echo "getting the latest regression benchmark result (needs to be in $HOME/limbo_reg_benchmarks)"
DIR=$HOME/limbo_reg_benchmarks
BENCHMARKS=$DIR/`ls -t $DIR | head -n 1`
cp $BENCHMARKS/regression_benchmarks.rst docs/reg_benchmarks.rst
cp -r $BENCHMARKS/regression_benchmark_figs docs/fig_benchmarks

