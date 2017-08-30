#!/bin/sh
echo "generating doxygen for Limbo, current path $PWD"
doxygen Doxyfile

echo "generating default params"
cd ..
./waf configure
./waf default_params


echo "getting the latest benchmark result (needs to be in $HOME/limbo_benchmarks)"
# get the last benchmark
DIR=$HOME/limbo_benchmarks
BENCHMARKS=$DIR/`ls $DIR|sort -n|head -n 1`
cp $BENCHMARKS/bo_benchmarks.rst docs/benchmarks.rst
cp -r $BENCHMARKS/fig_benchmarks docs
