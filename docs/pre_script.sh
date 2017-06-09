#!/bin/sh
echo "generating doxygen for Limbo, current path $PWD"
doxygen Doxyfile

echo "generating default params"
cd ..
./waf configure
./waf default_params


