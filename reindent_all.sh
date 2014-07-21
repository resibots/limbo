#!/bin/sh
find .|egrep "(cpp|hpp)$"|xargs astyle -L -A14 -N -H -c -p --indent=spaces=2
find .|grep ".orig$"|xargs rm
