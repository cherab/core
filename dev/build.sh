#!/bin/bash

CORES=`nproc --all`

echo "Rebuilding Cherab extension modules (in place)..."
python setup.py build_ext -j$CORES --inplace $1 $2 $3 $4 $5
