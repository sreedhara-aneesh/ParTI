#!/bin/bash

set -x

#TENSOR_DIR=/mnt/beegfs/$USER/csc548/parti/tensors

COMMAND=./build/examples/mttkrp_reorder

if [ -z ${TENSOR_DIR+x} ]; then
    echo "ERROR: env var TENSOR_DIR not specified."
    exit
fi

if [ -z ${OMP_NUM_THREADS+x} ]; then
    echo "WARNING: env var OMP_NUM_THREADS not specified, defaulting to 1"
    OMP_NUM_THREADS=1
fi

./get_data.sh
./build.sh

echo "----------"
echo "COMMAND INFORMATION"
echo "----------"
$COMMAND --help

for TENSOR in "$TENSOR_DIR"/*.tns; do

    if [ -f "$TENSOR" ]; then
        echo "----------"
        echo "TENSOR: $TENSOR"
        echo "----------"

        for E in 1 2 3 4 5; do
            echo "----------"
            echo "REORDER: e = $E"
            echo "----------"

            $COMMAND \
                -i "$TENSOR" \
                -e $E \
                -d -1 \
                -t "$OMP_NUM_THREADS"
                # -m -1
                # -n 1


        done

    fi

done