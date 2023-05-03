#!/bin/bash

set -x

#TENSOR_DIR=/mnt/beegfs/$USER/csc548/parti/tensors

if [ -z ${TENSOR_DIR+x} ]; then
    echo "ERROR: env var TENSOR_DIR not specified."
    exit
fi

cd "$TENSOR_DIR" || exit

if [ ! -f "chicago-crime-comm-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz
    gunzip chicago-crime-comm.tns.gz &&
    rm -rf chicago-crime-comm.tns.gz &&
    echo "4" > chicago-crime-comm-parti.tns &&
    echo "6186 24 77 32" >> chicago-crime-comm-parti.tns &&
    cat chicago-crime-comm.tns >> chicago-crime-comm-parti.tns &&
    rm chicago-crime-comm.tns
fi

if [ ! -f "flickr-4d-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-4d.tns.gz
    gunzip flickr-4d.tns.gz &&
    rm -rf flickr-4d.tns.gz &&
    echo "4" > flickr-4d-parti.tns &&
    echo "319686 28153045 1607191 731" >> flickr-4d-parti.tns &&
    cat flickr-4d.tns >> flickr-4d-parti.tns &&
    rm flickr-4d.tns
fi

if [ ! -f "nell-2-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
    gunzip nell-2.tns.gz &&
    rm -rf nell-2.tns.gz &&
    echo "3" > nell-2-parti.tns &&
    echo "12092 9184 28818" >> nell-2-parti.tns &&
    cat nell-2.tns >> nell-2-parti.tns &&
    rm nell-2.tns
fi

if [ ! -f "enron-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz
    gunzip enron.tns.gz &&
    rm -rf enron.tns.gz &&
    echo "4" > enron-parti.tns &&
    echo "6066 5699 244268 1176" >> enron-parti.tns &&
    cat enron.tns >> enron-parti.tns &&
    rm enron.tns
fi

if [ ! -f "delicious-4d-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-4d.tns.gz
    gunzip delicious-4d.tns.gz &&
    rm -rf delicious-4d.tns.gz &&
    echo "4" > delicious-4d-parti.tns &&
    echo "532924 x 17262471 x 2480308 x 1443" >> delicious-4d-parti.tns &&
    cat delicious-4d.tns >> delicious-4d-parti.tns &&
    rm delicious-4d.tns
fi

if [ ! -f "nips-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz
    gunzip nips.tns.gz &&
    rm -rf nips.tns.gz &&
    echo "4" > nips-parti.tns &&
    echo "2482 2862 14036 17" >> nips-parti.tns &&
    cat nips.tns >> nips-parti.tns &&
    rm nips.tns
fi

if [ ! -f "uber-parti.tns" ]; then
    wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz
    gunzip uber.tns.gz &&
    rm -rf uber.tns.gz &&
    echo "4" > uber-parti.tns &&
    echo "183 24 1140 1717" >> uber-parti.tns &&
    cat uber.tns >> uber-parti.tns &&
    rm uber.tns
fi