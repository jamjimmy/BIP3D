#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-12050}
export WANDB_API_KEY="154d2536f85d4b2adc51e83c9dfccac4ca62d214"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
