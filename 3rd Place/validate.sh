#!/bin/bash

GPU=$1
WORK_DIR=$2

CUDA_VISIBLE_DEVICES=${GPU} python3 ./src/validate.py \
    --load ${WORK_DIR} \
