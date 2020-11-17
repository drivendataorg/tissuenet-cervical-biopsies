#!/bin/bash


GPU=$1
CONFIG=$2
FOLD=$3

CUDA_VISIBLE_DEVICES=${GPU} python3 ./src/train.py --config "${CONFIG}" --fold "${FOLD}"
