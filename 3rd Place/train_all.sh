#!/bin/bash


GPU=0

for CFG in ./configs/*.yaml; do
    for FOLD in {0..7}; do
        sh ./train.sh ${GPU} ${CFG} ${FOLD}
    done;
done;
