#!/bin/bash


GPU=0

for WORK_DIR in effnet-b0 effnet-b0_p4_ts128_t144 effnet-b0_p4_ts192_t64; do
    for FOLD in {0..7}; do
        sh ./validate.sh "${GPU}$" "${WORK_DIR}"/"${FOLD}"/best.pth
    done
done
