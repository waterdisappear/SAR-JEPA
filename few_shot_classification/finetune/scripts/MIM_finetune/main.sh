#!/bin/bash

# custom config
DATA='./data/'
TRAINER=MIM_finetune

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

#{0..1}
for SEED in {0..4}

do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done