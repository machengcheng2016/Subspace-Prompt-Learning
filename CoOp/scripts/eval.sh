#!/bin/bash

cd ..

# custom config
DATA=/path/to/data
TRAINER=$1
SHOTS=4
NCTX=16
CSC=False
CTP=end

DATASET=$2
CFG=rn50_ep50


for SEED in 1 2 3
do
    DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer CoOp \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
        --load-epoch 50 \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
    fi
done
