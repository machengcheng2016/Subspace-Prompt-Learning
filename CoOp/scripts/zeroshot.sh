cd ..

DATA=/path/to/data
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=rn50_ep50

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only
