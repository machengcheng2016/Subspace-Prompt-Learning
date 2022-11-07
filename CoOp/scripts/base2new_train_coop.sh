cd ..

for dataset in $1
do
  for seed in 1 2 3
  do
    python3 train.py \
    --root /path/to/data \
    --seed ${seed} \
    --trainer CoOp \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp/rn50_ep100.yaml \
    --output-dir output/base2new/train_base/${dataset}/shots_4/CoOp/rn50_ep100/seed${seed} \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 4 \
    TRAIN.CHECKPOINT_FREQ 1 \
    DATASET.SUBSAMPLE_CLASSES base
  done
done
