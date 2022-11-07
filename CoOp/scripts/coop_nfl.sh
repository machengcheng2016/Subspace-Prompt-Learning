cd ..

shots=$1
epoch=$2
for dataset in $3
do
  for seed in 1 2 3
  do
    python3 train.py \
    --root /path/to/data \
    --seed ${seed} \
    --trainer CoOp_nfl \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp/rn50_ep${epoch}.yaml \
    --output-dir output/${dataset}/CoOp/rn50_ep${epoch}_${shots}shots/nctx16_cscFalse_ctpend/seed${seed} \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${shots} \
    TRAIN.CHECKPOINT_FREQ 1 \
    CLASSNAMES_NFL classnames_nfl/imagenet_100.pth \
    TEXT_FEATURES_NFL text_features_nfl/imagenet_100.pth
  done
done
