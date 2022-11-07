shots=$1
type=$2
epoch=$3
finish=$4
dim=$5
for dataset in $6
do
  for seed in 1 2 3
  do
    python3 train.py \
    --root /path/to/data \
    --seed ${seed} \
    --trainer CoOp_sub_nfl \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp/rn50_ep${epoch}.yaml \
    --output-dir output/${dataset}/CoOp_sub_nfl/rn50_ep${epoch}_${shots}shots/nctx16_cscFalse_ctpend/seed${seed} \
    DIRECT_RESUME output/${dataset}/CoOp_nfl/rn50_ep${epoch}_${shots}shots/nctx16_cscFalse_ctpend/seed${seed}/prompt_learner/model.pth.tar-1 \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${shots} \
    TRAIN.CHECKPOINT_FREQ 0 \
    TRAINER.U full_U/${dataset}-CoOp_nfl-${shots}shots-nctx16-seed${seed}-start1-finish${finish}-dim${dim}.pth \
    CLASSNAMES_AUG classnames_nfl/imagenet_100.pth \
    TEXT_FEATURES_AUG text_features_nfl/imagenet_100.pth
  done
done
