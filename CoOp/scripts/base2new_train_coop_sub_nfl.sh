cd ..

for dataset in $1
do
  for seed in 1 2 3
  do
    python3 train.py \
    --root /path/to/data \
    --seed ${seed} \
    --trainer CoOp_sub_nfl \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp_sub_nfl/rn50_ep100.yaml \
    --output-dir output/base2new/train_base/${dataset}/shots_4/CoOp_sub_nfl/rn50_ep100/seed${seed} \
    DIRECT_RESUME output/base2new/train_base/${dataset}/shots_4/CoOp_nfl/rn50_ep100/seed${seed}/prompt_learner/model.pth.tar-1 \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 4 \
    TRAIN.CHECKPOINT_FREQ 0 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.U full_P/b2n-${dataset}-CoOp_nfl-4shots-nctx16-seed${seed}-start1-finish30-dim10.pth \
    CLASSNAMES_NFL classnames_nfl/${dataset}.pth \
    TEXT_FEATURES_NFL text_features_nfl/${dataset}.pth
  done
done
