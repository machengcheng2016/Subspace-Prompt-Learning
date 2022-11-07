cd ..

sub=$1
loadep=$2
for dataset in $3
do
  for seed in 1 2 3
  do
    python3 train.py \
    --root /path/to/data \
    --seed ${seed} \
    --trainer CoOp_nfl \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp/rn50_ep100.yaml \
    --output-dir output/base2new/test_${sub}/${dataset}/shots_4/CoOp/rn50_ep100/seed${seed}/${loadep} \
    --model-dir output/base2new/train_base/${dataset}/shots_4/CoOp/rn50_ep100/seed${seed} \
    --load-epoch ${loadep} \
    --eval-only \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 4 \
    DATASET.SUBSAMPLE_CLASSES ${sub} \
    CLASSNAMES_NFL classnames_nfl/${dataset}.pth \
    TEXT_FEATURES_NFL text_features_nfl/${dataset}.pth
  done
done
