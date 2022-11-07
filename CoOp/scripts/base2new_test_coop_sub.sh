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
    --trainer CoOp_sub \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file configs/trainers/CoOp_sub/rn50_ep100.yaml \
    --output-dir output/base2new/test_${sub}/${dataset}/shots_4/CoOp_sub/rn50_ep100/seed${seed}/${loadep} \
    --model-dir output/base2new/train_base/${dataset}/shots_4/CoOp_sub/rn50_ep100/seed${seed} \
    --load-epoch ${loadep} \
    --eval-only \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 4 \
    DATASET.SUBSAMPLE_CLASSES ${sub} \
    TRAINER.U full_P/b2n-${dataset}-CoOp-4shots-nctx16-seed${seed}-start1-finish30-dim10.pth
  done
done
