# Subspace-Prompt-Learning
Official code for "[Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models](https://arxiv.org/abs/2211.02219)".

[Chengcheng Ma](https://scholar.google.com/citations?user=-Zir-A8AAAAJ&hl=en)<sup>1,2</sup>, [Yang Liu](https://scholar.google.com/citations?user=t1emSE0AAAAJ&hl=zh-CN)<sup>3</sup>, [Jiankang Deng](https://scholar.google.com/citations?user=Z_UoQFsAAAAJ)<sup>4</sup>, [Lingxi Xie](https://scholar.google.com/citations?user=EEMm7hwAAAAJ&hl=zh-CN)<sup>4</sup>, [Weiming Dong](https://scholar.google.com/citations?user=WKGx4k8AAAAJ&hl=zh-CN)<sup>1</sup>, [Changsheng Xu](https://scholar.google.com.sg/citations?user=hI9NRDkAAAAJ&hl=zh-CN)<sup>1</sup>.

<sup>1</sup> Chinese Academy of Sciences Institute of Automation (CASIA)  
<sup>2</sup> University of the Chinese Academy of Sciences (UCAS)  
<sup>3</sup> Alibaba DAMO Academy  
<sup>4</sup> Huawei Technologies Co Ltd  

## TL;DR
We propose `Subspace Prompt Tuning (SubPT)` to mitigate the overfitting issue in the well-known prompt tuning method [CoOp](https://github.com/KaiyangZhou/CoOp), and further propose `Novel Feature Learner (NFL)` to enhance the generalization ability onto novel categories beyond the training set.

`SubPT` is illustrated as:
![SubPT](https://github.com/machengcheng2016/Subspace-Prompt-Learning/blob/main/teaser.png)

The full picture of our method:
![Overview](https://github.com/machengcheng2016/Subspace-Prompt-Learning/blob/main/overview.png)

## Preparation
This repo is based on the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), and we add the [model_subspace_backward_and_update](https://github.com/machengcheng2016/Subspace-Prompt-Learning/blob/main/Dassl.pytorch/dassl/engine/trainer.py#L311) function into `Dassl.pytorch/dassl/engine/trainer.py` to support subspace prompt tuning. 

Before you go, please go to the `./Dassl.pytorch` directory and make installation as follows.
```
# Create a conda environment
conda create -n subpt python=3.7

# Activate the environment
conda activate subpt

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library
cd Dassl.pytorch
python setup.py develop
```
Then go to the `./CoOp` directory and run `pip install -r requirements.txt` to install a few more packages required by CLIP.

Follow [DATASETS.md](https://github.com/machengcheng2016/Subspace-Prompt-Learning/blob/main/CoOp/DATASETS.md) to install the datasets.

## Usage
### Few-shot classification on 11 datasets
Please go to the `./CoOp` directory, and run "CoOp+SubPT" as follows.
```
############### Step 1. run CoOp ###############
# [SHOTS] and [EPOCH] are pairwise hyper-parameters in CoOp, specified as 
# (1 shot, 50 epoch)
# (2 shots, 100 epoch)
# (4 shots, 100 epoch) 
# (8 shots, 200 epoch) 
# (16 shots, 200 epoch)
# [EPOCH] is specially set as 50 for all shots for ImageNet.
cd scripts
bash coop.sh [SHOTS] [EPOCH] [DATASET]


############### Step 2. compute dominate eigenvectors representing the early-stage gradient flow ###############
# [FINISH] and [DIM] are the only two hyper-paramters in SubPT, corresponding to the $t_early$ and $r$ in our paper.
# [FINISH] and [DIM] are optionally specified as follows. [DIM] < [FINISH]. Other [DIM] values lead to similar results.
# (10, 5)  for 1 shot
# (20, 10) for 2 shot
# (30, 10) for 4 shot
# (40, 10) for 8 shot 
# (50, 10) for 16 shot
cd ..
python compute_eigenvector.py --ckpt_path [CKPT_PATH] --start 1 --finish [FINISH] --save_name \
  full_P/[DATASET]-CoOp-[SHOTS]shots-nctx16-seed1-start1-finish[FINISH]-dim[DIM].pth --n_components [DIM]


############### Step 3. re-run CoOp with SubPT ###############
# Note that [SHOTS] and [EPOCH] are in correspondence with Step 1, and [FINISH] and [DIM] are in correspondence with Step 2.
cd scripts
bash coop_sub.sh [SHOTS] [EPOCH] [FINISH] [DIM] [DATASET]
```
To run "CoOp+NFL" and "CoOp+SubPT+NFL", just replace `coop.sh` with `coop_nfl.sh`, and replace `coop_sub.sh` with `coop_sub_nfl.sh`. 
Before Step 1, please remember to pre-compute the text features with zero-shot CLIP and save them in the `./CoOp/text_features_nfl` directory. (Hint: run `zeroshot2.sh` and add `torch.save` at [here](https://github.com/machengcheng2016/Subspace-Prompt-Learning/blob/main/CoOp/trainers/zsclip.py#L97)).

We kindly write a `./output/quick_view_all_acc.py` script for you, in order to measure the classification accuracy.





### Base-to-Novel Generalization on 11 datasets
Please go to the `./CoOp` directory, and run "CoOp+SubPT" as follows.
```
############### Step 1. run CoOp ###############
# [SHOTS] and [EPOCH] are fixed as (4 shots, 100 epoch).
cd scripts
bash base2new_train_coop.sh [DATASET]


############### Step 2. compute dominate eigenvectors representing the early-stage gradient flow ###############
# [FINISH] and [DIM] are fixed as 30 and 10, respectively.
cd ..
python compute_eigenvector.py --ckpt_path [CKPT_PATH] --start 1 --finish 30 --save_name \
  full_P/b2n-[DATASET]-CoOp-4shots-nctx16-seed1-start1-finish30-dim10.pth --n_components 10


############### Step 3. re-run CoOp with SubPT ###############
# Note that [FINISH] and [DIM] are in correspondence with Step 2.
cd scripts
bash base2new_train_coop_sub.sh [DATASET]
```
After training, do evaluation as follows
```
cd scripts
# [SUB] is "base" or "new". [LOADEP] is 100, except 50 for ImageNet.
bash base2new_test_coop_sub.sh [SUB] [LOADEP] [DATASET]
```



### Domain Generalization on 4 datasets
Please go to the `./CoOp` directory, and run evaluation as follows.
```
cd scripts
# [TRAINER] can be CoOp, CoOp_sub, or CoOp_sub_nfl.
# [DATASET] can be imagenetv2, imagenet-sketch, imagenet-a, or imagenet-r.
bash eval.sh [TRAINER] [DATASET]
```

### Zero-Shot CLIP Baseline
Please go to the `./CoOp/script` directory and run `bash zeroshot.sh [DATASET]`.


## Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:
```
@ARTICLE{ma2022understanding,
  author={Ma, Chengcheng and Liu, Yang and Deng, Jiankang and Xie, Lingxi and Dong, Weiming and Xu, Changsheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models}, 
  year={2023},
  volume={33},
  number={9},
  pages={4616-4629},
  doi={10.1109/TCSVT.2023.3245584}}
```


## Acknowledgements
* This repo is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).
* Thanks to [DLDR](https://github.com/nblt/DLDR) repo (see reference [33] in our paper). It inspires me a lot.


## Contact
Feel free to contact me via machengcheng2016@gmail.com if you have any problems about our paper or codes.
