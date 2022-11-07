# Subspace-Prompt-Learning
Official code for "Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models"

## TL;DR
We propose `Subspace Prompt Tuning (SubPT)` to mitigate the overfitting issue in the well-known prompt tuning method [CoOp](https://github.com/KaiyangZhou/CoOp), and further propose `Novel Feature Learner (NFL)` to enhance the generalization ability onto novel categories beyond the training set.

## Preparation
This code is based on the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
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

# Install this library (no need to re-build if the source code is modified)
cd Dassl.pytorch
python setup.py develop
```

## Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:
```
@misc{ma2022understanding,
  title={Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models}, 
  author={Chengcheng Ma and Yang Liu and Jiankang Deng and LingXi Xie and Weiming Dong and Changsheng Xu},
  year={2022},
  eprint={2211.02219},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
