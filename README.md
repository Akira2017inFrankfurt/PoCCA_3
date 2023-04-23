# Intorduction
This is the official code implementation for the paper "Local-to-Global Cross Attention-based Contrastive Learning on 3D Point Clouds".

## Dataset
#### Pretrain: [ShapeNet](https://drive.google.com/uc?id=1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB)
#### Classfication: [ModelNet40](https://drive.google.com/uc?id=15lmtRaHvVIPLOp_o7rms8e6VQp_en8WF), [ScanObjectNN](https://drive.google.com/uc?id=1r3nJ6gEu6cL59h7cIfMrBeY5p1P_JiIV)
#### Part Segmentation: [ShapeNetPart](https://drive.google.com/uc?id=1-Yhqgi7KH6guIej_4X8pciiybQB4pBWQ)

```Python
cd PoCCA_3
gdown https://drive.google.com/uc?id=1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB
gdown https://drive.google.com/uc?id=15lmtRaHvVIPLOp_o7rms8e6VQp_en8WF
gdown https://drive.google.com/uc?id=1r3nJ6gEu6cL59h7cIfMrBeY5p1P_JiIV
gdown https://drive.google.com/uc?id=1-Yhqgi7KH6guIej_4X8pciiybQB4pBWQ
```

## Installation
#### Pre-train and Test Enviorment:
- CentOS Linux release 7.9.2009
- Python 3.10
- PyTorch 2.0
- CUDA 11.7
- Torchvision 0.15.1
- Timm 0.6.13
- Einops 0.6.1

## Pre-trained Models
PoCCA pre-trained models with DGCNN feature extractor are available here[default].

## Train PoCCA
- For single GPU: run .ipynb scripts file in Pocca/train/
- For multiple GPUs: run train.py

The pre-trained model weights will be saved in weights/.

## SVM Classfication
