# Spatiotemporal-Residual-Propagation
Code release for ICCV 2019 paper "Spatiotemporal Feature Residual Propagation for Action Prediction;" 

# Environment Requirements:
1. Python 2.7
2. Pytorch 0.4.0
3. CUDA-8
4. ffmpeg
5. PIL

# Dataset Preparation
## Download
- JHMDB: http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets
- BIT: https://sites.google.com/site/alexkongy/software
- UCF101: https://www.crcv.ucf.edu/data/UCF101.php

For dataset loading and augmenting, we modify code from tsn (https://github.com/yjxiong/tsn-pytorch/blob/master/dataset.py) for every dataset and the modified version are locating in each sub-folder.


# Pretraining Process:
Since proposed method focus on feature propagation, we obtain classifiers offline by pretraining each dataset with TSN model.
- UCF-101: Directly from tsn pretrain;
- JHMDB-21: Pretrain on RGB;
- BIT: Pretrain on Flow;

See pretrain folder in each subfolder for more details; Pretrain Model Link:
- JHMDB-21 Split 1: https://drive.google.com/open?id=1dc8Ec6FAg5N-gHfW65f_mXpP4mlDzlpv;
- BIT: https://drive.google.com/open?id=14uZRSebu3pNraRjZfnmKl41jJeTDaSN://drive.google.com/open?id=14uZRSebu3pNraRjZfnmKl41jJeTDaSNM; 
- UCF-101 Split 1 flow: https://drive.google.com/open?id=1o8A8-4OKVlJL-QmOmSfppcUTyWxrMX1Q;
- UCF-101 Split 1 rgb: https://drive.google.com/open?id=1gXUrIBaf98OrVKHUy-dn2khC2EH3zK8X;

# Training and Comparison
Please see details in each subfolder for training and testing procedures
