"""
    Implementation of TSN-Flow branch for BIT-Interaction Classification;

    1. For each video, sparsely sample 3 segmenets, each segments contains 5 optical-flow frames, 10 in total for x & y directions ==> Input Shape as [batch * num_segments, H, W, 10];
    2. Re-write sample indice, only do segmenting and sampling from [20%, 80%] frames, to erase-out the confusion of starting and ending portion of actions;
    3. Load Pre-trained FLOW weights, rewrite the final fc layer to 8 classes;
    4. Using Consensus Method for 3 different segments;

    TSN model Reference:
        https://github.com/yjxiong/tsn-pytorch/model;

    TSN Dataset Referecne:
        https://github.com/yjxiong/tsn-pytorch/dataset;

    RGB & Optical Flow Extraction:
        https://github.com/yjxiong/temporal-segment-networks;

    DataLoader Augmentation for RGB, Optical-Flow and RGBDiff:
        /home/zhufl/Workspace/tsn-pytorch/main.py & test_model.py

    Notice:
        1. For Dataset, num_segments ==> number of frames per clips; new_length ===> number of clips;
        2. For TSN model, num_segments  ===> number of clips for optical-flow (each clips contain 5 frames individuallu for x & y); Or number of Images for RGB;

    Update:
        1. For testing first p% clip, assuming total clip is 10:
            0. Config p param in Config Session;
            1. Change num_segments for TSN model @Line63;
            2. Change input_var indexing @Line140;
        2. For
               p=0.1 ==> 28.12%;
               p=0.2 ==> 31.25%;
               p=0.3 ==> 42.19%;
               p=0.4 ==> 66.41%;
               p=0.5 ==> 82.03%;
               p=0.6 ==> 89.06%;
               p=0.7 ==> 91.04%;
               p=0.8 ==> 91.41%;
               p=0.9 ==> 92.97%;
               p=1.0 ==> 92.19%;
"""

import os, sys, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

cur_path = os.path.dirname(os.path.realpath(__file__))

''' Use class/package from tsn-pytorch'''
sys.path.append(os.path.join(cur_path, '../../tsn-pytorch'))
from models import TSN
from transforms import *
from ops import ConsensusModule
from dataset_BIT import TSNDataSet

''' Config '''
arch = 'BNInception'
num_class = 101
modality = 'Flow'
crop_fusion_type= 'avg'
num_segments = 5
flow_prefix = 'flow_'
batch_size = 32
workers = 1
data_length = 11
p = 10

class TSN_BIT(nn.Module):

    def __init__(self):
        super(TSN_BIT, self).__init__()
        self.tsn = TSN(num_class, num_segments=p, modality=modality,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7)

        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(101, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, input):

        x = self.activation(self.tsn(input))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

net = TSN_BIT().cuda().eval()

''' Load Dataset '''
model_name = 'TSN_Flow_2019-01-23_17-06-15.pth'
checkpoint = torch.load(os.path.join(cur_path, model_name))
print("Number of parameters recovered from modeo {} is {}".format(model_name, len(checkpoint)))

model_state = net.state_dict()
base_dict = {k:v for k, v in checkpoint.items() if k in model_state}

missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
for key, value in missing_dict.items():
    print("Missing motion branch param {}".format(key))

model_state.update(base_dict)
net.load_state_dict(model_state)

''' data_length can control how many segments can we get from individual video '''
train_list = os.path.join(cur_path, '../data/BIT_train.txt')
test_list = os.path.join(cur_path, '../data/BIT_test.txt')

""" Apply normalize onto input data """
input_mean = net.tsn.input_mean
input_std = net.tsn.input_std
scale_size = net.tsn.scale_size

if modality != 'RGBDiff':
    normalize = GroupNormalize(input_mean, input_std)
else:
    normalize = IdentityTransform()

train_loader = torch.utils.data.DataLoader(
    TSNDataSet("", test_list, num_segments=num_segments,
                new_length=data_length,
                modality=modality,
                image_tmpl="img_{:05d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix+"{}_{:05d}.jpg",
                test_mode=True,
                transform=torchvision.transforms.Compose([
                    GroupScale(int(scale_size)),
                    GroupCenterCrop([224, 224]),
                    Stack(roll=arch == 'BNInception'),
                    ToTorchFormatTensor(div=arch != 'BNInception'),
                    normalize,
                ])
    ),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True,
    drop_last=False)
print("Length of dataset is {}".format(len(train_loader)))

''' Start Testing Process '''
accur = []
gt = []
for epoch in range(1):
    for idx, (input, target, indice) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        # print(indice)

        b_shape = input.shape[0]

        ''' Only use fisrt p% for accuracy '''
        input = input.contiguous().view(b_shape, -1, 10, 224, 224)[:, :p,:, :, :]
        # input = input.contiguous().view(b_shape, -1, 10, 224, 224)[:, 1:,:, :, :]
        input_var = input.contiguous().view(b_shape * p, 10, 224, 224)
        ''' use all 10 clips '''
        # input_var = input.view(b_shape * data_length, 10, 224, 224)

        input_var = torch.autograd.Variable(input_var, volatile=True).cuda()
        target = target.detach()

        out = net(input_var).detach()
        out = out.data.cpu().numpy().copy()
        pred = np.argmax(out, 1)

        # import pdb;pdb.set_trace()
        accur += pred.tolist()
        gt += target.numpy().tolist()

        print("For epoch {}, batch {}".format(epoch, idx))

# import pdb;pdb.set_trace()
'''  count the over all accuracy & confusion matrix '''
cf = confusion_matrix(gt, accur).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
print 'cls_hit:'
print cls_hit
print 'cls_cnt:'
print cls_cnt
cls_acc = cls_hit / cls_cnt
print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
