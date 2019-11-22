"""
    Implementation of TSN-RGB branch for JHMDB Classification;

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

    Update:
        1. Modify dataset_JHMDB to sample fixed frame index when testing;
        2. Adding confusion matrix ploting func;

    Update 2019.01.25:
        1. Using only 2 frames out of 14; To compare with feature generation;
        2. Adding num_segments_test parameter, for controlling how many frames to be used after sampling;

    Update 2019.01.26:
        1. Dynamic adapt percentage p, for testing accuracy under different frame number;
            p=1 ==> 74.31%;
            p=2 ==> 76.23%;
            p=3 ==> 75.37%;
            p=4 ==> 76.80%;
            p=5 ==> 77.53%;
            p=6 ==> 77.92%;
            p=7 ==> 78.65%;
            p=8 ==> 77.76%;
            p=9 ==> 79.46%;
            p=10 ==> 81.81%;
            p=11 ==> 81.81%;
            p=12 ==> 83.04%;
            p=13 ==> 83.36%;
            p=14 ==> 83.39%;
"""

import os, sys, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from collections import OrderedDict

''' Use class/package from tsn-pytorch'''
cur_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(cur_path, '../../tsn-pytorch'))
from models import TSN
from transforms import *
from ops import ConsensusModule
from dataset_JHMDB import TSNDataSet

sys.path.append(os.path.join(cur_path, '../'))
from confusion_matrix_utils import *

''' Config '''
arch = 'BNInception'
num_class = 51
modality = 'RGB'
crop_fusion_type= 'avg'
num_segments = 14
flow_prefix = 'flow_'
batch_size = 32
workers = 1
data_length = 1
# actions = os.listdir('/home/zhufl/Data2/JHMDB')
# actions.remove('train_test_splits')
# print(actions)

p = 14
print("Using {} out of {} for testing".format(p, num_segments))

class TSN_BIT(nn.Module):

    def __init__(self):
        super(TSN_BIT, self).__init__()
        self.tsn = TSN(num_class, num_segments=p, modality=modality,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7)

        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(51, 32)
        self.fc2 = nn.Linear(32, 21)

    def forward(self, input):

        x = self.activation(self.tsn(input))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

net = TSN_BIT().cuda()
net.eval()

''' Load Dataset '''
model_name = 'TSN_RGB_2019-01-24_12-26-11.pth'
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
train_list = os.path.join(cur_path, '../data/JHMDB_train.txt')
test_list = os.path.join(cur_path, '../data/JHMDB_test.txt')

""" Apply normalize onto input data """
input_mean = net.tsn.input_mean
input_std = net.tsn.input_std
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

# net = nn.DataParallel(net, device_ids=[1,2,3]).cuda(1)

''' Start Testing Process '''
accur = []
gt = []
for epoch in range(1):
    for idx, (input, target, indice) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        # print(indice)

        with torch.no_grad():
            b_shape = input.shape[0]

            '''
                Selecting front 2 frames;
                Commen out if use all sampled frames;
            '''
            if p is not None:
                input = input.view(b_shape, num_segments, 3, 224, 224)
                input_var = input[:, :p, :, :, :]

            input_var = input_var.contiguous().view(-1, 3, 224, 224)
            input_var = torch.autograd.Variable(input_var, volatile=True).cuda()
            target = target.detach()

            out = net(input_var).detach()
            out = out.data.cpu().numpy().copy()
            pred = np.argmax(out, 1)

            accur += pred.tolist()
            gt += target.numpy().tolist()

            print("For epoch {}, batch {}".format(epoch, idx))

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

''' draw the confusion matrix '''
# draw_cnf(gt, accur, actions)
