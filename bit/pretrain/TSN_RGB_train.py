"""
    Implementation of TSN-RGB branch for BIT-Interaction Classification; 
    
    1. For each video, assume testing on 25 frames, then make sure that training and tesing has the same efficiency; 
    2. Re-write sample indice, only do segmenting and sampling from [20%, 80%] frames, to erase out the starting and ending portion of actions;
    3. Load Pre-trained RGB weights, rewrite the final fc layer to 8 classes;
    4. Using Consensus Method for 3 different segments;
    5. For RGB, tsn.num_segments should be num_segments; While for Flow, tsn.num_segments shoud be new_length;
    6. 

    RBG channel frame Sampling Reference:
        /home/zhufl/videoPrediction/dataset_off.py/func:_sample_indices_off;

    TSN model Reference:
        https://github.com/yjxiong/tsn-pytorch/model;

    TSN Dataset Referecne:
        https://github.com/yjxiong/tsn-pytorch/dataset;

    RGB & Optical Flow Extraction:
        https://github.com/yjxiong/temporal-segment-networks;
"""

import os, sys, cv2
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

''' Use class/package from tsn-pytorch'''
sys.path.append('/home/zhufl/Workspace/tsn-pytorch')
from models import TSN
from transforms import *
from ops import ConsensusModule
from dataset_BIT import TSNDataSet

sys.path.append('../')
from model_utils import topk_crossEntrophy

''' Config '''
arch = 'BNInception'
num_class = 101
modality = 'RGB'
crop_fusion_type= 'avg'
num_segments = 7
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 32
workers = 1
data_length = 1

class TSN_BIT(nn.Module):

    def __init__(self):
        super(TSN_BIT, self).__init__()
        self.tsn = TSN(num_class, num_segments=num_segments, modality=modality,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7)

        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(101, 32)
        self.fc2 = nn.Linear(32, 8)
        self.model_name = '2019-01-20_23-57-32.pth'

        self._load_tsn_rgb_weight()
        # self._load_pretrained_model(self.model_name)

    def _load_pretrained_model(self, model_name):

        """
            Load pretrained model that contains all weights for all layers
        """

        checkpoint = torch.load('/home/zhufl/videoPrediction/BIT_train_test/' + model_name)
        print("Number of parameters recovered from modeo {} is {}".format(model_name, len(checkpoint)))

        model_state = self.state_dict()
        base_dict = {k:v for k, v in checkpoint.items() if k in model_state}

        missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
        for key, value in missing_dict.items():
            print("Missing motion branch param {}".format(key))

        model_state.update(base_dict)
        self.load_state_dict(model_state)


    def _load_tsn_rgb_weight(self):
        """
            Loading Flow Weights and then fine-tune fc layers
        """

        flow_weights = '/home/zhufl/Workspace/tsn-pytorch/ucf101_rgb.pth'
        checkpoint = torch.load(flow_weights)

        base_dict = {}
        count = 0
        for k, v in checkpoint.items():

            count = count + 1
            print count, k
            if 415>count>18:
                base_dict.setdefault(k[7:], checkpoint[k])
            if count<19:
                base_dict.setdefault(k, checkpoint[k])
                base_dict.setdefault('new_fc.weight', checkpoint['base_model.fc-action.1.weight'])
                base_dict.setdefault('new_fc.bias', checkpoint['base_model.fc-action.1.bias'])

        self.tsn.load_state_dict(base_dict)

    def forward(self, input):

        x = self.activation(self.tsn(input))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

net = TSN_BIT().cuda()

''' Define trainable and fixed param '''
for name, param in net.named_parameters():
    if any(sub_name in name for sub_name in ['fc1', 'fc2', 'new_fc']):
        param.requires_grad = True
        print("fine tuning para name {}, shape {}".format(name,param.data.shape))
    else:
        param.requires_grad = False

''' 2 BN Layer Freezing Strategies '''
net.train()
# for m in net.tsn.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         m.eval()
#         m.weight.requires_grad = False
#         m.bias.requires_grad = False

param = filter(lambda p: p.requires_grad, net.parameters())

# Define optimizer
optimizer = torch.optim.Adam(param, lr = 0.0001, betas= (0.9, 0.99), weight_decay=0.0005)
# criterion = topk_crossEntrophy(top_k=0.3)
criterion = nn.CrossEntropyLoss()

''' Load Dataset '''
''' data_length can control how many segments can we get from individual video ''' 
train_list = '/home/zhufl/Data2/BIT_train_test_split/new_train.txt'
test_list = '/home/zhufl/Data2/BIT_train_test_split/new_test.txt'
train_augmentation = net.tsn.get_augmentation()

input_mean = net.tsn.input_mean
input_std = net.tsn.input_std
if modality != 'RGBDiff':
    normalize = GroupNormalize(input_mean, input_std)
else:
    normalize = IdentityTransform()

train_loader = torch.utils.data.DataLoader(
    TSNDataSet("", train_list, num_segments=num_segments,
                new_length=data_length,
                modality=modality,
                image_tmpl="img_{:05d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix+"{}_{:05d}.jpg",
                transform=torchvision.transforms.Compose([
                    train_augmentation,
                    Stack(roll=arch == 'BNInception'),
                    ToTorchFormatTensor(div=arch != 'BNInception'),
                    normalize,
                ])
    ),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True, drop_last=True)
print("Length of dataset is {}".format(len(train_loader)))

''' Start Training Process '''
for epoch in range(400):
    for idx, (input, target, indice) in enumerate(train_loader):
        optimizer.zero_grad()

        input_var = input.view(batch_size * num_segments, 3, 224, 224)
        input_var = torch.autograd.Variable(input_var, requires_grad=True).cuda()
        target = target.cuda()

        out = net(input_var)

        loss = criterion(out, target)
        loss.backward()
        # clip_grad_norm(param, max_norm=20)

        optimizer.step()
        print("For epoch {}, data batch {}, Cross Entrophy Loss is {}".format(epoch, idx, loss.item()))

''' save model to local '''
import datetime
state_name = 'TSN_RGB_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pth'
torch.save(net.state_dict(), state_name)
print("Saving model as {}".format(state_name))
