"""
    Implementation of TSN Residual Motion Generation for JHMDB Dataset Classification;

    1. Using the same sampling strategy for testing (Set test_mode = True), which is getting 14 fixed-index frames with center-cropping and normalization;
    2. Using 2 frames as warm-up, to generate the rest of 12 frames, in total 14 frames;
    3. Loading previous best performance weight_model;
    4. Compute the loss among them; Only optimize the Parameters of Generator;
    5. Try using generated feature for the classification;


    RBG channel frame Sampling Reference:
        /home/zhufl/videoPrediction/dataset_off.py/func:_sample_indices_off;

    TSN model Reference:
        https://github.com/yjxiong/tsn-pytorch/model;

    TSN Dataset Referecne:
        https://github.com/yjxiong/tsn-pytorch/dataset;

    RGB & Optical Flow Extraction:
        https://github.com/yjxiong/temporal-segment-networks;


    Update 2019.01.25:
        1. Pass batch_size to fea_gen_foward() in both TSN_BIT and BNInception module;
        2. Compute loss by re-forming list of tensor to stacked tensor;
        3. Find out that MSELoss of pytorch 0.4.1 is troublesome; So I implement my own referecne on https://discuss.pytorch.org/t/rmse-loss-function/16540/3;
        4. Upgrade to 1.0.0 solve the problem;

    Update 2019.01.26:
        1. Freeze all BN layer when train TemRes Generator;
        2. Remove all BN layer in Generator;

    Update 2019.01.27:
        1. Add back BN layer in Generator; Set Bias=False for Generator Residual-block;
        2. Add spatial gradient penalty to the Generator, reference: model_temp_res_gen_v2.py;
        3. To save training cost, spatial gradient is put onto dimenson-reduction feature diff;
        4. Could consider do spatial gradient onto high-dimension feature;

    Update 2019.02.20:
        1. Use Motion Vector as Generator to produce the next-step feature residual;
        2.
"""

import os, sys, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

''' Use class/package from tsn-pytorch'''
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tsn-pytorch'))
from model_tem_res_motion_vector import TSN
from transforms import *
from ops import ConsensusModule
from dataset_BIT import TSNDataSet

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from model_utils import topk_crossEntrophy

''' Config '''
arch = 'BNInception'
num_class = 101
modality = 'Flow'
crop_fusion_type= 'avg'
num_segments = 25
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 32
workers = 1
data_length = 10
warmup_t = 2
pred_t = data_length - warmup_t
slide_wind = 1

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
        self.model_name = 'TSN_Flow_2019-01-23_17-06-15.pth'

        # self._load_tsn_rgb_weight()
        self._load_pretrained_model(self.model_name)

    def _load_pretrained_model(self, model_name):
        """
            Load pretrained model that contains all weights for all layers;
            Allow missing parameters;
        """
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name))
        print("Number of parameters recovered from modeo {} is {}".format(model_name, len(checkpoint)))

        model_state = self.state_dict()
        base_dict = {k:v for k, v in checkpoint.items() if k in model_state}

        missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
        for key, value in missing_dict.items():
            print("Missing param {}".format(key))

        model_state.update(base_dict)
        self.load_state_dict(model_state)


    def _load_tsn_rgb_weight(self):
        """
            Loading Flow Weights and then fine-tune fc layers
        """

        flow_weights = '/home/zhufl/videoPrediction/JHMDB/hmdb51_rgb.pth'
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

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t, slide_wind):

        ''' generate feature
        @param: input: video input with shape [batch, num_seg, 3, 224, 224]
        @param: batch_size: batch
        @param: warmup_t: warm up time-step observation to compute residual for later prediction, >= 2;
        @param: pred-t: number of time-step to predict, >=1;

        @return:
            1. list of generated feature;
            2. list of origin feature;
            3. list of generated residual;
            4. list of origin residual
            5. list of generated diff grad;
            6. list of org diff grad;
        '''

        x, y, z, w, u, v = self.tsn.fea_gen_forward(input, batch_size, warmup_t, pred_t, slide_wind)
        return x, y, z, w, u, v

net = TSN_BIT()
net.tsn._init_kernelCNN(slide_wind)
net.cuda()

''' Define trainable and fixed param '''
for name, param in net.named_parameters():
    # if any(sub_name in name for sub_name in ['fc1', 'fc2', 'new_fc']):
    if any(sub_name in name for sub_name in ['kernelCNN']):
        param.requires_grad = True
        print("fine tuning para name {}, shape {}".format(name,param.data.shape))
    else:
        param.requires_grad = False

''' 2 BN Layer Freezing Strategies '''
net.train()
''''
    Freeze all parameters of BN layer
    This will make sure that original TSN feature doesn not change at all; Based on this, learning motion br anch is useful;
    In tensorflow code, need to pay attention this;
'''
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
param = filter(lambda p: p.requires_grad, net.parameters())

# Define optimizer
optimizer = torch.optim.Adam(param, lr = 0.00001, betas= (0.9, 0.99), weight_decay=0.0005)
criterion = nn.MSELoss(reduction='mean')

''' Load Dataset '''
''' data_length can control how many segments can we get from individual video '''
train_list = '../data/BIT_train.txt'
test_list = '../data/BIT_test.txt'
train_list = os.path.join(os.path.dirname(os.path.realpath(__file__)), train_list)
test_list = os.path.join(os.path.dirname(os.path.realpath(__file__)), test_list)
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
                test_mode=True,
                transform=torchvision.transforms.Compose([
                    GroupCenterCrop([224, 224]),
                    Stack(roll=arch == 'BNInception'),
                    ToTorchFormatTensor(div=arch != 'BNInception'),
                    normalize,
                ])
    ),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True, drop_last=True)
print("Length of dataset is {}".format(len(train_loader)))

''' Start Training Process '''
for epoch in range(100):
    for idx, (input, target, indice) in enumerate(train_loader):
        optimizer.zero_grad()

        input_var = input.view(batch_size * data_length, 10, 224, 224)
        input_var = torch.autograd.Variable(input_var, requires_grad=True).cuda()
        target = target.cuda()

        gen_fea, org_fea, gen_diff, org_diff, gen_diff_grad, org_diff_grad = net.fea_gen_forward(input_var, batch_size, warmup_t, pred_t, slide_wind)

        gen_fea = torch.stack(gen_fea).transpose_(0, 1)
        org_fea = torch.stack(org_fea).transpose_(0, 1)

        gen_diff = torch.stack(gen_diff).transpose_(0, 1)
        org_diff = torch.stack(org_diff).transpose_(0, 1)

        gen_diff_grad = torch.stack(gen_diff_grad).transpose_(0, 1)
        org_diff_grad = torch.stack(org_diff_grad).transpose_(0, 1)

        org_fea.detach()
        org_diff.detach()
        org_diff_grad.detach()

        loss1 = criterion(gen_fea, org_fea)
        loss2 = criterion(gen_diff, org_diff)
        loss3 = criterion(gen_diff_grad, org_diff_grad)

        loss = loss2 + 5 * loss3
        # loss = loss2
        loss.backward()

        # clip_grad_norm(param, max_norm=40)
        optimizer.step()
        print("For epoch {}, data batch {}, Fea MSE is {}, Diff MSE Loss is {} and Diff Sobel loss is {}".format(epoch, idx, loss1.item(), loss2.item(), loss3.item()))

''' save model to local '''
import datetime

''' Do not need to save weights for temporal-adaptive, since kernelCNN would generate weights '''
''' Or we can save all state_dict, when load, do not load TemporalAdaptiveCNN weights '''

state_name = 'TSN_TemResGen_MotionVector_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pth'
torch.save(net.state_dict(), state_name)
print("Saving model as {}".format(state_name))
