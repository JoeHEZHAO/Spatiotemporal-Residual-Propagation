"""
Implement RNN for residual motion propagtion, training with kalman filtering and jacobian matrix;

@Deprecated
Update 2019.01.31:
    1. Use TSN from model_temp_res_gen_rnn_kalman.py;
    2. Need to fix batch_size, so use drop_last flag in dataloader;

Update 2019.02.04:
    1. use model_tem_res_gen_v3;
    2. rnn for modeling kalman gain;
    3. Optimize kalman gain l2 norm, to make is less dependable on ground truth data; ===> In the end, all kalman gain equals to 0.5 evenly, which is not my goal;
    4. Put optim onto fea_diff and gen_diff;
    5. Instead of compute loss from deep feature, cmopute loss from feature diff;[Change return value in model_tem_res_gen_v3.py:fea_gen_forward]
    6. l2 norm on kalman gain is not very efficient; Try use l1 norm instead;
    7. Rethink about the usage of kalman gain; If do not optim on K, K ==> 1, fully depend on gt; If optim on K, K ==> 0, fully depend on generated feature;
    8. K could be influenced by sigmoid or relu:
        1. Remove relu and sigmoid and retry; ===> loss become Nan;
        2. Remove relu, keep sigmoid use l1 norm on kalman_gain ===> Start to learnable, lets see how small it can be (why ??);
        3. -relu, +sigmoid, l2 ===> ?;
        4. weights norm

Update 2019.02.05:
    1. Modify the kalman_gain loss, as a ranking loss, which encourge later time-step to utilize more origin feature;
    2. l2 norm for time t, belonging to [0, T],
       l2_norm = t / T * l2;
    3. Reference from https://basurafernando.github.io/papers/ICCV17.pdf;

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
sys.path.append('/home/zhufl/Temporal-Residual-Motion-Generation/tsn-pytorch')
from model_tem_res_gen_v3 import TSN
from transforms import *
from ops import ConsensusModule
from dataset_JHMDB import TSNDataSet
sys.path.append('../')
from model_utils import topk_crossEntrophy

''' Config '''
arch = 'BNInception'
num_class = 51
modality = 'RGB'
crop_fusion_type= 'avg'
num_segments = 25
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 32
workers = 1
data_length = 1
warmup_t = 2
pred_t = num_segments - warmup_t
train = True
kalman_update = True

class TSN_BIT(nn.Module):

    def __init__(self):
        super(TSN_BIT, self).__init__()
        self.tsn = TSN(num_class, num_segments=num_segments, modality=modality,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7)

        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(51, 32)
        self.fc2 = nn.Linear(32, 21)
        self.model_name = 'TSN_RGB_2019-01-24_12-26-11.pth'
        self._load_pretrained_model(self.model_name)

    def _load_pretrained_model(self, model_name):

        """
            Load pretrained model that contains all weights for all layers;
            Allow missing parameters;
        """

        checkpoint = torch.load('/home/zhufl/videoPrediction/JHMDB/' + model_name)
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

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t):
        """
            return:
                1. fea_gen;
                2. org_fea;
                3. diff_gen;
                4. org_diff;
                5. diff_gen_grad;
                6. org_diff_grad;
                7. gen_fea_grad;
                8. org_fea_grad;
                9. kalman_gain;
        """
        x, y, z, w, m, n, q, p, u = self.tsn.fea_gen_forward(input, batch_size, warmup_t, pred_t)
        return x, y, z, w, m, n, q, p, u

net = TSN_BIT().cuda()

''' Define trainable and fixed param '''
for name, param in net.named_parameters():
    if any(sub_name in name for sub_name in ['res1', 'res2', 'rnn']):
        param.requires_grad = True
        print("fine tuning para name {}, shape {}".format(name,param.data.shape))
    else:
        param.requires_grad = False

''' 2 BN Layer Freezing Strategies '''
net.train()
net.tsn.train = train
net.tsn.kalman_update = kalman_update

''''
    Freeze all parameters of BN layer
    This will make sure that original TSN feature doesn not change at all; Based on this, learning motion branch is useful;
    In tensorflow code, need to pay attention this;
'''
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
param = filter(lambda p: p.requires_grad, net.parameters())

# Define optimizer
optimizer = torch.optim.Adam(param, lr = 0.001, betas= (0.9, 0.99), weight_decay=0.0005)
criterion = nn.MSELoss(reduction='mean')

''' Load Dataset '''
''' data_length can control how many segments can we get from individual video '''
train_list = '/home/zhufl/Data2/jhmdb_frame/new_train.txt'
test_list = '/home/zhufl/Data2/jhmdb_frame/new_test.txt'
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
for epoch in range(15):
    for idx, (input, target, indice) in enumerate(train_loader):
        optimizer.zero_grad()

        input_var = input.view(batch_size * num_segments, 3, 224, 224)
        input_var = torch.autograd.Variable(input_var, requires_grad=True).cuda()
        target = target.cuda()

        # gen_fea, org_fea, gen_fea_grad, org_fea_grad, kalman_gain_list = net.fea_gen_forward(input_var, batch_size, warmup_t, pred_t)
        gen_fea, org_fea, gen_fea_diff, org_fea_diff, gen_fea_diff_grad, org_fea_diff_grad, gen_fea_grad, org_fea_grad, kalman_gain_list = net.fea_gen_forward(input_var, batch_size, warmup_t, pred_t)

        gen_fea = torch.stack(gen_fea).transpose_(0, 1)
        org_fea = torch.stack(org_fea).transpose_(0, 1)

        gen_fea_diff = torch.stack(gen_fea_diff).transpose_(0, 1)
        org_fea_diff = torch.stack(org_fea_diff).transpose_(0, 1)

        gen_fea_grad = torch.stack(gen_fea_grad).transpose_(0, 1)
        org_fea_grad = torch.stack(org_fea_grad).transpose_(0, 1)

        gen_fea_diff_grad = torch.stack(gen_fea_diff_grad).transpose_(0, 1)
        org_fea_diff_grad = torch.stack(org_fea_diff_grad).transpose_(0, 1)

        org_fea.detach()
        org_fea_diff.detach()
        org_fea_grad.detach()

        '''
            Though not training of mse of feature and feature grad, Printing out results help us monitoring training process;
        '''
        loss1 = criterion(gen_fea, org_fea)
        loss2 = criterion(gen_fea_diff, org_fea_diff)
        loss4 = criterion(gen_fea_grad, org_fea_grad)
        loss5 = criterion(gen_fea_diff_grad, org_fea_diff_grad)
        # loss3 = sum([x.norm(1) for idx, x in enumerate(kalman_gain_list) if idx < 1 ])
        loss3 = sum([x.norm(2) for idx, x in enumerate(kalman_gain_list)])

        loss = loss2 + loss5
        # loss = 0.01 * loss3
        loss.backward()
        optimizer.step()

        print(len(kalman_gain_list))
        # print("For epoch {}, data batch {}, MSE Loss is {} and Feature Diff loss is {}, norm of P/K is {}, value of Kalman Gain {}".format(epoch, idx, loss1.item(), loss2.item(), loss3.item(), kalman_gain_list[0][0]))
        # print("For epoch {}, data batch {}, MSE Loss is {} and Spatial Gradient loss is {}, norm of P/K is {}".format(epoch, idx, loss1.item(), loss2.item(), loss3.item()))
        print("For epoch {}, data batch {}: MSE Loss is {}, Feature Diff loss is {}, Feature Gradient is {}, norm K is {}, 1st Kalman Gain {}, 13st Kalman Gain {}"\
        .format(epoch, idx, loss1.item(), loss2.item(), loss4.item(), loss3.item(), kalman_gain_list[0][0].item(), kalman_gain_list[0][-1].item()))

''' save model to local '''
import datetime
state_name = 'TSN_TemResGen_Kalman_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pth'
torch.save(net.state_dict(), state_name)
print("Saving model as {}".format(state_name))
