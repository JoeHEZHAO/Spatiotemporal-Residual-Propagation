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
        1. Create file for testing
        2. Coding func gen_fea_cls() for TSN_BIT;
        3. Coding func gen_fea_cls() for TSN of model_temp_res_gen.py (/home/zhufl/Workspace/tsn-pytorch);
        4. Coding func gen_fea_cls() for BNInception of BNIncepton_model.py (/home/zhufl/Workspace/tsn-pytorch);
        5. func gen_fea_cls() of model_temp_res_gen.py is handling Consensus and Modality;

    Update 2019.01.25:
    For TSN_TemResGen_2019-01-25_13-18-10.pth
        1. Testing with org_fea, result is 83.35%; Means only Generator got trained and other-part stay the same;
        2. Testing with gen_fea, result is 57.46%;
        3. Reasons could be:
            1. Should not use train_augmentation for data augmentation; Only use Center Cropping;
            2. Training is not converged enough;

    Update 2019.01.28:
        1. Equip Multi-GPUs module with network;
        2. Important for DataParallel:
            1. Wrap the model with DataParallel;
            2. Wrap all func in one func;
            3. Be careful when change the data shape:
                1. Before put input into forward() func, batch/dim=0 is not divided;
                2. In forward() func, batch is divided by the number of GPU utiliz;
            4. DataParallel works for all submodule of network, but just remember to include whole training/testing
            process in one forward() func;
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

cur_path = os.path.dirname(os.path.realpath(__file__))

''' Config '''
arch = 'BNInception'
num_class = 101
modality = 'Flow'
crop_fusion_type= 'avg'
num_segments = 5
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 32
workers = 1
data_length = 10
warmup_t = 3
pred_t = data_length - warmup_t
slide_wind = 1

class TSN_BIT(nn.Module):

    def __init__(self):
        super(TSN_BIT, self).__init__()
        self.tsn = TSN(num_class, num_segments=num_segments, modality=modality,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7, batch=batch_size)

        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(101, 32)
        self.fc2 = nn.Linear(32, 8)
        self.model_name = 'TSN_TemResGen_MotionVector_2019-02-23_22-21-02.pth'

        # self._load_tsn_rgb_weight()
        self._load_pretrained_model(self.model_name)

    def _load_pretrained_model(self, model_name):

        """
            Load pretrained model that contains all weights for all layers;
            Allow missing parameters;
        """

        checkpoint = torch.load(os.path.join(cur_path, model_name))
        print("Number of parameters recovered from modeo {} is {}".format(model_name, len(checkpoint)))

        model_state = self.state_dict()
        base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
        # base_dict = {k:v for k, v in base_dict.items() if 'TemporalAdaptiveCNN' not in k}

        missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
        for key, value in missing_dict.items():
            print("Missing motion branch param {}".format(key))

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

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t, slide_wind):
        '''
            return: 
                warm_diff, fea_diff, gen_dif_grad, fea_dif_grad, gen_fea, org_fea
        '''
        x, y, _, _, _, _ = self.tsn.fea_gen_forward(input, batch_size, warmup_t, pred_t, slide_wind)
        return x, y

    def gen_fea_cls(self, input):
        """
            Wrapper func for model_temp_res_gen.TSN.gen_fea_cls;

            Input: Intermediate feature [batch * num_segments/data_length, 192, 28, 28];
            Call: gen_fea_cls of model_temp_res_gen.TSN, to output [batch, 101/51], depends which weight is being used;
            Ouput: [batch, num_class]
        """
        x = self.tsn.gen_fea_cls(input)
        x = self.activation(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input, warmup_t, pred_t, slide_wind):

        b_shape = int(input.shape[0] / data_length)
        gen_fea, org_fea = self.fea_gen_forward(input, b_shape, warmup_t, pred_t, slide_wind)
        gen_fea = torch.stack(gen_fea).transpose_(0, 1).contiguous().view(-1, 192, 28, 28)
        org_fea = torch.stack(org_fea).transpose_(0, 1).contiguous().view(-1, 192, 28, 28)

        pred = self.gen_fea_cls(gen_fea)
        return pred

net = TSN_BIT()
net.tsn._init_kernelCNN(slide_wind)
net.eval()
net.cuda()

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

test_loader = torch.utils.data.DataLoader(
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
    num_workers=workers, pin_memory=True, drop_last=False)
print("Length of dataset is {}".format(len(test_loader)))

''' Implement Multi-GPU strategy '''
net = torch.nn.DataParallel(net, device_ids=[0]).cuda() # multi-GPU
print("Number of GPU we have is {}".format(torch.cuda.device_count()))

''' Start Testing Process '''
accur = []
gt = []
for epoch in range(1):
    for idx, (input, target, indice) in enumerate(test_loader):

        ''' with no_grad(), help decrease memory cost '''
        with torch.no_grad():

            ''' Dynamically infer the batch_size '''
            b_shape = input.shape[0]
            input_var = input.view(b_shape * data_length, 10, 224, 224).cuda()
            input_var = torch.autograd.Variable(input_var).cuda()
            target = target.detach()

            '''
                use org_fea for testing if accuray is still the same;
                use gen_fea for testing our feature generating ability;
            '''

            out = net(input_var, warmup_t, pred_t, slide_wind).detach()
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