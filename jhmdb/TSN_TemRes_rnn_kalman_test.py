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

    Update 2019.02.06:
        1. Set up Num_frame_use for showing accuracy under limited number of frames;
        2. Use Num_frame_use in forward func; After stack list of tensor together one;
        3. Set Num_frame_use as 4, to use 2 obs and 2 generated future;
        4. Set Num_frame_use as None, to use x obx and all the other generated future, 14 - x for now;

"""
from __future__ import division
import os, sys, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

cur_path = os.path.dirname(os.path.realpath(__file__))
''' Use class/package from tsn-pytorch'''
sys.path.append(os.path.join(cur_path, '../../tsn-pytorch'))
from model_tem_res_gen_v3 import TSN
from transforms import *
from ops import ConsensusModule
from dataset_JHMDB import TSNDataSet
sys.path.append(os.path.join(cur_path, '../'))
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
train = False
kalman_update = False
Num_frame_use = None

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
        self.model_name = 'TSN_TemResGen_Kalman_2019-02-07_21-56-10.pth'
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

    # def forward(self, input):
    #     x = self.activation(self.tsn(input))
    #     x = self.activation(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t):
        """
            For testing, return 5 parameters:
                1. gen_fea;
                2. org_fea;
                3. gen_fea_grad;
                4. org_fea_grad;
                5. kalman_gain;
        """
        x, y, _, _, kalman_gain = self.tsn.fea_gen_forward(input, batch_size, warmup_t, pred_t)
        return x, y

    def gen_fea_cls(self, input, num_frame_use):
        """
            Wrapper func for model_temp_res_gen.TSN.gen_fea_cls;

            Input: Intermediate feature [batch * num_segments/data_length, 192, 28, 28];
            Call: gen_fea_cls of model_temp_res_gen.TSN, to output [batch, 101/51], depends which weight is being used;
            Ouput: [batch, num_class]
        """
        x = self.tsn.gen_fea_cls(input, num_frame_use)
        x = self.activation(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input, warmup_t, pred_t, num_frame_use):

        b_shape = int(input.shape[0] / num_segments)
        gen_fea, org_fea = self.fea_gen_forward(input, b_shape, warmup_t, pred_t)
        gen_fea = torch.stack(gen_fea).transpose_(0, 1)
        org_fea = torch.stack(org_fea).transpose_(0, 1)

        # [:None] equals to [:] ==> 'all'
        gen_fea = gen_fea[:, :num_frame_use, :, :, :].contiguous().view(-1, 192, 28, 28)
        org_fea = org_fea[:, :num_frame_use, :, :, :].contiguous().view(-1, 192, 28, 28)

        pred = self.gen_fea_cls(org_fea, num_frame_use)
        return pred

net = TSN_BIT()
net.eval()
net.tsn.train = train
net.tsn.kalman_update = kalman_update

''' Load Dataset '''
''' data_length can control how many segments can we get from individual video '''
train_list = os.path.join(cur_path, '../data/JHMDB_train.txt')
test_list = os.path.join(cur_path, '../data/JHMDB_test.txt')
train_augmentation = net.tsn.get_augmentation()

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
    num_workers=workers, pin_memory=True, drop_last=False)
print("Length of dataset is {}".format(len(train_loader)))

''' Implement Multi-GPU strategy '''
net = torch.nn.DataParallel(net, device_ids=[0]).cuda() # multi-GPU
print("Number of GPU we have is {}".format(torch.cuda.device_count()))

''' Start Testing Process '''
accur = []
gt = []
for epoch in range(1):
    for idx, (input, target, indice) in enumerate(train_loader):

        ''' with no_grad(), help decrease memory cost '''
        with torch.no_grad():

            ''' Dynamically infer the batch_size '''
            b_shape = input.shape[0]
            input_var = input.view(b_shape * num_segments, 3, 224, 224).cuda()
            input_var = torch.autograd.Variable(input_var).cuda()
            target = target.detach()

            '''
                use org_fea for testing if accuray is still the same;
                use gen_fea for testing our feature generating ability;
            '''

            out = net(input_var, warmup_t, pred_t, Num_frame_use).detach()
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
