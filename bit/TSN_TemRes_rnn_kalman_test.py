"""
    Implementation of TSN Residual Motion Generation for BIT Dataset Classification; 
    
    1. Using the same sampling strategy for testing (Set test_mode = True), which is getting 10 fixed-index clips with center-cropping and normalization;
    2. Using 2 frames as warm-up, to generate the rest of 8 frames, in total 10 clips;
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
        1. Migrate TSN Residual Motion Generation code from JHMDB to BIT;
        2. Modify the dataloader, set test_model=True, to allow 10 clips;
        3. Modify the loss to MSELoss();
        4. Remember, BIT is using Flow branch of UCF101 pretrained Weights, so num_class=101; 
           ucf101_flow.pth located @ '/home/zhufl/Workspace/tsn-pytorch/ucf101_flow.pth'
        5. data_length=10 & num_segments=5;

    Update 2019.01.26:
        1. org_fea only has 89.84%, compared with 92.97%, it is a large drop-down;
        2. Potential Problem might be the changing of first BN layer; Try freeze all BN layers, 
        as in RGB_OFF_v2.py, to see if there is performance drop with org_fea;
        3. TSN_TemResGen_2019-01-27_15-00-55.pth confirmed that freezing all BN layer, for keep origin accuracy, is necessary;
        4. How to make TemResGen works for BIT ??
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
from deprecated import deprecated

cur = os.path.dirname(os.path.realpath(__file__))

''' Use class/package from tsn-pytorch'''
sys.path.append(os.path.join(cur, '../../tsn-pytorch'))
from model_tem_res_gen_v3 import TSN
from transforms import *
from ops import ConsensusModule
from dataset_BIT import TSNDataSet

sys.path.append('../')
from model_utils import topk_crossEntrophy

''' Config '''
arch = 'BNInception'
num_class = 101
modality = 'Flow'
crop_fusion_type= 'avg'
num_segments = 5
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 16
workers = 1
data_length = 10
warmup_t = 4
pred_t = 10 - warmup_t
train = False
kalman_update = True

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
        self.model_name = 'TSN_TemResGen_Kalman_2019-03-11_15-28-57.pth'
        self._load_pretrained_model(self.model_name)

    def _load_pretrained_model(self, model_name):

        """
            Load pretrained model that contains all weights for all layers;
            Allow missing parameters;
        """

        checkpoint = torch.load('/home/zhufl/Temporal-Residual-Motion-Generation/videoPrediction/BIT_train_test/' + model_name)
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

        flow_weights = '/home/zhufl/Workspace/tsn-pytorch/ucf101_flow.pth'
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

    @deprecated
    def forward(self, input):
        x = self.activation(self.tsn(input))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

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

    def forward(self, input, warmup_t, pred_t):

        b_shape = int(input.shape[0] / num_segments)
        gen_fea, org_fea = self.fea_gen_forward(input, b_shape, warmup_t, pred_t)
        gen_fea = torch.stack(gen_fea).transpose_(0, 1).contiguous().view(-1, 192, 28, 28)
        org_fea = torch.stack(org_fea).transpose_(0, 1).contiguous().view(-1, 192, 28, 28)

        pred = self.gen_fea_cls(gen_fea)
        return pred

net = TSN_BIT().cuda()
net.eval()
net.tsn.train = train
net.tsn.kalman_update = kalman_update

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

''' Start Testing Process '''
accur = []
gt = []
for epoch in range(1):
    for idx, (input, target, indice) in enumerate(train_loader):

        with torch.no_grad():

            ''' Dynamically infer the batch_size '''
            b_shape = input.shape[0]
            input_var = input.view(batch_size * data_length, 10, 224, 224)
            input_var = torch.autograd.Variable(input_var, requires_grad=True).cuda()
            target = target.detach()

            gen_fea, org_fea = net.fea_gen_forward(input_var, b_shape, warmup_t, pred_t)
            gen_fea = torch.stack(gen_fea).transpose_(0, 1).contiguous().view(b_shape * data_length, -1, 28, 28)
            org_fea = torch.stack(org_fea).transpose_(0, 1).contiguous().view(b_shape * data_length, -1, 28, 28)

            ''' 
                use org_fea for testing if accuray is still the same;
                use gen_fea for testing our feature generating ability;
            '''
            out = net.gen_fea_cls(gen_fea).detach()

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