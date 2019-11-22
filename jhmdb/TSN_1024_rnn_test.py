'''
Author: He Zhao
Date: 2019.03.07
Introduction:
    1. Test file for last layer [1024] + rnn style, action prediction;
'''

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

''' Use class/package from tsn-pytorch '''
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '../../tsn-pytorch'))
from transforms import *
from ops import *

# Noted that dataset.py in tsn-pytorch is different from dataset.py in this folder; However, yield good results;
# from dataset import TSNDataSet
from dataset_JHMDB import TSNDataSet

from model_temp_res_gen_rnn import *
sys.path.append(os.path.join(cur_path, '../'))
from model_utils import *
from utils import *

''' Config '''
arch = 'BNInception'
num_class = 51
modality = 'RGB'
crop_fusion_type= 'avg'
num_segments = 14
flow_prefix = 'flow_'
rgb_prefix = 'image_'
batch_size = 16
workers = 1
data_length = 1
warmup_t = 4
pred_t = num_segments - warmup_t

fea_stage = 28
fea_chns = 192

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
        self.model_name = 'TSN_TemResGen_lstm_1024_28_192_2019-03-07_23-13-18.pth'

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
        """Loading Flow Weights and then fine-tune fc layers
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

    def forward(self, input, batch_size, warmup_t, pred_t, fea_stage):
        x, y = self.tsn.fea_gen_forward(input, batch_size, warmup_t, pred_t, fea_stage)
        return x, y

    def gen_fea_cls(self, input, fea_stage):
        x = self.tsn.gen_fea_cls(input, fea_stage)
        x = self.activation(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    # @staticmethod
    # def weights_init(m):
    #     classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    #     if classname.find('Linear') != -1:
    #         weight_shape = list(m.weight.data.size()) #?? list containing the shape of the weights in the object "m"
    #         fan_in = weight_shape[1] # dim1
    #         fan_out = weight_shape[0] # dim0
    #         w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
    #         m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
    #         m.bias.data.fill_(0) # initializing all the bias with zeros

if __name__ == '__main__':

    net = TSN_BIT()
    net.eval().cuda()

    ''' Load Dataset '''
    ''' data_length can control how many segments can we get from individual video '''
    train_list = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', 'JHMDB_train.txt')
    test_list = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', 'JHMDB_test.txt')
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
        for idx, (input, target, indices) in enumerate(train_loader):

            b_shape = input.shape[0]
            input_var = input.view(b_shape * num_segments, 3, 224, 224)
            input_var = torch.autograd.Variable(input_var, requires_grad=True).cuda()
            target = target.detach()

            gen_fea, org_fea = net(input_var, b_shape, warmup_t, pred_t, fea_stage)
            
            gen_fea = torch.stack(gen_fea).transpose_(0, 1).contiguous().view(-1, 1024)
            org_fea = torch.stack(org_fea).transpose_(0, 1).contiguous().view(-1, 1024)
            
            ''' Sub-sample number of frames for recognition '''
            # gen_fea = gen_fea.contiguous().view(-1, fea_chns, fea_stage, fea_stage)
            # org_fea = org_fea.contiguous().view(-1, fea_chns, fea_stage, fea_stage)

            out = net.gen_fea_cls(gen_fea, fea_stage).detach()
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