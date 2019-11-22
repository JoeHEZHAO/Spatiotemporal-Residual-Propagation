import torch
from torch import nn
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
from torch.autograd import Variable, grad

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../videoPrediction'))
from util import *
from jacobian import *

'''
    ## Inherit from model_temp_res_gen.py;
    ## Implement kalman filter for residual motion;

    Update 2019.01.29 First Try:
        1. Prediction: y_t_prior = G(y_t-1_post)
                    P_t_prior = f(P_t-1), P_0 sample from N(mean, variance); (Could conditioned on action label)

        2. Update: K_t = P_t_prior * H * (H * P_t_prior * H^t + R)-1
                y_t_post = y_t_prior + K_t * (Z_t - H * y_t_prior)
                P_t_post = (I - K_t * H) * P_t_prior

        3. Add kalman filter two stage to fea_gen_forward()
        4. inverse is working as a normalize term for P, resulting K to be [0, 1] ??

    Updata 2019.01.30 8pm:
        1. Adding Jacobian Matrix, for input and output of self.resgen func;
        2. 
'''

from BNInception_model import bninception

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print((
        """To see if 
            Initializing TSN with base model: {}.
            TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        '''
            Define Residual Motion Generator
        '''
        self.res_gen = TemResGen()

        '''
            Import Sobel Filter for Image gradient
        '''
        self.image_grad = SobelFilter_Diagonal(1, 1)

        self.sigmoid = torch.nn.Sigmoid()
    
    def _kalman_param_init(self, size):

        '''
            Feature Diff is [28, 28];
            Sample covariance matrix P_0 from Normal(mean, variance) with same size [28, 28];

            process noise and measurement noise is fixed constant (0.05) size [28, 28];

            measurement transition matrix H is identity matrix;

            state transition matrix A is Jacobian Matrix, regards to input and output of self.res_gen;

            Choosing of init P_0 is quite chaos; Are we training P_0 ?? According to all previous implementation, P is always not training;
            
            I will do:
                1. Sample P from Normal(10, 1), larger P means less accurate our prediction is; Smaller P means prediction is reliable;
                2. P_t input to same self.res_gen, to generate the prior of t+1;
                3. Following equation above, to predict all obs;
                4. Each intermediate P would be penalty by l2 norm, to enforce self.res_gen generate accurate results;
        
            Update 2019.01.29:
                1. Dynamically infer param size;
                2. Put this into begining of fea_gen_forward();

        '''

        self.R = torch.ones(size, requires_grad=True).cuda() * 0.05

        self.Q = torch.ones(size, requires_grad=True).cuda() * 0.05

        self.P = Variable(torch.from_numpy(np.random.normal(loc=0, scale=0.5, size=size).astype('float64'))).cuda()

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':

            self.base_model = bninception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '../'))
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)


    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t):

        '''
        2019.01.29 equipped with Kalman Filter Procedure to ops 3

        Operations:
            1. Extract Intermediate Layer Features;
            2. Find Feature Diff;
            3. Loop for Generate New Feature Diff and form New feature:
                3.1 This time, try generating feature diff with only one input;
                3.2 Assume take in 14 frames, 13 fea diff, then use the first fea diff to generate the rest;
                3.3 Warm_up and pred_time is going to be different for different datasets; Need to be adaptive;
                3.4 Fix warm_up_t as 1 and pred_t as 12 now, for JHMDDB dataset;
            4. Return Feature Gradient of org and gen, to calculate the loss;
            5. Return both Generated and Origin Feature, Feature Gradient to calculate the loss;
            6. Return P_list[1:], to optim on l2 norm;
        '''

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        input = input.view((-1, sample_len) + input.size()[-2:])

        ''' ops 1 '''
        with torch.no_grad():
            org_fea = self.base_model.extract_feature(input)

        if self.modality == 'Flow':
            org_fea = list(torch.unbind((org_fea.view(batch_size, self.new_length * 2, -1, 28, 28)), 1))
        else:
            org_fea = list(torch.unbind((org_fea.view(batch_size, self.num_segments, -1, 28, 28)), 1))

        ''' ops 2 '''
        fea_diff = []
        for x, y in zip(org_fea[1:], org_fea[:-1]):
            fea_diff.append(torch.unsqueeze(torch.mean((x-y), dim=-3), 1))

        ''' ops 3 '''
        gen_diff = []
        gen_fea = [x.clone() for t, x in enumerate(org_fea) if t < warmup_t] # start from first two, going to be extended every time step;
        gen_diff = [x.clone() for t, x in enumerate(fea_diff) if t < warmup_t - 1] # start from first one, going to be extended every time step;

        self._kalman_param_init(fea_diff[0].shape)

        P_list = []
        P_list.append(self.P)
        K_list = []
        for i in range(pred_t): 

            '''  Prediction Step '''
            pre_diff = Variable(gen_diff[-1].cuda(), requires_grad=True)
            new_diff = self.res_gen(pre_diff)

            ''' test on jacobian matrix func '''
            b_shape = gen_diff[0].shape[0]
            # j_matrix = self.jacobian(gen_diff[-1].view(b_shape, -1), new_diff.view(b_shape, -1))
            # H_matrix = compute_jacobian(pre_diff, new_diff)

            new_p = self.res_gen(P_list[-1].float()) + self.Q

            ''' Update v1 Step element-wise multiply '''
            ''' Problem: K too small, so that 1-K could be close to 1 '''
            ''' Solution: normalize K by norm 2 and use sigmoid to map it into [0, 1] '''
            K = new_p.mul(torch.inverse(new_p + self.R))
            K = self.sigmoid(K / K.norm(2))
            new_diff = (1-K).mul(new_diff) + K.mul(fea_diff[i+1]) # fea_diff[i+1] is the gt for new_diff[t]
            new_p = (1-K).mul(new_p)

            ''' Update v2 Step: matrix multiplication '''
            # K = torch.matmul(new_p, torch.inverse(new_p + self.R))
            # K = self.sigmoid(K / K.norm(2))
            # new_diff = torch.matmul((1-K),new_diff) + torch.matmul(K, fea_diff[i+1]) # fea_diff[i+1] is the gt for new_diff[t]
            ''' new_p increase by power of 10 '''
            # new_p = torch.matmul(1-K, new_p)
            # new_p = self.sigmoid(new_p / new_p.norm(2))

            ''' Save all intermediate states '''
            new_fea = gen_fea[-1] + new_diff
            gen_fea.append(new_fea)
            gen_diff.append(new_diff)
            
            P_list.append(new_p)
            K_list.append(K)

        ''' ops 4 '''
        if self.train:
            gen_dif_grad = [self.image_grad(x) for x in gen_diff]
            org_dif_grad = [self.image_grad(x) for x in fea_diff]
            return gen_fea, org_fea, gen_dif_grad, org_dif_grad, P_list[:1]

    def gen_fea_cls(self, input):
        """
            Input: Generate Intermediate Feature with size [batch * num_segments/data_length, 192, 28, 28]
            Output: Last layer of TSN network, [batch, 1024]

            Notice: This part assume to be pre-trained; Usually it is not going to be trained;

            Operation:
                1. Call gen_fea_cls function of base model;
        """

        x = self.base_model.gen_fea_cls(input)

        if self.dropout > 0:
            x = self.new_fc(x)

        if not self.before_softmax:
            x = self.softmax(x)
        if self.reshape:
            if self.modality == 'RGB':
                x = x.view((-1, self.num_segments) + x.size()[1:])
            elif self.modality == 'Flow':
                x = x.view((-1, self.new_length * 2) + x.size()[1:])

        x = self.consensus(x)
        return x.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

class TemResGen(nn.Module):

    def __init__(self):
        super(TemResGen, self).__init__()
        self.res1 = Bottleneck(1, 1)
        self.res2 = Bottleneck(1, 1)

    def forward(self, input):

        x = self.res1(input)
        x = self.res2(input)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = out

        return out

if __name__ == '__main__':

    # testing
    # Why this model do not pre-load weights ?
    batch = 4
    num_segments = 14

    model = TSN(101, num_segments, 'RGB',
            base_model='BNInception',
            consensus_type='avg', dropout=0.8, partial_bn=True)

    # print(model)

    from torch.autograd import Variable
    test = Variable(torch.rand(batch, num_segments, 3, 224, 224))

    out = model.fea_gen_forward(test)
    print(out.shape)