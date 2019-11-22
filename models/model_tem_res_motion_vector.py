'''
    Try for using self-defined BNInception Model, which allows extract intermediate features;
    The module is located at BNInception_model.py
    Inspired by recent work, pixel-level motion modeling seems makes more sense;

    Update 2019.01.25:
        1. Use two-layer Residual-Block with bottleneck, reference :
        2. TODO: adaptive specify warm_up_t and pred_t;
        3. TODO: remove all intermediate variable memory, if needs to train on larger batch_size; Use 64 for 14 frames for now;
            3.1 JHMDB: 14 frames;
            3.2 UCF101: 25 frames;
            3.3 BIT: 10 clips;
        4. TODO: def gen_fea_cls(self, input), that use generated feature to do final classification;
        5. Adding warmup_t, pred_t to def fea_gen_forward(); Allow dynamic warmup and pred time step changing;
        6. Modify the org_fea torch.unbind for Flow branch, since data_length is working rather than num_segments;
           However, data_length is not passing to TSN model but TSN_dataloader, using 2 * new_length;
        7. Attention: Consensus happens within TSN, before JHMDB/BIT fine-tuning process; So in my self-defined gen_fea_cls, I need to to Consensus;

    Update 2019.01.25 3:43pm:
        1. Done all before requirements;
        2. Need to solve the issue of generated feature not performance well;

    Update 2019.02.20 1:10pm:
        1. Generate multi-scale motion vector with size [3, 3], [5, 5], [7, 7] from stacked-residual-motion [192, 28, 28] * 2, for each channel;
        2. Apply motion vector onto the most recent feature residual and sum together different channel;
 
    Update 2019.02.22 5:17pm:        
        1. Apply different scale sobel filter loss, to keep both local and global structure salienct;
        2. Use the Down-Stream Auxliary Loss is [l2, mse, gan] on a second-last-layer activation, as discrminative-aware loss;
        3. 

'''
import os, sys
import torch
from torch import nn
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import torch.nn.functional as F
from BNInception_model import bninception

cur_path  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '../videoPrediction'))
from util import *

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True, batch=32):
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
        """
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

        # Define Sobel Operator for Local Spatial Structure Loss
        self.diff_grad = SobelFilter_Diagonal(192, 192)

    def _init_kernelCNN(self, slide_wind):

        '''Define Residual Motion Generator
            @param: warmup: Number of frames used within sliding window;
        '''
        self.kernelCNN = KernelCNN(192 * slide_wind)

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

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t, len_wind):

        '''Generate future feature from residual
        Annotation:
            1. Extract Intermediate Layer Features;
            2. Find Feature Diff;
            3. Loop for Generate New Feature Diff and form New feature:
                3.1 This time, try generating feature diff with only one input;
                3.2 Assume take in 14 frames, 13 fea diff, then use the first fea diff to generate the rest;
                3.3 Warm_up and pred_time is going to be different for different datasets; Need to be adaptive;
                3.4 Fix warm_up_t as 1 and pred_t as 12 now, for JHMDDB dataset;
                3.5 Generate motion vectors for next time feature maps, for now it is [3, 3] that works for all channels;
            4. Return both Generated and Origin Feature, to calculate the loss;

        @param: input: video input with shape [batch, num_seg, 3, 224, 224]
        @param: batch_size: batch
        @param: warmup_t: warm up time-step observation to compute residual for later prediction, >= 2;
        @param: pred-t: number of time-step to predict, >=1;
        @param: len_wind: length of sliding window for motion vector inference; 
        
        @return: 
            1. list of generated feature;
            2. list of origin feature;
            3. list of generated residual;
            4. list of origin residual
            5. list of generated diff grad;
            6. list of org diff grad;
        '''

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        input = input.view((-1, sample_len) + input.size()[-2:])

        ''' ops 1 '''
        org_fea = self.base_model.extract_feature(input)

        if self.modality == 'Flow':
            org_fea = list(torch.unbind((org_fea.view(batch_size, self.new_length * 2, -1, 28, 28)), 1)) 
        else:
            org_fea = list(torch.unbind((org_fea.view(batch_size, self.num_segments, -1, 28, 28)), 1)) 

        ''' ops 2 '''
        fea_diff = []
        for x, y in zip(org_fea[1:], org_fea[:-1]):
            fea_diff.append(x-y)

        ''' ops 3 '''
        gen_fea = org_fea[:warmup_t] # start from first two, going to be extended every time step;
        warm_diff = fea_diff[:warmup_t-1] # start from first one, going to be extended every time step;

        for i in range(pred_t):

            # generate kernels for motion_vector
            k_3x3, k_5x5, k_7x7 = self.kernelCNN(torch.cat(warm_diff[-len_wind:], 1))

            # normalize tensor
            norm = k_3x3.norm(2)
            k_3x3 = k_3x3.div(norm)
            norm = k_5x5.norm(2)
            k_5x5 = k_5x5.div(norm)
            norm = k_7x7.norm(2)
            k_7x7 = k_7x7.div(norm)

            # k_3x3 = F.normalize(k_3x3, p=2, dim=1)
            # k_5x5 = F.normalize(k_5x5, p=2, dim=1)
            # k_7x7 = F.normalize(k_7x7, p=2, dim=1)
            # print(torch.max(k_3x3))

            # set weight value to temporaladaptive kernel
            # setattr(self.TemporalAdaptiveCNN, 'conv2d_3x3.weight', torch.unsqueeze(k_3x3, dim=1))
            # setattr(self.TemporalAdaptiveCNN, 'conv2d_5x5.weight', torch.unsqueeze(k_5x5, dim=1))
            # setattr(self.TemporalAdaptiveCNN, 'conv2d_7x7.weight', torch.unsqueeze(k_7x7, dim=1))

            '''schedule sampling '''
            if self.coin_flip:
                flip_coin = ((pred_t - i) / pred_t) * 0.4
                rand_p = random.random()

                if rand_p > flip_coin:
                    # new_diff = self.res_gen(fea_diff[i + warmup_t - 2])
                    x = F.conv2d(torch.transpose(fea_diff[i + warmup_t - 2], 1, 0),  torch.unsqueeze(k_3x3, dim=1), stride=1, padding=1, groups=batch_size)
                    y = F.conv2d(torch.transpose(fea_diff[i + warmup_t - 2], 1, 0),  torch.unsqueeze(k_5x5, dim=1), stride=1, padding=2, groups=batch_size)
                    z = F.conv2d(torch.transpose(fea_diff[i + warmup_t - 2], 1, 0),  torch.unsqueeze(k_7x7, dim=1), stride=1, padding=3, groups=batch_size)
                else:
                    x = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_3x3, dim=1), stride=1, padding=1, groups=batch_size)
                    y = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_5x5, dim=1), stride=1, padding=2, groups=batch_size)
                    z = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_7x7, dim=1), stride=1, padding=3, groups=batch_size)
            else:
                # set weight value to temporal-adaptive kernel
                x = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_3x3, dim=1), stride=1, padding=1, groups=batch_size)
                y = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_5x5, dim=1), stride=1, padding=2, groups=batch_size)
                z = F.conv2d(torch.transpose(warm_diff[-1], 1, 0),  torch.unsqueeze(k_7x7, dim=1), stride=1, padding=3, groups=batch_size)
            
            new_diff = (x + y + z) / 3.0

            # new_diff = self.TemporalAdaptiveCNN(torch.transpose(warm_diff[-1], 1, 0))
            new_diff = torch.transpose(new_diff, 1, 0)
            new_fea = gen_fea[-1] + new_diff

            # Update
            gen_fea.append(new_fea)
            warm_diff.append(new_diff)

        gen_dif_grad = [self.diff_grad(x) for x in warm_diff]
        org_dif_grad = [self.diff_grad(x) for x in fea_diff]

        ''' ops 4 '''
        return gen_fea, org_fea, warm_diff, fea_diff, gen_dif_grad, org_dif_grad

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

class TemporalAdaptiveCNN(nn.Module):

    '''Multi-scale cnn kernel for motion propagation 
    '''

    def __init__(self, dim, batch):
        super(TemporalAdaptiveCNN, self).__init__()
        self.dim = dim
        self.batch = batch
        self.conv2d_3x3 = nn.Conv2d(self.batch, self.batch, 3, stride=1, 
            padding=1, bias=False, groups=self.batch)

        self.conv2d_5x5 = nn.Conv2d(self.batch, self.batch, 5, stride=1, 
            padding=2, bias=False, groups=self.batch)

        self.conv2d_7x7 = nn.Conv2d(self.batch, self.batch, 7, stride=1, 
            padding=3, bias=False, groups=self.batch)

    def forward(self, input):
        x = self.conv2d_3x3(input)
        y = self.conv2d_5x5(input)
        z = self.conv2d_7x7(input)
        
        return x + y + z

class KernelCNN(nn.Module):

    def __init__(self, input_dim):
        super(KernelCNN, self).__init__()
        self.inp_d = input_dim
        self.conv1 = conv1x1(self.inp_d, 192, bias=True)
        self.conv2 = conv3x3(192, 128, bias=True)
        self.conv3 = conv3x3(128, 32, bias=True)
        self.fc1 = torch.nn.Linear(512, 9)
        self.fc2 = torch.nn.Linear(512, 25)
        self.fc3 = torch.nn.Linear(512, 49)

        self.max_poo2d = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        '''Kernel Generation from TCN of Feature Residual
            @param: input: [192 * 3, 28, 28]
            @param: return: [batch, 9] ==> reshape [batch, 3, 3]
        '''
        x = self.max_poo2d(self.relu(self.conv1(input)))
        x = self.max_poo2d(self.relu(self.conv2(x)))
        x = self.max_poo2d(self.relu(self.conv3(x)))
        b_shape = x.shape[0]
        x = x.view(b_shape, -1)

        m = self.relu(self.fc1(x))
        y = self.relu(self.fc2(x))
        z = self.relu(self.fc3(x))
        return m.view(b_shape, 3, 3), y.view(b_shape, 5, 5), z.view(b_shape, 7, 7)

class TemResGen(nn.Module):

    def __init__(self):
        super(TemResGen, self).__init__()
        self.res1 = Bottleneck(192, 48)
        self.res2 = Bottleneck(192, 48)

    def forward(self, input):

        x = self.res1(input)
        x = self.res2(input)
        return x

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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
        out = out

        return out

if __name__ == '__main__':

    # testing
    # Why this model do not pre-load weights ?
    batch = 32
    warmup_t = 3
    num_segments = 14
    pred_t = num_segments - warmup_t

    model = TSN(101, num_segments, 'RGB',
            base_model='BNInception',
            consensus_type='avg', dropout=0.8, partial_bn=True)

    from torch.autograd import Variable
    test_input = Variable(torch.rand(batch, num_segments, 3, 224, 224))

    out = model.fea_gen_forward(test_input, batch, warmup_t, pred_t)


    # test_obs = Variable(torch.rand(batch, 192 * 3, 28, 28))
    # test_last = Variable(torch.rand(batch, 192, 28, 28))

    # k_cnn = KernelCNN(192 * 3)
    # k_3x3, k_5x5, k_7x7 = k_cnn(test_obs)
    # print(k_3x3.shape, k_5x5.shape, k_7x7.shape)
    # print("Producing kernel shape {} from last three observations".format(k_3x3[1]))
    
    # motion_vector = TemporalAdaptiveCNN(192, batch)
    # print(motion_vector.conv2d_3x3.weight[1])
    # setattr(motion_vector, 'conv2d_3x3.weight', torch.unsqueeze(k_3x3, dim=1))
    # setattr(motion_vector, 'conv2d_5x5.weight', torch.unsqueeze(k_5x5, dim=1))
    # setattr(motion_vector, 'conv2d_7x7.weight', torch.unsqueeze(k_7x7, dim=1))

    # print("Init motion_vector_sampler kernel with output from kernelCNN {}, for next-time inference".format(motion_vector.conv2d_3x3.weight.shape))
    # print(motion_vector.conv2d_3x3.weight[1])

    # # Perform adaptive cnn onto test_last
    # test_last = torch.transpose(test_last, 1, 0)
    # out = motion_vector(test_last)
    # print(out.shape)
    # out_convert = torch.transpose(test_last, 1, 0)
    # print(out_convert.shape)