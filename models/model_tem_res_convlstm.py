'''
    Inherit from model_temp_res_gen.py;

    Updates 2019.01.26:
        1. Use channel-pooling for low-dimensional (1 channel) feature representation; I.E. torch.mean((x-y), dim=-3)
        2. Use three 3x3 residual block for 1-channel feature generating;
        3. Adding 1-channel new feature diff to each channel of observed feature;
        4. Remove BN layer, so set bias as True for conv1x1 and conv3x3;

    Update 2019.01.27:
        0. Adding back bn layer and set Bias=False;
        1. Import self-defined Sobel Filter for image gradient regularization;
        2. Sobel Filter Reference: /home/zhufl/videoPredicton/util.py: SobelFilter/SobelFilter_Diagonal;
        3. def fea_gen_forward() return gradient for gen_fea and org_fea;

    Update 2019.01.28:
        1. torch.max(input, dim=1) return 2; https://pytorch.org/docs/stable/torch.html

    Update 2019.02.04:
        1. Add ground truth feature diff for each time step;
        2. Use RNN to generate weights (Kalman Gain) for each added ground truth feature;
        3. Optimize overall MSE and weighted penalty on early Kalman Gain, to encourage on early accuracy modeling;
        4. Kalman Gain should depends on generated feature itself;
        5. Jacobian matrix is hard to model or it will take huge effort (Like train model in Multiple/Separate Step);
        6. Use RNN to model the relation between generated fea_diff and kalman gain for next step;
        7. In rnn, enable relu to convert value to positive; Then use sigmoid to output a probability between [0, 1];
        8. Could also be viewed as attention module: http://openaccess.thecvf.com/content_cvpr_2018/papers/Nguyen_Weakly_Supervised_Action_CVPR_2018_paper.pdf;
        9. TODO: What is the gradient flow in my case ?? I could really try the way ECCV18 did it; And find a way to modify it;
        10. TODO: If the ECCV18 way really did well, I think it would be ok to adopt;
        11. TODO: Set self.kalman_update, to allow ground truth feature_diff get involved;

    Update 2019.02.06:
        1. In def gen_fea_cls() function, Need to set parameter Num_frame_use, to allow using partial number of frames to do testing;
        2. For now, I manually set it as 4;
        3. Should I pass it as parameter ? or Should I init the parameter together with class ?


    Update 2019.02.08:
        1. Change the rnn kalman gain to convlstm kalman gain;
        2. Input: Difference among gt_diff and new_diff;
        3. Output: [batch, 1, 28, 28], spatial preserving kalman gain;
        4. Put torch.sigmoid onto new_kalman_gain, since originally, its value is not range from [0, 1];
'''
import torch
from torch import nn
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../videoPrediction'))
from util import *
from BNInception_model import bninception
from rnn import *
from convlstm import *
from models import TSN as base_model

class kalman_rnn(RNN):

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = torch.sigmoid(output)
        return output, hidden

class TSN(base_model):

    def __init__(self):
        # Init Temporal Residual MOtion Generation Component;
        super(TSN, self).__init__(101, num_segments, 'RGB',
            base_model='BNInception',
            consensus_type='avg', dropout=0.8, partial_bn=True)
        '''
            Define Residual Motion Generator
        '''
        self.res_gen = TemResGen(192, 192)

        '''
            Import Sobel Filter for Image gradient
        '''
        self.diff_grad = SobelFilter_Diagonal(192, 192)
        self.fea_grad = SobelFilter_Diagonal(192, 192)

        '''
            RNN for recursive kalman gain, which is singular value;
        '''
        # self.rnn = kalman_rnn(1568, 256, 1).cuda()
        self.rnn = ConvLSTMCell(192, 192).cuda()

        self.kalman_update = False
        self.train = False

    def fea_gen_forward(self, input, batch_size, warmup_t, pred_t):

        '''
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
            fea_diff.append((x-y).detach())

        ''' Regenerate feature from fea_diff and the starter feature, which is org_fea[0] '''
        ''' To testify that feature re-added is able to reproduce same results '''
        # accumlate_fea = []
        # accumlate_fea.append(org_fea[0])
        # for i in range(len(fea_diff)):
        #     accumlate_fea.append(accumlate_fea[-1] + fea_diff[i])
        # import pdb;pdb.set_trace()

        ''' ops 3 '''
        gen_diff = []
        kalman_gain_list = []
        gen_fea = [x.clone() for t, x in enumerate(org_fea) if t < warmup_t] # start from first two, going to be extended every time step;
        gen_diff = [x.clone() for t, x in enumerate(fea_diff) if t < warmup_t - 1] # start from first one, going to be extended every time step;

        b_shape = gen_diff[0].shape[0]
        state = None
        state_kalman = None
        # import pdb;pdb.set_trace()

        for i in range(pred_t):
            # if i == 0:
                # hidden = self.rnn.initHidden(b_shape).cuda()
            state = self.res_gen(gen_diff[-1], state)
            new_diff = state[0]

            # new_kalman_gain, hidden = self.rnn(torch.cat([torch.mean(gen_diff[-1], dim=-3).view(b_shape, -1), torch.mean(fea_diff[warmup_t - 2 + i], dim=-3).view(b_shape, -1)], -1), hidden)
            state_kalman = self.rnn(fea_diff[warmup_t - 2 + i] - gen_diff[-1], state_kalman)
            new_kalman_gain = torch.sigmoid(state_kalman[0])

            ''' kalman gain update procedure, if training '''
            ''' use last time generated fea diff and gt diff, for next-time kalman-gain '''
            if self.kalman_update:
                # new_diff = new_kalman_gain.view(b_shape, 1, 1, 1) * fea_diff[warmup_t - 1 + i] + (1.0 - new_kalman_gain.view(b_shape, 1, 1, 1)) * new_diff
                # new_diff = (1 - new_kalman_gain.view(b_shape, 1, 1, 1)) * fea_diff[warmup_t - 1 + i] + new_kalman_gain.view(b_shape, 1, 1, 1) * new_diff
                new_diff = new_kalman_gain.mul(fea_diff[warmup_t - 1 + i]) + (1.0 - new_kalman_gain).mul(new_diff)

            # Update
            new_fea = gen_fea[-1] + new_diff
            gen_fea.append(new_fea)
            gen_diff.append(new_diff)
            kalman_gain_list.append(new_kalman_gain)

        ''' ops 4 '''
        gen_dif_grad = [self.diff_grad(x) for x in gen_diff]
        org_dif_grad = [self.diff_grad(x) for x in fea_diff]

        gen_fea_grad = [self.fea_grad(x) for x in gen_fea]
        org_fea_grad = [self.fea_grad(x) for x in org_fea]

        if self.train:
            return gen_fea, org_fea, gen_diff, fea_diff, gen_dif_grad, org_dif_grad, gen_fea_grad, org_fea_grad, kalman_gain_list
        else:
            return gen_fea, org_fea, gen_dif_grad, org_dif_grad, kalman_gain_list
            # return gen_fea, accumlate_fea, gen_dif_grad, org_dif_grad, kalman_gain_list

    def gen_fea_cls(self, input, Num_frame_use=None):
        """
            Input: Generate Intermediate Feature with size [batch * num_segments/data_length, 192, 28, 28]
            Output: Last layer of TSN network, [batch, 1024]

            Notice: This part assume to be pre-trained; Usually it is not going to be trained;

            Operation:
                1. Call gen_fea_cls function of base model;
                2. Check num_frame_use
                3.
        """

        x = self.base_model.gen_fea_cls(input)

        if self.dropout > 0:
            x = self.new_fc(x)

        if not self.before_softmax:
            x = self.softmax(x)
        if self.reshape:
            if self.modality == 'RGB':
                if Num_frame_use:
                    x = x.view((-1, Num_frame_use) + x.size()[1:])
                else:
                    x = x.view((-1, self.num_segments) + x.size()[1:])

            elif self.modality == 'Flow':
                x = x.view((-1, self.new_length * 2) + x.size()[1:])

        x = self.consensus(x)
        return x.squeeze(1)


class TemResGen(nn.Module):

    def __init__(self, dim_in, hidden):
        super(TemResGen, self).__init__()
        self.convlstm = ConvLSTMCell(dim_in, hidden)
        # self.convlstm2 = ConvLSTMCell(dim_in, hidden)

    def forward(self, input, state):

        x, y = self.convlstm(input, state)
        # x, y = self.convlstm2(x, y)
        return x, y

if __name__ == '__main__':

    # testing
    # Why this model do not pre-load weights ?
    batch = 32
    num_segments = 14

    model = TSN().cuda()
    print(model)

    from torch.autograd import Variable
    test = Variable(torch.rand(batch, num_segments, 3, 224, 224)).cuda()

    out = model.fea_gen_forward(test, batch, 2, 12)
    print(out[0][0].shape)
