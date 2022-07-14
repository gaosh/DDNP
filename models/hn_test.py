from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset
from .k_gradients import *

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    #U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0):
    # if inference:
    #     y = logits + offset + 0.57721
    #     return F.sigmoid(y / T)

    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    #y = logits + offset
    return F.sigmoid(y / T)

def hard_concrete(out):
    out_hard = torch.zeros(out.size())
    out_hard[out>=0.5]=1
    out_hard[out<0.5] = 0
    if out.is_cuda:
        out_hard = out_hard.cuda()
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    out_hard = (out_hard - out).detach() + out
    return out_hard

class importance_generator_ip(nn.Module):
    def __init__(self, structure=None, wn_flag=False, score_function='sigmoid', learnable_input=False):
        super(importance_generator_ip, self).__init__()

        self.bn1 = nn.LayerNorm([128])

        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128, bidirectional=False)

        self.h0 = torch.zeros(1, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad = False

        if not learnable_input:
            self.inputs.requires_grad = False



        # if wn_flag:
        #     self.linear_list = [weight_norm(nn.Linear(128, structure[i], bias=False)) for i in range(len(structure))]
        # else:
        self.linear_list = []
        for i in range(len(structure)):
            layers = []
            layers.append(nn.Linear(128, structure[i], bias=False, ))
            if wn_flag:
                layers.append(nn.LayerNorm([structure[i]]))
            self.linear_list.append(nn.Sequential(*layers))
        # self.linear_list = [nn.Linear(128, structure[i], bias=False, ) for i
        #                         in range(len(structure))]


        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.r_list = [torch.zeros(self.structure[i]) for i in range(len(self.structure))]
        self.r_lr = 0.99
        self.base = 3.0

        if score_function == 'sigmoid':
            self.score_function = torch.sigmoid
        elif score_function == 'mae':
            self.score_function = torch.abs
        elif score_function == 'mse':
            self.score_function = mse_function
        else:
            raise NotImplementedError

    def set_fn_list(self, fn_list):
        self.fn_list = fn_list
        print(self.fn_list[0].max())
        print(self.fn_list[0].min())

    def forward(self, k_masks, detach=False, ipout=False):
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        if detach:
            with torch.no_grad():
                outputs, hn = self.Bi_GRU(self.inputs, self.h0)
                outputs = [F.relu(self.bn1(outputs[i,:])) for i in  range(len(self.structure))]
                outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        else:
            outputs, hn = self.Bi_GRU(self.inputs, self.h0)
            outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]
            outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        ori_masks = []
        cout_lists = []
        for i in range(len(outputs)):
            im_output = outputs[i]
            # [:,:-1]
            g = sample_gumbel(im_output.size())
            if outputs[i].is_cuda:
                g = g.cuda()
            if not self.training:
                g = 0

            if self.training:
                if self.score_function == torch.sigmoid:
                    importance = torch.sigmoid(im_output + g  + self.base).squeeze()
                else:
                    score = self.score_function(im_output)
                    importance =  score/score.max()
                if im_output.is_cuda:
                    self.r_list[i] = self.r_list[i].cuda()
                self.r_list[i] = self.r_lr * self.r_list[i] + (1 - self.r_lr) * importance.detach()
                # importance = self.fn_list[i]
            else:
                importance = self.r_list[i]
                # importance = self.fn_list[i]
            _, index = torch.sort(importance, descending=True)



            c_mask = k_masks[i].gather(0, index.squeeze().argsort())

            if self.training:

                current_out = torch.sigmoid((im_output + g + self.base) / 0.4).squeeze()
                c_mask = c_mask*current_out
                if ipout:
                    # pre_discrete = current_out
                    cout_lists.append(hard_concrete(current_out))

            ori_masks.append(c_mask)

        if ipout and self.training:

            return ori_masks, cout_lists

        else:
            return ori_masks


class k_generator(nn.Module):
    def __init__(self, structure, soft_range = 0.05, offset=3.0, learnable_input=False):
        super(k_generator, self).__init__()


        self.num_layers = len(structure)
        self.structure = structure
        self.inputs = nn.Parameter(torch.randn(1, 32))


        # self.inputs = nn.Parameter(torch.randn(1, 64))
        #test only
        # self.inputs = nn.Parameter(0.95*torch.ones(1, self.num_layers))
        self.offset = offset
        self.soft_range = soft_range

        if not learnable_input:
            self.inputs.requires_grad = False


        layers = []

        layers.append(nn.Linear(32, 64, bias=False))
        layers.append(nn.LayerNorm([64]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, self.num_layers, bias=False))

        self.fc_layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm([self.num_layers])

    def forward(self, given_inputs=None):
        if given_inputs is not None:
            inputs = given_inputs
        else:
            inputs = self.inputs

        masks = []
        outputs = torch.sigmoid((self.fc_layers(inputs))+self.offset).squeeze()
        # outputs = self.inputs.squeeze()
        # print(self.structure[0])

        for i in range(self.num_layers):
            if self.training:
                current_range = self.soft_range*self.structure[i]
                c_mask = k_gradient.apply(outputs[i], self.structure[i], int(current_range), 0.025)
            else:
                c_mask = k_gradient.apply(outputs[i], self.structure[i], 0, 0)
            masks.append(c_mask)

        return masks, outputs
