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


class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #train = train
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        # print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None

class k_generator(nn.Module):
    def __init__(self, structure, soft_range = 0.05, offset=3.0, learnable_input=False):
        super(k_generator, self).__init__()


        self.num_layers = len(structure)
        self.structure = structure
        self.inputs = nn.Parameter(torch.randn(1, self.num_layers))


        # self.inputs = nn.Parameter(torch.randn(1, 64))
        #test only
        # self.inputs = nn.Parameter(0.95*torch.ones(1, self.num_layers))
        self.offset = offset
        self.soft_range = soft_range

        if not learnable_input:
            self.inputs.requires_grad = False



        # self.linear_list = [nn.Linear(64, 1, bias=False, ) for i
        #                     in range(len(structure))]

        self.fc_layers = nn.Linear(self.num_layers, self.num_layers, bias=False)
        self.ln = nn.LayerNorm([self.num_layers])

    def forward(self,):
        masks = []
        outputs = torch.sigmoid((self.fc_layers(self.inputs))+self.offset).squeeze()
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

class simple_k(nn.Module):
    def __init__(self, structure, soft_range=0.05, offset=3.0):
        super(simple_k, self).__init__()

        self.num_layers = len(structure)
        self.structure = structure
        self.params = nn.Parameter(torch.randn(1, self.num_layers))

        self.offset = offset
        self.soft_range = soft_range
        self.ln = nn.LayerNorm([self.num_layers])
    def forward(self,):
        masks = []
        outputs = self.ln(self.params)
        outputs = torch.sigmoid(outputs + self.offset).squeeze()

        for i in range(self.num_layers):
            if self.training:
                current_range = self.soft_range*self.structure[i]
                c_mask = k_gradient.apply(outputs[i], self.structure[i], int(current_range), 0.025)
            else:
                c_mask = k_gradient.apply(outputs[i], self.structure[i], 0, 0)
            masks.append(c_mask)

        return masks, outputs

def mse_function(inputs):
    return inputs.pow(2)

class importance_generator(nn.Module):
    def __init__(self, structure=None, wn_flag=False, score_function='sigmoid', learnable_input=False):
        super(importance_generator, self).__init__()

        self.bn1 = nn.LayerNorm([256])

        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)

        self.h0 = torch.zeros(2, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)

        if not learnable_input:
            self.inputs.requires_grad = False


        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(256, structure[i], bias=False)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(256, structure[i], bias=False, ) for i
                                in range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.r_list = [torch.zeros(self.structure[i]) for i in range(len(self.structure))]
        self.r_lr = 0.995

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

    def random_forward(self, lb = 0.05):
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i,:])) for i in  range(len(self.structure))]

        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        ori_masks = []

        for i in range(len(outputs)):
            im_output = outputs[i]

            g = sample_gumbel(im_output.size())

            if outputs[i].is_cuda:
                g = g.cuda()

            current_out = torch.sigmoid((im_output + g + 3.0) / 0.4).squeeze()
            mask = torch.zeros(self.structure[i]).squeeze()
            mask[:int(0.5*self.structure[i])] = 1
            idx = torch.randperm(self.structure[i])
            mask = mask[idx]
            mask = mask + lb
            mask[mask >= 1] = 1
            if im_output.is_cuda:
                mask = mask.cuda()

            c_mask = mask * current_out
            ori_masks.append(c_mask)

        return ori_masks

    def forward(self, k_masks, detach=False):
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

        for i in range(len(outputs)):
            im_output = outputs[i]
            # [:,:-1]
            g = sample_gumbel(im_output.size())
            if outputs[i].is_cuda:
                g = g.cuda()
            if not self.training:
                g = 0
            # current_out = torch.exp(outputs[i])
            # g = 0
            # if self.training:
            # current_out = torch.sigmoid((outputs[i] + g + 3.0) / 0.7).squeeze()
            # else:
            #     current_out = torch.sigmoid((outputs[i] + 3.0)/ 0.7).squeeze()
            #
            # current_out = torch.sigmoid((outputs[i] + 3.0) / 0.7).squeeze()
            # alpha = torch.sigmoid(outputs[i][:,-1])
            if self.training:
                # importance = torch.sigmoid(im_output).squeeze() * (self.fn_list[i]/self.fn_list[i].max())
                importance = self.score_function(im_output).squeeze() * (self.fn_list[i] / self.fn_list[i].max())
                if im_output.is_cuda:
                    self.r_list[i] = self.r_list[i].cuda()
                self.r_list[i] = self.r_lr * self.r_list[i] + (1 - self.r_lr) * importance.detach()
            else:
                importance = self.r_list[i]
            _, index = torch.sort(importance, descending=True)



            c_mask = k_masks[i].gather(0, index.squeeze().argsort())

            if self.training:
                # importance = torch.sigmoid(outputs[i]).squeeze()
                # c_mask = mask_gradient.apply(importance, c_mask)
                # dc_mask = (c_mask-(importance/2).detach()) + importance/2
                # c_mask = dc_mask
                current_out = torch.sigmoid((im_output + g + 3.0) / 0.4).squeeze()
                c_mask = c_mask*current_out

            ori_masks.append(c_mask)

        return ori_masks

class importance_generator_ip(nn.Module):
    def __init__(self, structure=None, wn_flag=False, score_function='sigmoid', learnable_input=False):
        super(importance_generator_ip, self).__init__()

        self.bn1 = nn.LayerNorm([256])

        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)

        self.h0 = torch.zeros(2, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad = False

        if not learnable_input:
            self.inputs.requires_grad = False

        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(256, structure[i], bias=False)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(256, structure[i], bias=False, ) for i
                                in range(len(structure))]

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
                    importance = 0.5*torch.sigmoid(im_output + g  + self.base).squeeze() + 0.5* (self.fn_list[i]/self.fn_list[i].max())
                else:
                    score = self.score_function(im_output)
                    importance = 0.5 * score/score.max() + 0.5 * (
                                self.fn_list[i] / self.fn_list[i].max())
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



class simple_gate(nn.Module):
    def __init__(self, width):
        super(simple_gate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight

class simplified_imp(nn.Module):
    def __init__(self, structure=None, wn_flag=False, arch='gate'):
        super(simplified_imp, self).__init__()
        self.structure = structure

        if arch == 'gate':
            self.p_list = nn.ModuleList([simple_gate(structure[i]) for i in range(len(structure))])
        elif arch == 'fc':
            self.inputs = nn.Parameter(torch.Tensor(len(structure), 64))
            nn.init.orthogonal_(self.inputs)
            self.inputs.requires_grad = False
            if wn_flag:
                self.p_list = nn.ModuleList([weight_norm(nn.Linear(64, structure[i])) for i in range(len(structure))])
            else:
                self.p_list = nn.ModuleList([nn.Linear(64, structure[i]) for i in range(len(structure))])
        else:
            raise NotImplementedError
        self.arch = arch


        self.r_list = [torch.zeros(self.structure[i]) for i in range(len(self.structure))]
        self.r_lr = 0.995

    def set_fn_list(self, fn_list):
        self.fn_list = fn_list
        print(self.fn_list[0].max())
        print(self.fn_list[0].min())

    def forward(self, k_masks):
        if self.arch == 'gate':
            outputs = [self.p_list[i]() for i in range(len(self.structure))]
        elif self.arch == 'fc':
            outputs = [self.p_list[i](self.inputs[i,:]) for i in range(len(self.structure))]

        ori_masks = []
        for i in range(len(outputs)):
            im_output = outputs[i]
            # [:,:-1]
            g = sample_gumbel(im_output.size())
            if outputs[i].is_cuda:
                g = g.cuda()
            if not self.training:
                g = 0

            if self.training:
                importance = torch.sigmoid(im_output).squeeze() * (self.fn_list[i] / self.fn_list[i].max())
                if im_output.is_cuda:
                    self.r_list[i] = self.r_list[i].cuda()
                self.r_list[i] = self.r_lr * self.r_list[i] + (1 - self.r_lr) * importance.detach()
            else:
                importance = self.r_list[i]
            _, index = torch.sort(importance, descending=True)

            c_mask = k_masks[i].gather(0, index.squeeze().argsort())

            if self.training:
                current_out = torch.sigmoid((im_output + g + 3.0) / 0.4).squeeze()
                c_mask = c_mask * current_out

            ori_masks.append(c_mask)

        return ori_masks

class Simplified_Gate(nn.Module):
    def __init__(self, structure=None, T=0.4, base = 3, ):
        super(Simplified_Gate, self).__init__()
        self.structure = structure
        self.T = T
        self.base = base

        self.p_list = nn.ModuleList([simple_gate(structure[i]) for i in range(len(structure))])
    def forward(self,):

        if self.training:
            outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in range(len(self.structure))]
        else:
            outputs = [hard_concrete(gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base)) for i in range(len(self.structure))]

        out = torch.cat(outputs, dim=0)
        # print(out.size())
        return out

    def resource_output(self):
        outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in
                   range(len(self.structure))]

        outputs = [hard_concrete(outputs[i]) for i in
                   range(len(self.structure))]

        out = torch.cat(outputs, dim=0)

        return out

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

    def cal_len(self, inputs):
        # width=[]
        arch_vector = self.transfrom_output(inputs)
        # print(type(arch_vector))
        width = [arch_vector[i].sum()/arch_vector[i].size(0) for i in range(len(arch_vector))]
        width = [width[i].cpu().item() for i in range(len(arch_vector))]
        return width


def transfrom_output(inputs, structure):
    arch_vector = []
    start = 0
    for i in range(len(structure)):
        end = start + structure[i]
        arch_vector.append(inputs[start:end])
        start = end

    return arch_vector

class k_static(nn.Module):
    def __init__(self, structure, width=0.5):
        super(k_static, self).__init__()


        self.num_layers = len(structure)
        self.structure = structure
        self.inputs = nn.Parameter(torch.ones(1, self.num_layers)*width)

        self.inputs.requires_grad = False
        self.soft_range = 0.05
    def forward(self,):
        masks = []
        # outputs = torch.sigmoid((self.fc_layers(self.inputs))+self.offset).squeeze()
        outputs = self.inputs.squeeze()
        # print(type(outputs))
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

class im_static(nn.Module):
    def __init__(self, fn_list, structure=None, scale=3):
        super(im_static, self).__init__()
        self.structure = structure
        self.fn_list = fn_list

        full_sn = torch.cat(self.fn_list)

        max_score = full_sn.max()
        min_score = full_sn.min()

        full_sn = (full_sn - min_score)/(max_score - min_score)
        full_sn = 2*(full_sn - 0.5)
        full_sn = scale * full_sn

        self.mean = full_sn.mean()
        print(self.mean)
        self.fn_list = self.transfrom_output(full_sn)

        for i in range(len(self.fn_list)):
            self.fn_list[i] = nn.Parameter(self.fn_list[i])

        self.fn_list = nn.ParameterList(self.fn_list)

        for i in range(len(self.fn_list)):
            self.fn_list[i].requires_grad = False

    def set_fn_list(self, fn_list):
        self.fn_list = fn_list
        print(self.fn_list[0].max())
        print(self.fn_list[0].min())
        # (self.fn_list[i] / self.fn_list[i].max())

    def forward(self, k_masks):
        outputs = self.fn_list
        ori_masks = []
        cout_lists = []

        for i in range(len(outputs)):
            im_output = outputs[i]

            _, index = torch.sort(im_output, descending=True)

            c_mask = k_masks[i].gather(0, index.squeeze().argsort())

            if self.training:

                current_out = torch.sigmoid((im_output - self.mean) / 0.7).squeeze()
                # c_mask = c_mask*current_out
                # if ipout:
                    # pre_discrete = current_out
                cout_lists.append(current_out)

            ori_masks.append(c_mask)

        if self.training:

            return ori_masks, cout_lists

        else:
            return ori_masks

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

# if __name__ == '__main__':
#     net = k_generator()
#     y = net()
#     print(y)
