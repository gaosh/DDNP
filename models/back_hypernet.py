import torchsort
from .pytorch_ops import soft_rank, soft_sort
from .gumbel_ops import my_gumbel_sinkhorn, my_matching
import torch
import torch.nn as nn
import torch.nn.functional as F

class importance_generator_sr(nn.Module):
    def __init__(self, structure=None, wn_flag=True):
        super(importance_generator_sr, self).__init__()

        self.bn1 = nn.LayerNorm([256])

        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128,bidirectional=True)

        self.h0 = torch.zeros(2,1,128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure),1, 64))
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad=False


        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(256, structure[i], bias=False)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(256, structure[i], bias=False,) for i
                                in range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(self.linear_list)

    def forward(self, k_masks):
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i,:])) for i in  range(len(self.structure))]


        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        ori_masks = []
        for i in range(len(outputs)):
            # g = sample_gumbel(outputs[i].size())
            # # print(g.size())
            # if outputs[i].is_cuda:
            #     g = g.cuda()
            # current_out = torch.exp(outputs[i])
            current_out = torch.sigmoid(outputs[i])
            # print(c_mask)
            if self.training:
                # print(current_out)
                _, index = torch.sort(current_out, descending=True)

                c_mask = k_masks[i].gather(0, index.squeeze().argsort())

                # if current_out.size(1) == 16:
                #     s = 0.02
                # elif current_out.size(1) == 32:
                #     s = 0.01
                # elif current_out.size(1) == 64:
                #     s = 0.001
                soft_r = torchsort.soft_rank(current_out.cpu(), regularization_strength=0.005)
                # print(current_out)
                # print(soft_r)
                # soft_r = c_mask.size(0) - soft_r
                soft_mask = soft_indexing(c_mask, soft_r.squeeze().cuda()-1.0)
                # print(soft_mask)
                c_mask = soft_mask.gather(0, index.squeeze().argsort())
            else:
                _, index = torch.sort(current_out, descending=True)
                c_mask = k_masks[i].gather(0, index.squeeze().argsort())

                soft_r = torchsort.soft_rank(current_out.cpu(), regularization_strength=1e-5)

                soft_mask = soft_indexing(c_mask, soft_r.squeeze().cuda()-1.0, hard=True)

                c_mask = soft_mask.gather(0, index.squeeze().argsort())
                # print(c_mask)
            # print(c_mask)
            ori_masks.append(c_mask)
        # out = torch.cat(outputs, dim=1)


        return ori_masks

def soft_indexing(x, index, hard=False):
    # soft_x = torch.zeros(x.size())
    x_size = x.size(0)
    # if x.is_cuda:
    #     soft_x = x.cuda()
    # print(x.size())
    # print(index.size())
    soft_list = []
    # print(index)
    # print(x)
    for i in range(x.size(0)):

        x_range = torch.arange(x_size)
        i_lt = torch.LongTensor([i])
        if x.is_cuda:
            x_range = x_range.cuda()
            i_lt = i_lt.cuda()

        c_idx = torch.index_select(index, 0, i_lt)
        num = torch.exp(-(x_range-c_idx).pow(2))

        den = num.sum().detach()
        # print(num/den)
        c_v = ((num/den)*x).sum()
        soft_list.append(c_v)
        # if i == 0:
        #     soft_x = c_v
        # else:
        #     soft_x = torch.cat([soft_x, c_v])
    soft_x = torch.stack(soft_list)

    if hard:
        out_hard = torch.zeros(soft_x.size())
        out_hard[soft_x >= 0.5] = 1
        out_hard[soft_x < 0.5] = 0
        if soft_x.is_cuda:
            out_hard = out_hard.cuda()
        soft_x = (soft_x - out_hard).detach() + soft_x
    # print(num)
    # print(den)
    # print(soft_list[0])
    # print(soft_x)
    return soft_x

class permutation_generator(nn.Module):
    def __init__(self, structure=None,):
        super(permutation_generator, self).__init__()
        self.structure = structure
        # torch.eye(self.structure[i])
        # torch.randn(self.structure[i], self.structure[i]).abs()
        params_list = [nn.Parameter(torch.randn(self.structure[i], self.structure[i]).abs()) for i in range(len(self.structure))]
        self.params = nn.ParameterList(params_list)

    def forward(self, k_masks):
        perm_mats = self.actual_permutation()
        # print(perm_mats[0].size())
        # print(k_masks[0].size())
        p_kmasks = [torch.matmul(perm_mats[i], k_masks[i]) for i in range(len(self.structure))]
        # print(p_kmasks[0].size())
        return p_kmasks

    def projection(self):
        for i in range(len(self.structure)):
            # c_perm

            c_perm = F.relu(self.params[i].data)
            c_perm = c_perm / c_perm.sum(0, keepdim=True).detach()
            c_perm = c_perm / c_perm.sum(1, keepdim=True).detach()
            self.params[i].data = c_perm

    def actual_permutation(self):
        self.actual_perms = []
        for i in range(len(self.structure)):
            if not self.training:
                c_perm = self.params[i]
                # c_perm = torch.round(c_perm)
                max_idx = torch.argmax(c_perm, 0, keepdim=True)
                one_hot = torch.FloatTensor(c_perm.shape)
                one_hot.zero_()
                if max_idx.is_cuda:
                    one_hot = one_hot.cuda()

                one_hot.scatter_(0, max_idx, 1)

                c_perm = one_hot
            else:
                c_perm = self.params[i]

            self.actual_perms.append(c_perm)
        return self.actual_perms

    def regularization(self):
        losses = []
        for i in range(len(self.structure)):
            c_perm = self.params[i]
            c_loss = c_perm.abs().sum(0) - (c_perm.pow(2).sum(0)).sqrt()
            c_loss = c_loss.sum()
            r_loss = c_perm.abs().sum(1) - (c_perm.pow(2).sum(1)).sqrt()
            r_loss = r_loss.sum()

            losses.append(c_loss+r_loss)

        return torch.sum(torch.stack(losses))

class gumbel_matching(nn.Module):
    def __init__(self, structure=None,):
        super(gumbel_matching, self).__init__()
        self.structure = structure
        # torch.eye(self.structure[i])
        # torch.randn(self.structure[i], self.structure[i]).abs()
        params_list = [nn.Parameter(3*torch.eye(self.structure[i])) for i in range(len(self.structure))]
        self.params = nn.ParameterList(params_list)

    def forward(self, k_masks):
        perm_mats = self.actual_permutation()
        # print(perm_mats[0].size())
        # print(k_masks[0].size())
        p_kmasks = [torch.matmul(perm_mats[i], k_masks[i]) for i in range(len(self.structure))]
        # print(p_kmasks[0].size())
        return p_kmasks

    def actual_permutation(self):
        self.actual_perms = []
        for i in range(len(self.structure)):

            log_alpha = self.params[i]

            soft_perms_inf, log_alpha_w_noise = my_gumbel_sinkhorn(log_alpha.unsqueeze(0))
            if not self.training:
                # log_alpha_w_noise_flat = torch.transpose(log_alpha_w_noise, 0, 1)
                # n_numbers = log_alpha_w_noise.size(-1)
                # log_alpha_w_noise_flat = log_alpha_w_noise.view(-1, n_numbers, n_numbers)
                #
                # hard_perms_inf = my_matching(log_alpha_w_noise_flat)
                # perm = hard_perms_inf.float()
                # if log_alpha_w_noise.is_cuda:
                #     perm = perm.cuda()
                c_perm = self.params[i]
                # c_perm = torch.round(c_perm)
                max_idx = torch.argmax(c_perm, 0, keepdim=True)
                one_hot = torch.FloatTensor(c_perm.shape)
                one_hot.zero_()
                if max_idx.is_cuda:
                    one_hot = one_hot.cuda()

                one_hot.scatter_(0, max_idx, 1)

                perm = one_hot
            else:
                perm = soft_perms_inf

            self.actual_perms.append(perm.squeeze())

        return self.actual_perms
