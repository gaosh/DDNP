import torch
import torch.nn as nn


def smoothstep(x, tau):
    #-tau/2 < x < tau/2
    out = -2*(x/tau).pow(3) + (3/2)*(x/tau) + 0.5
    return torch.clamp(out, min=0.0, max=1.0)


class k_gradient(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, k, size, soft_range, lb):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        topk_mask = torch.zeros(size,1).squeeze()
        if k.is_cuda:
            topk_mask = topk_mask.cuda()
        # print(soft_range)
        # print(k)
        # print(size)

        if soft_range + torch.round(k * size) >= size:
            soft_range = size - torch.round(k * size) - 1
            soft_range = int(soft_range.item())
        elif torch.round(k * size) - soft_range < 0:
            soft_range = torch.round(k * size) - 0
            soft_range = int(soft_range.item())
        # print(soft_range)
        if soft_range<=0:
            topk_mask[:torch.round(k * size).long()] = 1
        else:


            offset = k*size - torch.round(k*size)
            # print(offset.is_cuda)
            # print()
            # print(soft_range)
            x = torch.arange(-soft_range, soft_range+1)
            if k.is_cuda:
                offset = offset.cuda()
                x = x.cuda()
            x = x.float() + offset
            soft_masks = smoothstep(x, tau=2*soft_range)
            soft_masks = torch.flip(soft_masks, dims=[0])


            if torch.round(k * size) - soft_range > 0:
                topk_mask[:(torch.round(k*size)-soft_range).long()] = 1
            # print(soft_masks.size())
            # print(torch.round(k * size))
            # print(soft_range)
            # print(topk_mask[(torch.round(k*size)-soft_range).long():(torch.round(k*size)+soft_range + 1).long()].size())
            # print(soft_masks.size())
            # print(topk_mask[(torch.round(k * size) - soft_range).long():].size())
            topk_mask[(torch.round(k*size)-soft_range).long():(torch.round(k*size)+soft_range + 1).long()] = soft_masks


        # ctx.grad_w = grad_w
        topk_mask = topk_mask + lb
        topk_mask[topk_mask>=1] = 1
        return topk_mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        grad_input = grad_output.sum()/grad_output.size(0)
        return grad_input, None, None, None

class mask_gradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, importance, sorted_mask):
        return sorted_mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        grad_input_im = grad_output
        grad_input_km = grad_output
        return grad_input_im, grad_input_km