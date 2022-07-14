from tqdm import tqdm
import torch
import os
from utils import display_structure, loss_fn_kd, loss_label_smoothing, display_factor, display_structure_hyper, LabelSmoothingLoss
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sampler import ImbalancedAccuracySampler
import torch.nn as nn

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def train_topk_dc(epoch, net, gens, train_loader, optimizers, res_constraint, args):
    tqdm_loader = tqdm(train_loader)
    net.eval()
    gens['k'].train()
    gens['im'].train()

    train_loss = 0
    resource_loss = 0
    p_losses = 0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        k_masks, k = gens['k']()
        ori_masks = gens['im'](k_masks, detach=True)

        net.set_vritual_gate(ori_masks)

        outputs = net(inputs)

        c_loss = criterion(outputs, targets)
        res_loss = 2 * res_constraint(k)

        loss = c_loss + res_loss

        optimizers['k'].zero_grad()
        loss.backward()
        optimizers['k'].step()

        rand_masks = gens['im'].random_forward()
        net.set_vritual_gate(rand_masks)
        rand_outputs = net(inputs)
        cr_loss = criterion(rand_outputs, targets)

        optimizers['k'].zero_grad()
        cr_loss.backward()
        optimizers['k'].step()

        with torch.no_grad():
            _, predicted = outputs.detach().max(1)
            local_correct = predicted.eq(targets).sum()

        total += targets.size(0)
        train_loss += c_loss.item()
        resource_loss += res_loss.item()

        correct += local_correct.item()

    print(
        ' * Epoch{epoch: d} Loss {loss:.3f} Res Loss {resloss: .3f} Acc@1 {top1:.3f}'
            .format(epoch=epoch, loss=train_loss / len(train_loader), resloss=resource_loss / len(train_loader),
                    top1=correct / total))

def train_topk(epoch, net, gens, train_loader, optimizers, res_constraint, args, txt_name=None):
    tqdm_loader = tqdm(train_loader)
    net.eval()
    gens['k'].train()
    gens['im'].train()

    train_loss = 0
    resource_loss = 0
    p_losses = 0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        k_masks, k = gens['k']()

        if args.ip:
            ori_masks, ip_list = gens['im'](k_masks, ipout=True)
        else:
            ori_masks = gens['im'](k_masks)

        net.set_vritual_gate(ori_masks)

        outputs = net(inputs)

        c_loss = criterion(outputs, targets)
        if args.ip:
            res_loss = 2 * res_constraint(ip_list, ad_input = k)
            p_loss = pursuit_loss(ip_list,k)
            loss = c_loss + res_loss + p_loss

        else:
            res_loss = 2 * res_constraint(k)

        # pursuit_loss

            loss = c_loss + res_loss

        # if args.perm:
        #     p_loss = gens['im'].regularization()
        #     loss = loss +  args.p_lam * p_loss

        optimizers['im'].zero_grad()
        optimizers['k'].zero_grad()
        loss.backward()
        optimizers['im'].step()
        optimizers['k'].step()

        with torch.no_grad():
            _, predicted = outputs.detach().max(1)
            local_correct = predicted.eq(targets).sum()

        total += targets.size(0)
        train_loss += c_loss.item()
        resource_loss += res_loss.item()

        correct += local_correct.item()

        # if args.perm:
        #     p_losses += p_loss.item()
        #     gens['im'].projection()
    #
    # if args.perm:
    #     print(
    #         ' * Epoch{epoch: d} Loss {loss:.3f} Res Loss {resloss: .3f} P Loss {ploss: .3f} Acc@1 {top1:.3f}'
    #             .format(epoch=epoch, loss=train_loss / len(train_loader), resloss=resource_loss / len(train_loader), ploss=p_losses/len(train_loader),
    #                     top1=correct / total))

    # else:
    print(
        ' * Epoch{epoch: d} Loss {loss:.3f} Res Loss {resloss: .3f} Acc@1 {top1:.3f}'
            .format(epoch=epoch, loss=train_loss / len(train_loader), resloss=resource_loss / len(train_loader), top1=correct / total))

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(train_loss / len(tqdm_loader)) + ' ' +  str(resource_loss / len(tqdm_loader))

        file_txt.write(contents + ' \n')
        file_txt.close()

def train_topk_simple(epoch, net, gens, train_loader, optimizer, args, res_constraint=None, txt_name=None):
    tqdm_loader = tqdm(train_loader)
    net.eval()
    gens['k'].train()
    gens['im'].train()
    # if 'k' in gen:
    #     gen['k'].train()
    # elif 'im' in gen:
    #     gen['im'].train()

    train_loss = 0
    resource_loss = 0
    p_losses = 0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # if args.k_flag:
        k_masks, k = gens['k']()

        # if args.im_only:
        if args.setting == 'im_only':

            ori_masks, ip_list = gens['im'](k_masks,ipout=True)
        else:
            ori_masks, ip_list = gens['im'](k_masks,)
        # elif args.k_only:
        #     ori_masks = gens['im'](k_masks)

        net.set_vritual_gate(ori_masks)
        outputs = net(inputs)
        if args.setting == 'im_only':
            # net.set_vritual_gate(ori_masks)
            # outputs = net(inputs)
            c_loss = criterion(outputs, targets)

            p_loss = pursuit_loss(ip_list,k)
            loss = c_loss + p_loss # + p_loss
            res_loss = torch.Tensor([0])
        elif args.setting == 'k_only':
            p_loss = pursuit_loss_mae(ip_list, k)
            res_loss = 2 * res_constraint(k)

            loss = res_loss + p_loss
            c_loss = torch.Tensor([0])
        optimizer.zero_grad()
        # optimizers['k'].zero_grad()
        loss.backward()
        # optimizers['im'].step()
        optimizer.step()

        with torch.no_grad():
            _, predicted = outputs.detach().max(1)
            local_correct = predicted.eq(targets).sum()

        total += targets.size(0)
        train_loss += c_loss.item()
        resource_loss += res_loss.item()

        correct += local_correct.item()

    print(
        ' * Epoch{epoch: d} Loss {loss:.3f} Res Loss {resloss: .3f} Acc@1 {top1:.3f}'
            .format(epoch=epoch, loss=train_loss / len(train_loader), resloss=resource_loss / len(train_loader),
                    top1=correct / total))

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(train_loss / len(tqdm_loader)) + ' ' + str(resource_loss / len(tqdm_loader))

        file_txt.write(contents + ' \n')
        file_txt.close()


def retrain(epoch, net, criterion,trainloader, optimizer, smooth=True, scheduler=None, alpha=0.5):
    #net.activate_weights()
    #net.set_training_flag(False)
    tqdm_loader = tqdm(trainloader)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = alpha

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        if scheduler is not None:
            scheduler.step()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        if smooth:
            loss_smooth = LabelSmoothingLoss(classes=10,smoothing=0.1)(outputs, targets)
            loss_c = criterion(outputs, targets)
            loss = alpha*loss_smooth + (1-alpha)*loss_c
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch, train_loss / len(trainloader), 100. * correct / total, correct, total))

def valid(epoch, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model', txt_name=None):
    txtdir = './txt/'
    if stage == 'valid_model':

        tqdm_loader = tqdm(testloader)
    elif stage == 'valid_gate':
        #net.foreze_weights()
        if hyper_net is None:
            net.set_training_flag(True)
        tqdm_loader = testloader
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    if hyper_net is not None:
        hyper_net.eval()
        vector = hyper_net()
        # print(vector)
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if hyper_net is not None:
                net.set_vritual_gate(vector)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    is_best=False
    if hyper_net is not None:
        if epoch>100:
            if acc > best_acc:
                best_acc = acc
                is_best=True
        else:
            best_acc = 0
    else:
        if acc>best_acc:
            best_acc = acc
            is_best = True
    if model_string is not None:

        if is_best:
            print('Saving..')
            if hyper_net is not None:

                state = {
                    'net': net.state_dict(),
                    'hyper_net': hyper_net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector':vector
                    #'gpinfo':gplist,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    # 'gpinfo':gplist,
                }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/%s.pth.tar'%(model_string))

    print( 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
                    % (test_loss/len(testloader), 100.*correct/total, correct, total, best_acc))

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(correct/total) + ' ' + str(test_loss / len(testloader))

        file_txt.write(contents + ' \n')
        file_txt.close()


    return best_acc

def valid_topk(epoch, net, testloader, best_acc, gens=None, model_string=None,txt_name=None):
    # txtdir = './txt/'


    tqdm_loader = tqdm(testloader)

    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    if gens is not None:
        gens['k'].eval()
        gens['im'].eval()
        k_masks, k = gens['k']()
        ori_masks = gens['im'](k_masks)
        net.set_vritual_gate(ori_masks)
        # print(vector)
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    is_best=False
    if gens is not None:
        if epoch>100:
            if acc > best_acc:
                best_acc = acc
                is_best=True
        else:
            best_acc = 0
    else:
        if acc>best_acc:
            best_acc = acc
            is_best = True
    if model_string is not None:

        if is_best:
            print('Saving..')
            if gens is not None:

                state = {
                    'net': net.state_dict(),
                    'gen_im': gens['im'].state_dict(),
                    'gen_k': gens['k'].state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector':ori_masks
                    #'gpinfo':gplist,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    # 'gpinfo':gplist,
                }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/%s.pth.tar'%(model_string))

    print( 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
                    % (test_loss/len(testloader), 100.*correct/total, correct, total, best_acc))

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(correct/total) + ' ' + str(test_loss / len(testloader))

        file_txt.write(contents + ' \n')
        file_txt.close()

    return best_acc

def collect_fn(net, loader):
    tqdm_loader = tqdm(loader)
    net.enable_fn()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            net(inputs)

    return net.final_fn()


def add_noise_to_weights(model, alpha):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):

                pertub = torch.randn(m.weight.size())
                if m.weight.data.is_cuda:
                    pertub = pertub.cuda()

                w_n = m.weight.norm(p=2)
                p_n = pertub.norm(p=2)

                p = (pertub/p_n)*w_n

                m.weight.add_(p*alpha)

            if isinstance(m, nn.Conv2d):
                pertub = torch.randn(m.weight.size())
                if m.weight.data.is_cuda:
                    pertub = pertub.cuda()

                w_n = m.weight.norm(p=2)
                p_n = pertub.norm(p=2)

                p = (pertub / p_n) * w_n

                m.weight.add_(p * alpha)

def train_smooth(epoch, net, trainloader, optimizer, hyper_net, resource_constraint, args, txt_name):
    train_loss = 0
    resource_loss = 0

    correct = 0
    total = 0
    tqdm_loader = tqdm(trainloader)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        vector = hyper_net()
        # print(args.w_pertub)
        if args.w_pertub:

            copy_net = copy.deepcopy(net)
            copy_net.set_vritual_gate(vector)
            add_noise_to_weights(copy_net, alpha=args.alpha)
        else:

            copy_net = net
            copy_net.set_vritual_gate(vector)
        outputs = copy_net(inputs)
        res_loss = 2 * resource_constraint(hyper_net.resource_output())

        loss = criterion(outputs, targets) + res_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        resource_loss += res_loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(train_loss / len(tqdm_loader)) + ' ' +  str(resource_loss / len(tqdm_loader))

        file_txt.write(contents + ' \n')
        file_txt.close()

    print('Epoch: %d Loss: %.3f Res-Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (epoch, train_loss / len(trainloader), resource_loss / len(trainloader), 100. * correct / total, correct, total))



def pursuit_loss(ipout, k_vector, w=1, d_k=True, d_i=False, m=0.02):
    sum_loss = 0
    # resource_ratio = (sum_flops / self.t_flops)
    # abs_rv = torch.clamp(resource_ratio, min=self.p)
    # loss = torch.log((abs_rv / (self.p)))

    if d_i:
        k_vector = k_vector.detach()
    for i in range(len(ipout)):
        r = (torch.rand(1) - 0.5) * 2 * m
        r = r.to(k_vector.device)
        if d_k:
            ipout[i] = ipout[i].detach()


        diff = ipout[i].sum()/ipout[i].size(0)-(k_vector[i]) + r
        sum_loss = sum_loss + torch.log(diff.abs()+1)

    return w*sum_loss

def pursuit_loss_mae(ipout, k_vector, w=1, d_k=True, d_i=False, m=0.02):
    sum_loss = 0
    # resource_ratio = (sum_flops / self.t_flops)
    # abs_rv = torch.clamp(resource_ratio, min=self.p)
    # loss = torch.log((abs_rv / (self.p)))

    if d_i:
        k_vector = k_vector.detach()
    for i in range(len(ipout)):
        r = (torch.rand(1) - 0.5) * 2 * m
        r = r.to(k_vector.device)
        if d_k:
            ipout[i] = ipout[i].detach()


        diff = ipout[i].sum()/ipout[i].size(0)-(k_vector[i]) + r
        sum_loss = sum_loss + diff.abs()

    return w*sum_loss