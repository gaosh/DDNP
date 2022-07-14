from train import *
from utils import *

from models.resnet_hyper import ResNet as ResNet_hyper
from models.hypernet import k_generator, importance_generator, im_static, importance_generator_ip, simple_k
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--stage', default='train-gate', type=str)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--reg_w', default=1, type=float)
# parser.add_argument('--perm', default=False, type=str2bool)
# # args.perm
# parser.add_argument('--p_lam', default=0.1, type=float)
parser.add_argument('--loss', default='log', type=str)
parser.add_argument('--discrete', default=False, type=str2bool)
parser.add_argument('--ip', default=False, type=str2bool)
parser.add_argument('--s_fun', default='sigmoid', type=str)
parser.add_argument('--im_type', default='im_net', type=str)
parser.add_argument('--k_type', default='k_net', type=str)
parser.add_argument('--learnable_input', default=False, type=str2bool)

parser.add_argument('--record_arch', default=False, type=str2bool)
args = parser.parse_args()
depth = args.depth
model_name = 'resnet'

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=True, transform=transform_test)
train_sampler,val_sampler = TrainVal_split(trainset, 0.05, shuffle_dataset=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2,shuffle=True)
validloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=4,sampler=val_sampler)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

gens = {}

net = ResNet_hyper(depth=depth, gate_flag=True)
width, structure = net.count_structure()

if args.discrete:
    if args.k_type == 'k_net':

        gens['k'] = k_generator(structure, soft_range=0)
    else:
        gens['k'] = simple_k(structure, soft_range=0)


else:
    if args.k_type == 'k_net':

        gens['k'] = k_generator(structure,)
    else:
        gens['k'] = simple_k(structure,)

if args.im_type == 'im_net':
    if args.ip:
        gens['im'] = importance_generator_ip(structure, learnable_input=args.learnable_input)
    else:
        gens['im'] = importance_generator(structure, score_function = args.s_fun, learnable_input=args.learnable_input)
else:
    gens['im'] = simplified_imp(structure, arch=args.im_type)
    # gens['im'] = gumbel_matching(structure)


if args.discrete:
    dis_str = 'd_'
else:
    dis_str = 's_'

if args.learnable_input:
    li_str = 'li_'
else:
    li_str = ''

if args.ip:
    ip_str = '_ip'
else:
    ip_str = ''
# if args.c_loss:
#     c_str = 'c_'
#     if args.c_only:
#         c_str = 'o' + c_str
# else:
#     c_str = ''

train_name = './analysis/rn56%s_'%(args.p)  + args.im_type + '_' + args.k_type + '_' + dis_str + str(args.reg_w) + '_' + li_str + args.s_fun + ip_str + '.txt'
test_name = './analysis/rn56%stest_'%(args.p) + args.im_type + '_' + args.k_type + '_' + dis_str + str(args.reg_w) + '_' + li_str + args.s_fun + ip_str + '.txt'

stat_dict = torch.load('./checkpoint/%s-base.pth.tar'%(model_name+str(depth)))
net.load_state_dict(stat_dict['net'])
net.foreze_weights()



size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(net)

res_reg = Flops_constraint_resnet_k(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                          w=args.reg_w, structure=structure, loss_type=args.loss, ar_v = args.ip)

Epoch = args.epoch

gens['k'].cuda()
gens['im'].cuda()
net.cuda()


# params = list(gens['k'].parameters()) + list(gens['im'].parameters())
optimizers = {}
schedulers = {}

# k_lr = 1e-3
if args.k_type != 'k_net':
    k_lr = 1e-2
else:
    k_lr = 1e-3
# optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
optimizers['im'] = optim.AdamW(gens['im'].parameters(), lr=1e-3, weight_decay=1e-4)
optimizers['k'] = optim.AdamW(gens['k'].parameters(), lr=k_lr, weight_decay=0)
schedulers['im'] = MultiStepLR(optimizers['im'], milestones=[int(Epoch*1.0)], gamma=0.1)
# schedulers['k']  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['k'], int(Epoch), eta_min=0)
# if args.ip:
#     schedulers['k'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['k'], int(Epoch), eta_min=1e-5)
# else:
schedulers['k'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['k'], int(Epoch), eta_min=0)
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['im'], int(Epoch), eta_min=0)
best_acc = 0
fn_list = collect_fn(net, trainloader)

gens['im'].set_fn_list(fn_list)

txt_name = './analysis/arch_rn56%s_'%(args.p) + args.im_type + '.txt'
valid(0, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model',)
torch.set_printoptions(precision=3)
for epoch in range(Epoch):
    train_topk(epoch, net, gens, validloader, optimizers, res_reg, args,txt_name=train_name)
    # train_topk_dc(epoch, net, gens, validloader, optimizers, res_reg, args)
    schedulers['k'].step()
    schedulers['im'].step()
    # print(gens['im'].actual_perms[0])
    # valid_topk(epoch, net, testloader, best_acc, gens=None, model_string=None, )
    best_acc = valid_topk(epoch, net, testloader, best_acc, gens, model_string='%s-pruned'%(model_name+str(depth)),txt_name=test_name)
    with torch.no_grad():
        _,outputs = gens['k']()
        print(outputs)
        if args.record_arch:
            file_txt = open(txt_name, 'a')
            outputs = outputs.squeeze().cpu().tolist()
            contents = ' '.join([str(i) for i in outputs])
            file_txt.write(contents + ' \n')
            file_txt.close()
