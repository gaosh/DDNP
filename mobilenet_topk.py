from train import *
from utils import *

from models.mobilenetv2_hyper import MobileNetV2
from models.hypernet import k_generator, importance_generator, permutation_generator, gumbel_matching
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
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--stage', default='train-gate', type=str)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--reg_w', default=2, type=float)
parser.add_argument('--perm', default=False, type=str2bool)
# args.perm
parser.add_argument('--p_lam', default=0.1, type=float)
parser.add_argument('--loss', default='log', type=str)

args = parser.parse_args()
depth = args.depth
model_name = 'mobnet'

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
validloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=4,sampler=val_sampler)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

gens = {}

net = MobileNetV2()
width, structure = net.count_structure()

gens['k'] = k_generator(structure)
if args.perm:
    gens['im'] = permutation_generator(structure)
else:
    gens['im'] = importance_generator(structure)

gens['k'] = k_generator(structure)

gens['im'] = importance_generator(structure)
    # gens['im'] = gumbel_matching(structure)

stat_dict = torch.load('./checkpoint/%s-base.pth.tar' % (model_name))
net.load_state_dict(stat_dict['net'])
net.foreze_weights()

size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(net)

res_reg = Flops_constraint_mobnet_k(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                          w=args.reg_w, structure=structure,)

Epoch = args.epoch

gens['k'].cuda()
gens['im'].cuda()
net.cuda()

optimizers = {}
schedulers = {}
# optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
optimizers['im'] = optim.AdamW(gens['im'].parameters(), lr=1e-3, weight_decay=1e-4)
optimizers['k'] = optim.AdamW(gens['k'].parameters(), lr=1e-3, weight_decay=0)
schedulers['im'] = MultiStepLR(optimizers['im'], milestones=[int(Epoch*1.0)], gamma=0.1)
schedulers['k']  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['k'], int(Epoch*0.5), eta_min=0)
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['im'], int(Epoch), eta_min=0)
best_acc = 0
fn_list = collect_fn(net, trainloader)

gens['im'].set_fn_list(fn_list)
valid(0, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model',)
torch.set_printoptions(precision=3)
for epoch in range(Epoch):
    train_topk(epoch, net, gens, validloader, optimizers, res_reg, args)

    schedulers['k'].step()
    schedulers['im'].step()

    best_acc = valid_topk(epoch, net, testloader, best_acc, gens, model_string='%s-pruned'%(model_name))
    with torch.no_grad():
        _,outputs = gens['k']()
        print(outputs)
