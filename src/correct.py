import os
import json
import random

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from model import load_model, resume_model
from dataset import load_dataset
from arguments import advparser as parser
from train import train, test
from utils import *


dispatcher = AttrDispatcher('crt_method')


class CorrectionUnit(nn.Module):
    def __init__(self, num_filters, Di, k):
        super(CorrectionUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, Di, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(Di)
        self.conv2 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(Di)
        self.conv3 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(Di)
        self.conv4 = nn.Conv2d(
            Di, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out += self.shortcut(x)
        return out


class ConcatCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ConcatCorrect, self).__init__()
        self.indices = indices
        self.others = [i for i in range(conv_layer.out_channels)
                       if i not in indices]
        self.conv = conv_layer
        num_filters = len(indices)
        self.cru = CorrectionUnit(num_filters, num_filters, 3)

    def forward(self, x):
        out = self.conv(x)
        out_lower = out[:, self.others]
        out_upper = self.cru(out[:, self.indices])
        out = torch.cat([out_lower, out_upper], dim=1)
        #  out[:, self.indices] = self.cru(out[:, self.indices])
        return out


class ReplaceCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ReplaceCorrect, self).__init__()
        self.indices = indices
        self.conv = conv_layer
        self.cru = nn.Conv2d(
            conv_layer.in_channels,
            len(indices),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            groups=conv_layer.groups,
            bias=False)

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = self.cru(x)
        return out


class NoneCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(NoneCorrect, self).__init__()
        self.indices = indices
        self.conv = conv_layer

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = 0
        return out


def construct_model(opt, model, patch=True):
    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, f'susp_filters_{opt.fs_method}.json'
    ))) if opt.susp_side in ('front', 'rear') else {}

    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for layer_name in conv_names:
        module = rgetattr(model, layer_name)

        num_susp = int(module.out_channels * opt.susp_ratio)
        if opt.susp_side == 'front':
            indices = sus_filters[layer_name]['indices'][:num_susp]
        elif opt.susp_side == 'rear':
            indices = sus_filters[layer_name]['indices'][-num_susp:]
        elif opt.susp_side == 'random':
            indices = random.sample(range(module.out_channels), num_susp)
        else:
            raise ValueError('Invalid suspicious side')

        if module.groups != 1:
            continue

        if patch is False:
            correct_module = NoneCorrect(module, indices)
        elif opt.pt_method == 'DC':
            correct_module = ConcatCorrect(module, indices)
        elif 'DP' in opt.pt_method:
            correct_module = ReplaceCorrect(module, indices)
        else:
            raise ValueError('Invalid correct type')
        rsetattr(model, layer_name, correct_module)
    return model


def extract_indices(model):
    info = {}
    for n, m in model.named_modules():
        if isinstance(m, ConcatCorrect) \
                or isinstance(m, ReplaceCorrect) \
                or isinstance(m, NoneCorrect):
            info[n] = m.indices
    return info


@dispatcher.register('patch')
def patch(opt, model, device):
    if opt.pt_method == 'DP-s':
        _, trainloader = load_dataset(opt, split='train', aug=True)
        _, valloader = load_dataset(opt, split='val', aug=True)
    elif opt.pt_method == 'DP-SS':
        trainset, _ = load_dataset(opt, split='train', aug=True)
        _, valloader = load_dataset(opt, split='val', aug=True)
    elif opt.pt_method == 'SS-DP':
        _, trainloader = load_dataset(opt, split='train')
        _, valloader = load_dataset(opt, split='val')
        model = resume_model(opt, model, state=f'sensei_base')
    else:
        _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
        _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = construct_model(opt, model)
    model = model.to(device)

    for name, module in model.named_modules():
        if 'cru' in name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    sel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    if opt.resume:
        ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['sched'])
        start_epoch = ckp['cepoch']
        best_acc = ckp['acc']
        for n, m in model.named_modules():
            if isinstance(m, ConcatCorrect) \
                    or isinstance(m, ReplaceCorrect) \
                    or isinstance(m, NoneCorrect):
                m.indices = ckp['indices'][n]
    else:
        best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')

    for epoch in range(start_epoch + 1, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        if opt.pt_method == 'DP-SS':
            trainset.selective_augment(model, sel_criterion, opt.batch_size, device)  # type: ignore
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4
            )
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc,
                'indices': extract_indices(model)
            }
            torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('finetune')
def finetune(opt, model, device):
    _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state=f'finetune_g{opt.gpu}'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('sensei')
def sensei(opt, model, device):
    trainset, _ = load_dataset(opt, split='train', aug=True)
    _, valloader = load_dataset(opt, split='val', aug=True)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    sel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        trainset.selective_augment(model, sel_criterion, opt.batch_size, device)  # type: ignore
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4
        )
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state=f'sensei_g{opt.gpu}'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    dispatcher(opt, model, device)


if __name__ == '__main__':
    main()

