import os
import json
import random
import copy

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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


@dispatcher.register('apricot')
def apricot(opt, model, device):
    guard_folder(opt, folder='apricot')

    trainset, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    # create rDLMs
    NUM_SUBMODELS = 20
    SUBSET_SIZE = 10000
    SUBMODEL_EPOCHS = 40
    subset_step = int((len(trainset) - SUBSET_SIZE) // NUM_SUBMODELS)

    for sub_idx in tqdm(range(NUM_SUBMODELS), desc='rDLMs'):
        submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
        if os.path.exists(submodel_path):
            continue

        subset_indices = list(range(subset_step * sub_idx, subset_step * sub_idx + SUBSET_SIZE))
        subset = torch.utils.data.Subset(trainset, subset_indices)
        subloader = torch.utils.data.DataLoader(
            subset, batch_size=opt.batch_size, shuffle=True, num_workers=4
        )

        submodel = copy.deepcopy(model).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            submodel.parameters(),
            lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_acc, *_ = test(submodel, valloader, criterion, device, desc='Baseline')
        for epoch in tqdm(range(0, SUBMODEL_EPOCHS), desc='epochs'):
            train(submodel, subloader, optimizer, criterion, device)
            acc, *_ = test(submodel, valloader, criterion, device)
            if acc > best_acc:
                print('Saving...')
                state = {
                    'cepoch': epoch,
                    'net': submodel.state_dict(),
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict(),
                    'acc': acc
                }
                torch.save(state, submodel_path)
                best_acc = acc
            scheduler.step()
        print('[info] the best submodel accuracy is {:.4f}%'.format(best_acc))

    # submodels
    sm_equals_path = get_model_path(opt, folder='apricot', state='sub_pred_equals')

    if not os.path.exists(sm_equals_path):
        submodel = copy.deepcopy(model).to(device).eval()
        seqloader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
        )
        submodels_equals = []
        for sub_idx in tqdm(range(NUM_SUBMODELS), desc='subModelPreds'):
            submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
            state = torch.load(submodel_path)
            submodel.load_state_dict(state['net'])

            equals = []
            for inputs, targets in tqdm(seqloader, desc='Batch', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = submodel(inputs)
                _, predicted = outputs.max(1)
                eqs = predicted.eq(targets).flatten()
                equals.append(eqs)
            equals = torch.cat(equals)

            submodels_equals.append(equals)
        submodels_equals = torch.stack(submodels_equals)
        torch.save(submodels_equals, sm_equals_path)
    else:
        submodels_equals = torch.load(sm_equals_path)

    # Fixing process
    NUM_LOOP_COUNT = 3
    BATCH_SIZE = 20
    LEARNING_RATE = 0.001

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    best_acc, *_ = test(model, valloader, criterion, device)
    best_weights = copy.deepcopy(model.state_dict())

    submodel_weights = []
    for sub_idx in range(NUM_SUBMODELS):
        submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
        state = torch.load(submodel_path)
        submodel_weights.append(state['net'])

    for loop_idx in range(NUM_LOOP_COUNT):
        print('Loop: {}'.format(loop_idx))
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(range(len(trainset))),
            batch_size=BATCH_SIZE, drop_last=False
        )
        for indices in tqdm(sampler, desc='Sampler'):
            model.eval()
            base_weights = copy.deepcopy(model.state_dict())

            inputs = torch.stack([trainset[ind][0] for ind in indices]).to(device)
            targets = torch.tensor([trainset[ind][1] for ind in indices]).to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                equals = predicted.eq(targets).flatten()

            for ind, equal in zip(indices, equals):
                if equal:
                    continue
                correct_submodels = [
                    submodel_weights[i] for i in range(NUM_SUBMODELS)
                    if submodels_equals[i][ind]
                ]
                if len(correct_submodels) == 0:
                    continue

                # use strategy 2
                for key in base_weights.keys():
                    if 'num_batches_tracked' in key:
                        continue
                    correct_weight = torch.mean(
                        torch.stack([m[key] for m in correct_submodels]), dim=0)
                    correct_diff = base_weights[key] - correct_weight
                    p_corr = len(correct_submodels) / NUM_SUBMODELS  # proportion
                    base_weights[key] = base_weights[key] + LEARNING_RATE * p_corr * correct_diff

            model.load_state_dict(base_weights)
            acc, *_ = test(model, valloader, criterion, device, desc='Eval')
            if acc > best_acc:
                best_weights = copy.deepcopy(base_weights)
                torch.save({'net': best_weights}, get_model_path(opt, state='apricot'))
                best_acc = acc

            train(model, trainloader, optimizer, criterion, device)
            acc, *_ = test(model, valloader, criterion, device, desc='Eval')
            if acc > best_acc:
                best_weights = copy.deepcopy(base_weights)
                torch.save({'net': best_weights}, get_model_path(opt, state='apricot'))
                best_acc = acc

            model.load_state_dict(best_weights)

        # violate the original
        break


@dispatcher.register('robot')
def robot(opt, model, device):
    trainset, _ = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    # compute FOL
    ROBOT_ELLIPSIS = 0.01

    seqloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
    )
    fols = []
    for inputs, targets in tqdm(seqloader, desc='FOL'):
        cur_batch_size = targets.size(0)

        with torch.enable_grad():
            grad_inputs = inputs.clone().detach()
            grad_inputs, targets = grad_inputs.to(device), targets.to(device)
            grad_inputs.requires_grad = True
            model.zero_grad()
            outputs = model(grad_inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        grads_flat = grad_inputs.grad.cpu().numpy().reshape(cur_batch_size, -1)
        grads_norm = np.linalg.norm(grads_flat, ord=1, axis=1)
        grads_diff = grads_flat - inputs.numpy().reshape(cur_batch_size, -1)
        i_fols = -1. * (grads_flat * grads_diff).sum(axis=1) + ROBOT_ELLIPSIS * grads_norm
        fols.append(i_fols)
    fols = np.concatenate(fols)

    SELECTED_NUM = 10000
    indices = np.argsort(fols)
    sel_indices = np.concatenate((indices[:SELECTED_NUM//2], indices[-SELECTED_NUM//2:]))
    print(sel_indices)
    seqloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, sel_indices.tolist()),
        batch_size=opt.batch_size, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    RETRAIN_EPOCHS = 40
    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, RETRAIN_EPOCHS):
        print('Epoch: {}'.format(epoch))
        train(model, seqloader, optimizer, criterion, device)
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
            torch.save(state, get_model_path(opt, state='robot'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('gini')
def deepgini(opt, model, device):
    trainset, _ = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    # compute gini index
    seqloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
    )
    ginis = []
    for inputs, targets in tqdm(seqloader, desc='Gini'):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        probs = F.softmax(outputs, dim=1)
        gini = probs.square().sum(dim=1).mul(-1.).add(1.)
        ginis.append(gini.detach().cpu())
    ginis = torch.cat(ginis)

    indices = torch.argsort(ginis, descending=True)
    train_size = int(len(trainset) * 0.1)
    sel_indices = indices[:train_size]
    seqloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, sel_indices.tolist()),
        batch_size=opt.batch_size, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    RETRAIN_EPOCHS = 40
    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, RETRAIN_EPOCHS):
        print('Epoch: {}'.format(epoch))
        train(model, seqloader, optimizer, criterion, device)
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
            torch.save(state, get_model_path(opt, state='gini'))
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

