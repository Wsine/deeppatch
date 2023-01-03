import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import load_model
from dataset import load_dataset
from arguments import advparser as parser
from train import test
from correct import construct_model, ReplaceCorrect
from utils import *


class AdvWrapper(nn.Module):
    def __init__(self, module):
        super(AdvWrapper, self).__init__()
        self.module = module
        self.boundary = 0

    def avg_diam_distance(self, x):
        v = x.view(-1).reshape(-1, 1)
        m = torch.triu(torch.matmul(v, v.t()), diagonal=1)
        d = m.sum().div(v.size(0) * (v.size(0) - 1))
        return d

    def neuron_coverage(self, x):
        nc = F.relu(x).gt(1e-6).sum().div(x.numel())
        return nc

    def diffenentiate_activation(self, x, y):
        batch, numel = x.view(x.size(0), -1).size()
        d = torch.logical_xor(F.relu(x).gt(0), F.relu(y).gt(0))
        nc = d.view(batch, numel).sum(1).div(numel)
        return nc

    def forward(self, x):
        global indicator
        if isinstance(self.module, ReplaceCorrect):
            out = self.module.conv(x)
            repl = self.module.cru(x)

            if indicator is None:
                out = (out, repl, self.module.indices)
            elif indicator is True:
                out[:, self.module.indices] = repl

        elif isinstance(self.module, nn.BatchNorm2d):
            if indicator is None:
                p_out, p_repl, p_indices = x
                out1 = self.std_bn(p_out)
                p_out[:, p_indices] = p_repl
                out2 = self.module(p_out)
                self.dnc = self.diffenentiate_activation(out1[:, p_indices], out2[:, p_indices])
                if self.dnc.lt(self.boundary).sum().div(p_out.size(0)).gt(0.5):
                    indicator = True
                    out = out2
                else:
                    indicator = False
                    out = out1
            elif indicator is True:
                out = self.module(x)
            else:
                out = self.std_bn(x)
        else:
            out = self.module(x)

        return out


def switch_on_the_fly(opt, model, device):
    backbone = copy.deepcopy(model)
    model = construct_model(opt, model, patch=True)

    # Resume
    ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}'))
    model.load_state_dict(ckp['net'])

    def _clean_indicator_hook(module, pinput):
        global indicator
        indicator = None

    # Patch
    first_repl = True
    for name, module in model.named_modules():
        if isinstance(module, ReplaceCorrect):
            new_module = AdvWrapper(module)
            if first_repl is True:
                new_module.register_forward_pre_hook(_clean_indicator_hook)
                first_repl = False
            rsetattr(model, name, new_module)
        elif isinstance(module, nn.BatchNorm2d):
            old_module = rgetattr(backbone, name)
            new_module = AdvWrapper(module)
            new_module.std_bn = old_module
            rsetattr(model, name, new_module)

    model = model.to(device)

    # Calibrate
    diff_list = []
    def _calib_hook(module, finput, foutput):
        diff_list.append(module.dnc)

    first_bn_name = None
    for n, m in model.named_modules():
        if isinstance(m, AdvWrapper) and isinstance(m.module, nn.BatchNorm2d):
            first_bn_name = n
            handle = m.register_forward_hook(_calib_hook)
            break

    criterion = torch.nn.CrossEntropyLoss()
    _, valloader = load_dataset(opt, split='val')
    test(model, valloader, criterion, device, desc='Calibrate')
    std_mean = torch.cat(diff_list).mean()

    diff_list.clear()
    # _, valloader = load_dataset(opt, split='val', noise=True, noise_type='expand')
    # test(model, valloader, criterion, device, desc='Calibrate')
    _, spatialloader = load_dataset(opt, split='val', aug=True)
    test(model, spatialloader, criterion, device, desc='Calibrate')
    noise_mean = torch.cat(diff_list).mean()

    boundary = (std_mean + noise_mean) / 2
    decision_factor = 1.0
    rgetattr(model, first_bn_name).boundary = boundary * decision_factor
    handle.remove()

    # Evaluate
    _, testloader = load_dataset(opt, split='test')
    std_acc, _ = test(model, testloader, criterion, device)
    print('[info] the normal accuracy is {:.4f}%'.format(std_acc))

    # partial_noisy_acc = []
    # for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
    #     _, testloader = load_dataset(opt, split='test', noise=True, noise_type='replace', gblur_std=std)
    #     acc, _ = test(model, testloader, criterion, device)
    #     print('[info] the robustness accuracy for std {:.1f} is {:.4f}%'.format(std, acc))
    #     partial_noisy_acc.append(acc)
    #
    # _, testloader = load_dataset(opt, split='test', noise=True, noise_type='append')
    # noisy_acc, _ = test(model, testloader, criterion, device)
    # print('[info] the robustness accuracy is {:.4f}%'.format(noisy_acc))

    _, spatialloader = load_dataset(opt, split='test', aug=True)
    spatial_acc, _ = test(model, spatialloader, criterion, device)
    print('[info] the spatial robustness accuracy is {:.4f}%'.format(spatial_acc))

    for srange in [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]:
        _, spatialloader = load_dataset(opt, split='test', aug=True, sub_range=srange)
        acc, _ = test(model, spatialloader, criterion, device)
        print('[info] the spatial robustness accuracy for sub-range {}/{} is {:.4f}%'.format(srange[0], srange[1], acc))

    # print('Saving model...')
    # state = {
    #     'net': model.state_dict(),
    #     'std_acc': std_acc,
    #     'noisy_acc': noisy_acc,
    #     'partial_noisy_acc': partial_noisy_acc,
    #     'spatial_acc': spatial_acc
    # }
    # torch.save(state, get_model_path(opt, state=f'switch_{opt.fs_method}_g{opt.gpu}'))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)

    switch_on_the_fly(opt, model, device)


if __name__ == '__main__':
    main()

