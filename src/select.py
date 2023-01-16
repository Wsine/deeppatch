import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from model import load_model
from dataset import load_dataset
from arguments import advparser as parser
from correct import patch
from train import test
from utils import *


dispatcher = AttrDispatcher('fs_method')


@dispatcher.register('ratioestim')
def blame_ratio_eval(opt, model, device):
    for r in range(9):
        opt.susp_ratio = 0.15 + 0.1 * r  # from [0.15, 0.95, 0.1]
        model2 = copy.deepcopy(model)
        opt.fs_method = f'ratioestim_r{str(int(opt.susp_ratio*100))}'
        output_file = get_model_path(opt, state=f'patch_{opt.fs_method}')
        if os.path.exists(output_file):
            continue
        patch(opt, model2, device)


def extract_feature_map(lname, model, dataloader, device):
    feature_map = []

    def _hook(module, finput, foutput):
        feature_map.append(foutput.detach())

    module = rgetattr(model, lname)
    handle = module.register_forward_hook(_hook)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (pred_labels, trg_labels) = test(
        model, dataloader, criterion, device,
        desc='Extract', return_label=True, tqdm_leave=False)
    feature_map = torch.cat(feature_map, dim=0)

    handle.remove()

    return feature_map, (pred_labels, trg_labels), base_acc


@dispatcher.register('perfloss')
def performance_loss(opt, model, device):
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='expand')
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    base_acc, _ = test(model, valloader, criterion, device)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    suspicious = {}
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules'):
        module = rgetattr(model, lname)
        perfloss = []
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            acc, _ = test(model, valloader, criterion, device, tqdm_leave=False)
            perfloss.append(base_acc - acc)
            handle.remove()

        score = sorted(perfloss)
        indices = sorted(range(len(perfloss)), key=lambda i: perfloss[i])
        suspicious[lname] = {
            'score': score,
            'indices': indices
        }

    return suspicious


@dispatcher.register('featswap')
@torch.no_grad()
def featuremap_swap(opt, model, device):
    _, valloader1 = load_dataset(opt, split='val', noise=False)
    _, valloader2 = load_dataset(opt, split='val', noise=True, noise_type='random')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)

    suspicious = {}
    num_modules = len(list(model.modules()))
    for lname, module in tqdm(model.named_modules(), total=num_modules, desc='Modules'):
        if isinstance(module, nn.Conv2d):
            fmaps, _, _ = extract_feature_map(lname, model, valloader1, device)

            def _substitute_feature(filter_index):
                def __hook(module, finput, foutput):
                    global fmaps_idx
                    batch_size = foutput.size(0)
                    foutput[:, filter_index] = fmaps[fmaps_idx:fmaps_idx+batch_size, filter_index]
                    fmaps_idx += batch_size
                    return foutput
                return __hook

            recover = []
            for fidx in tqdm(range(module.out_channels), desc='Filters', leave=False):
                handler = module.register_forward_hook(_substitute_feature(fidx))
                global fmaps_idx
                fmaps_idx = 0
                swap_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)
                recover.append(swap_acc - base_acc)
                handler.remove()

            score = sorted(recover, reverse=True)
            indices = sorted(range(len(recover)), key=lambda i: recover[i], reverse=True)
            suspicious[lname] = {
                'score': score,
                'indices': indices
            }

    return suspicious


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)

    result = dispatcher(opt, model, device)
    if 'ratioestim' in opt.fs_method:
        return

    result_name = 'susp_filters.json'
    export_object(opt, result_name, opt.fs_method, result)


if __name__ == '__main__':
    main()

