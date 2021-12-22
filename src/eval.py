import copy

import torch

from dataset import load_dataset
from model import load_model
from arguments import advparser as parser
from train import test
from correct import construct_model, ReplaceCorrect
from utils import *


def evaluate(opt, model, device, eval_std=True, eval_noise=False, eval_spatial=False):
    criterion = torch.nn.CrossEntropyLoss()

    if eval_std is True:
        _, testloader = load_dataset(opt, split='test')
        acc, _ = test(model, testloader, criterion, device)
        print('[info] the base accuracy is {:.4f}%'.format(acc))

    if eval_noise is True:
        for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
            _, noiseloader = load_dataset(opt, split='test', noise=True,
                                          noise_type='replace', gblur_std=std)
            acc, _ = test(model, noiseloader, criterion, device)
            print('[info] the robustness accuracy for std {:.1f} is {:.4f}%'.format(std, acc))

        _, noiseloader = load_dataset(opt, split='test', noise=True, noise_type='append')
        acc, _ = test(model, noiseloader, criterion, device)
        print('[info] the noise robustness accuracy is {:.4f}%'.format(acc))

    if eval_spatial is True:
        _, spatialloader = load_dataset(opt, split='test', aug=True)
        acc, _ = test(model, spatialloader, criterion, device)
        print('[info] the spatial robustness accuracy is {:.4f}%'.format(acc))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    if opt.crt_method == 'none':  # pretrained
        if opt.fs_method == 'ratioestim':
            print('[CASE] Ratio Esitmation')
            model = load_model(opt, pretrained=False)
            for r in range(9):
                opt.susp_ratio = 0.15 + 0.1 * r  # from [0.15, 0.95, 0.1]
                model2 = copy.deepcopy(model)
                model2 = construct_model(opt, model2).to(device)
                rsymbol = str(int(opt.susp_ratio*100))
                ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_r{rsymbol}_g{opt.gpu}'))
                model2.load_state_dict(ckp['net'])
                for n, m in model2.named_modules():
                    if isinstance(m, ReplaceCorrect):
                        m.indices = ckp['indices'][n]
                evaluate(opt, model2, device, eval_std=False, eval_noise=True)
        else:
            model = load_model(opt, pretrained=True).to(device)
            evaluate(opt, model, device)
    elif opt.crt_method == 'patch':  # deeppatch and deepcorrect
        model = load_model(opt, pretrained=False)
        model = construct_model(opt, model).to(device)
        ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
        if opt.pt_method == 'DP-s':
            evaluate(opt, model, device, eval_spatial=True)
        else:
            evaluate(opt, model, device, eval_noise=True)
    elif opt.crt_method == 'finetune':
        model = load_model(opt, pretrained=False).to(device)
        ckp = torch.load(get_model_path(opt, state=f'{opt.crt_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
        evaluate(opt, model, device, eval_noise=True)
    elif opt.crt_method == 'sensei':
        model = load_model(opt, pretrained=False).to(device)
        ckp = torch.load(get_model_path(opt, state=f'{opt.crt_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
        evaluate(opt, model, device, eval_spatial=True)
    else:
        print('Please check input parameters manually')


if __name__ == '__main__':
    main()

