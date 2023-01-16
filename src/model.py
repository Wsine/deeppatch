import torch
import torchvision

from utils import get_model_path

from vendor.playground.svhn.model import svhn as svhn_loader


def load_model(opt, pretrained=False):
    if 'cifar' in opt.dataset:
        model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            f'{opt.dataset}_{opt.model}',
            pretrained=pretrained
        )
    elif opt.dataset == 'svhn':
        model = svhn_loader(n_channel=32, pretrained=pretrained)
    elif opt.dataset == 'tinyimagenet':
        # Load ResNet18
        model = torchvision.models.resnet18()
        # Finetune Final few layers to adjust for tiny imagenet input
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 200)
        # w224_state = load_torch_object(ctx, 'resnet18_224_w.pt')
        # model.load_state_dict(w224_state['net'])
        if opt.model == 'resnet18f':
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            model.maxpool = torch.nn.Sequential()  # type: ignore
        if pretrained is True:
            model = resume_model(opt, model, state='pretrained')
    else:
        raise ValueError('Invalid dataset name')
    return model


def load_checkpoint(opt, state='pretrained'):
    ckp = torch.load(
        get_model_path(opt, state=state),
        map_location=torch.device('cpu')
    )
    return ckp


def resume_model(opt, model, state='pretrained'):
    ckp = load_checkpoint(opt, state)
    model.load_state_dict(ckp['net'])
    return model

