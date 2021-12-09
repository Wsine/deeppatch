import torch

from utils import get_model_path


def load_model(opt, pretrained=False):
    if 'cifar' in opt.dataset:
        model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            f'{opt.dataset}_{opt.model}',
            pretrained=pretrained
        )
        return model
    else:
        raise ValueError('Invalid dataset name')


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
