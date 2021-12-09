import math
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from utils import cache_object


class RandomApply(object):
    def __init__(self, tran, p=0.5):
        self.tran = tran
        self.prob = p

    def __call__(self, x):
        p = random.random()
        if p < self.prob:
            x = self.tran(x)
        return x


class RandomApplyOne(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        t = random.choice(self.trans)
        return t(x)  # type: ignore


class PostTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.dataset)


def load_noisy_dataset(_, baseset, comm_trsf, noise_type, gblur_std=None):
    if noise_type == 'random':
        trsf = RandomApply(RandomApplyOne([
            T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
            for std in [0.5, 1., 1.5, 2., 2.5, 3.]
        ]), p=0.5)
        dataset = PostTransformDataset(
            baseset, transform=T.Compose([trsf] + comm_trsf)
        )
    elif noise_type == 'replace':
        assert gblur_std is not None, 'gblur_std should be a floating number'
        trsf = T.GaussianBlur(math.ceil(4*gblur_std)//2*2+1, sigma=gblur_std)
        dataset = PostTransformDataset(
            baseset, transform=T.Compose([trsf] + comm_trsf)
        )
    elif noise_type == 'expand' or noise_type == 'append':
        incset = [PostTransformDataset(
            baseset, transform=T.Compose(comm_trsf)
        )] if noise_type == 'append' else []
        for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
            trsf = T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
            incset.append(PostTransformDataset(
                baseset, transform=T.Compose([trsf] + comm_trsf)
            ))
        dataset = torch.utils.data.ConcatDataset(incset)
    else:
        raise ValueError('Invalid noise_type parameter')

    return dataset


@cache_object(filename='dataset_mean_std.pkl')
def compute_mean_std(opt, dataset_name):
    entry = eval(f'torchvision.datasets.{dataset_name}')
    dataset = entry(root=opt.data_dir, train=True, download=False, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    mean, std = 0., 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def load_dataset(opt, split, noise=False, **kwargs):
    entry = eval(f'torchvision.datasets.{opt.dataset.upper()}')

    # handle split
    if split == 'test':
        base_dataset = entry(root=opt.data_dir, train=False, download=True)
    elif split == 'val':
        base_largeset = entry(root=opt.data_dir, train=True, download=True)
        _, base_dataset = train_test_split(
            base_largeset, test_size=1./50, random_state=2021, stratify=base_largeset.targets)
    elif split == 'train':
        base_largeset = entry(root=opt.data_dir, train=True, download=True)
        base_dataset, _ = train_test_split(
            base_largeset, test_size=1./50, random_state=2021, stratify=base_largeset.targets)
    else:
        raise ValueError('Invalid parameter of split')

    # handle noise
    mean, std = compute_mean_std(opt, opt.dataset.upper())
    common_transformers = [T.ToTensor(), T.Normalize(mean, std)]
    if noise is True:
        dataset = load_noisy_dataset(opt, base_dataset, common_transformers, **kwargs)
    else:
        dataset = PostTransformDataset(
            base_dataset, transform=T.Compose(common_transformers)
        )

    # handle data loader
    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=2
    )

    return dataset, dataloader
