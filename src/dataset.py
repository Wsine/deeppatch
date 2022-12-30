import math
import random

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tinyimagenet import TinyImageNetDataset
from augmentor import GeneticAugmentor, NeighborAugmentor, RandomAugmentor
from vendor.augmix import augmentations
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


class RandomAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        image_shape = self.dataset[0][0].size()[-2:]
        self.augmentor = RandomAugmentor(image_shape)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.augmentor(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


class NeighborAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_neighbors, **kwargs):
        self.dataset = dataset
        self.num_neighbors = num_neighbors
        image_shape = self.dataset[0][0].size()[-2:]
        self.augmentor = NeighborAugmentor(image_shape, num_neighbors, **kwargs)

    def __getitem__(self, idx):
        x, y = self.dataset[idx // self.num_neighbors]
        n_idx = idx // len(self.dataset)
        x = self.augmentor(x, n_idx)
        return x, y

    def __len__(self):
        return len(self.dataset) * self.num_neighbors


# refer: https://dl.acm.org/doi/10.1145/3377811.3380415
class SenseiAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rob_threshold, **kwargs):
        self.dataset = dataset
        self.is_robust = [False] * len(dataset)
        image_shape = self.dataset[0][0].size()[-2:]
        self.augmentors = [
            GeneticAugmentor(image_shape, **kwargs)
            for _ in range(len(self.dataset))
        ]
        self.non_rob_idx = []
        self.rob_thres = rob_threshold

    def __getitem__(self, idx):
        if self.is_robust[idx] is True:
            return self.dataset[idx]
        else:
            (x, y), augmentor = self.dataset[idx], self.augmentors[idx]
            x, *_ = augmentor(x, single=True)
            return x, y

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def selective_augment(self, model, criterion, batch_size, device):
        model.eval()
        sel_batch = math.floor(batch_size / self.augmentors[0].popsize)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=sel_batch, shuffle=False, num_workers=4
        )
        for bidx, (xs, ys) in enumerate(tqdm(dataloader, desc='Select')):
            inputs, targets = [], []
            for i, (x, y) in enumerate(zip(xs, ys)):
                idx = bidx * sel_batch + i
                augmentor = self.augmentors[idx]
                inputs.extend(augmentor(x, single=False))
                targets.append(torch.ones(augmentor.popsize, dtype=torch.long) * y)
            inputs = torch.stack(inputs).to(device)  # type: ignore
            targets = torch.cat(targets).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for i in range(ys.size(0)):
                idx = bidx * sel_batch + i
                augmentor = self.augmentors[idx]
                bloss = loss[i * augmentor.popsize : (i + 1) * augmentor.popsize]
                augmentor.fitness(bloss, GeneticAugmentor.Fitness.LARGEST)
                self.is_robust[idx] = True if torch.all(bloss.lt(self.rob_thres)) else False


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


def augmix(image, preprocess, args):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if args.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
    1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)  # pyright: ignore

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.args = type('AugMixArgs', (object,), {
            'all_ops': False,
            'mixture_width': 3,
            'mixture_depth': 3,
            'aug_severity': 3
        })

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return augmix(x, self.preprocess, self.args), y
        else:
            im_tuple = (self.preprocess(x),
                        augmix(x, self.preprocess, self.args),
                        augmix(x, self.preprocess, self.args))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


@cache_object(filename='dataset_mean_std.pkl')
def compute_mean_std(opt, dataset_name):
    entry = eval(f'torchvision.datasets.{dataset_name}')
    dataset = entry(root=opt.data_dir, train=True, download=False, transform=T.ToTensor())
    data_r = torch.stack([x[0, :, :] for x, _ in dataset])
    data_g = torch.stack([x[1, :, :] for x, _ in dataset])
    data_b = torch.stack([x[2, :, :] for x, _ in dataset])
    mean = data_r.mean(), data_g.mean(), data_b.mean()
    std = data_r.std(), data_g.std(), data_b.std()
    return mean, std


def load_dataset(opt, split, noise=False, aug=False, mix=False, **kwargs):
    if opt.dataset == 'tinyimagenet':
        entry = TinyImageNetDataset
        # there are no labels in test split
        extra_args = {'mode': 'train' if split == 'train' else 'val'}
        common_transformers = [T.ToTensor()]
        stratify = 'targets'
    elif opt.dataset == 'svhn':
        entry = torchvision.datasets.SVHN
        extra_args = {
            'split': 'test' if split == 'test' else 'train',
            'target_transform': lambda t: (10 if t == 0 else t) - 1
        }
        mean = std = (0.5, 0.5, 0.5)
        common_transformers = [T.ToTensor(), T.Normalize(mean, std)]
        stratify = 'labels'
    else:  # cifar10 / cifar100
        entry = eval(f'torchvision.datasets.{opt.dataset.upper()}')
        extra_args = {'train': False if split == 'test' else True}
        mean, std = compute_mean_std(opt, opt.dataset.upper())
        common_transformers = [T.ToTensor(), T.Normalize(mean, std)]
        stratify = 'targets'

    # handle split
    if split == 'test':
        base_dataset = entry(root=opt.data_dir, download=True, **extra_args)
    elif split == 'val':
        base_largeset = entry(root=opt.data_dir, download=True, **extra_args)
        _, base_dataset = train_test_split(
            base_largeset, test_size=1./50, random_state=opt.seed, stratify=getattr(base_largeset, stratify))
    elif split == 'train':
        base_largeset = entry(root=opt.data_dir, download=True, **extra_args)
        base_dataset, _ = train_test_split(
            base_largeset, test_size=1./50, random_state=opt.seed, stratify=getattr(base_largeset, stratify))
    else:
        raise ValueError('Invalid parameter of split')

    # handle noise/mix
    if noise is True:
        dataset = load_noisy_dataset(opt, base_dataset, common_transformers, **kwargs)
    elif mix is True:
        dataset = AugMixDataset(
            base_dataset, preprocess=T.Compose(common_transformers), no_jsd=False)
    else:
        dataset = PostTransformDataset(
            base_dataset, transform=T.Compose(common_transformers)
        )

    # handle augmentation
    if aug is True:
        if split == 'train':
            if opt.crt_method == 'sensei' or opt.pt_method == 'DP-SS':
                dataset = SenseiAugmentedDataset(
                    dataset,
                    opt.robust_threshold, popsize=opt.popsize, crossover_prob=opt.crossover_prob
                )
                return dataset, None
            else:
                dataset = RandomAugmentedDataset(dataset)
        else:
            dataset = NeighborAugmentedDataset(dataset, opt.num_neighbors, seed=opt.seed, **kwargs)

    # handle data loader
    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=4
    )

    return dataset, dataloader

