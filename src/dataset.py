import math
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from augmentor import GeneticAugmentor, NeighborAugmentor
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
        sidx = self.non_rob_idx[idx]
        (x, y), augmentor = self.dataset[sidx], self.augmentors[sidx]
        x, *_ = augmentor(x, single=True)
        return x, y

    def __len__(self):
        return len(self.non_rob_idx)

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
        self.non_rob_idx = [i for i, r in enumerate(self.is_robust) if r is False]


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
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4
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


def load_dataset(opt, split, noise=False, aug=False, **kwargs):
    assert opt.dataset.upper() in dir(torchvision.datasets)
    entry = eval(f'torchvision.datasets.{opt.dataset.upper()}')

    # handle split
    if split == 'test':
        base_dataset = entry(root=opt.data_dir, train=False, download=True)
    elif split == 'val':
        base_largeset = entry(root=opt.data_dir, train=True, download=True)
        _, base_dataset = train_test_split(
            base_largeset, test_size=1./50, random_state=opt.seed, stratify=base_largeset.targets)
    elif split == 'train':
        base_largeset = entry(root=opt.data_dir, train=True, download=True)
        base_dataset, _ = train_test_split(
            base_largeset, test_size=1./50, random_state=opt.seed, stratify=base_largeset.targets)
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

    # handle augmentation
    if aug is True:
        if split == 'train':
            dataset = SenseiAugmentedDataset(
                dataset,
                opt.robust_threshold, popsize=opt.popsize, crossover_prob=opt.crossover_prob
            )
            return dataset, None
        else:
            dataset = NeighborAugmentedDataset(dataset, opt.num_neighbors, seed=opt.seed)

    # handle data loader
    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=4
    )

    return dataset, dataloader

