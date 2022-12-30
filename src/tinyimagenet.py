# taken from : https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54
from PIL import Image
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from tqdm import tqdm

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""
TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
TINY_IMAGENET_MD5 = '90528d7ca1a48142e341f4ef8d21d0de'

def download_and_unzip(URL, root_dir, base_dir):
    zipfile = os.path.join(root_dir, 'tiny-imagenet-200.zip')
    txtfile = os.path.join(root_dir, base_dir, 'wnids.txt')
    if check_integrity(zipfile, TINY_IMAGENET_MD5) and os.path.isfile(txtfile):
        print('Files already downloaded and verified')
        return
    download_and_extract_archive(
        URL, root_dir, md5=TINY_IMAGENET_MD5
    )

"""Creates a paths datastructure for the tiny imagenet.
Args:
    root_dir: Where the data is located
    download: Download if the data is not there
Members:
    label_id:
    ids:
    nit_to_words:
    data_dict:
"""
class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        base_dir = 'tiny-imagenet-200'
        if download:
            download_and_unzip(TINY_IMAGENET_URL, root_dir, base_dir)
        train_path = os.path.join(root_dir, base_dir, 'train')
        val_path = os.path.join(root_dir, base_dir, 'val')
        test_path = os.path.join(root_dir, base_dir, 'test')

        wnids_path = os.path.join(root_dir, base_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, base_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                                                            os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.
Args:
    root_dir: Root directory for the data
    mode: One of "train", "test", or "val"
    preload: Preload into memory
    load_transform: Transformation to use at the preload time
    transform: Transformation to use at the retrieval time
    download: Download the dataset
Members:
    tinp: Instance of the TinyImageNetPaths
    img_data: Image data
    label_data: Label data
"""
class TinyImageNetDataset(Dataset):
    def __init__(self,
            root, mode, preload=False, download=True, max_samples=None,
            load_transform=None, transform=None, target_transform=None):
        tinp = TinyImageNetPaths(root, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.target_transform = target_transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.imgs = []
        self.targets = []

        self.max_samples = max_samples
        self.samples = tinp.paths[self.mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]  # type: ignore

        if self.preload:
            load_desc = "Preloading {} data...".format(self.mode)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                with open(s[0], 'rb') as f:
                    img = Image.open(f).convert('RGB')
                self.imgs.append(img)
                if self.mode != 'test':
                    self.targets.append(s[self.label_idx])

            if load_transform:
                for lt in load_transform:
                    result = lt(self.imgs, self.targets)
                    self.imgs, self.targets = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])
        else:
            if self.mode != 'test':
                self.targets = [self.samples[i][self.label_idx] for i in range(self.samples_num)]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.imgs[idx]
            lbl = None if self.mode == 'test' else self.targets[idx]
        else:
            s = self.samples[idx]
            with open(s[0], 'rb') as f:
                img = Image.open(f).convert('RGB')
            lbl = None if self.mode == 'test' else s[self.label_idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl

