import os
import numpy as np
import pickle

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
from .transformer import Rescale, CenterCrop, RandomCrop, ToTensor, Normalize

from .load_CUB200_bbox import CUB200_Dataset
from .load_Imgnet224 import Imgnet224

#from torchvision.utils import save_image

def load_datasets(in_data_name, ood_data_name, n_shots):

    # transform image, label, bbox altogether
    dataset_transforms = dict(
        train=T.Compose([Rescale((256, 256)), RandomCrop(224), ToTensor(), Normalize()]),
        test=T.Compose([Rescale((224, 224)), ToTensor(), Normalize()])
    )

    T_normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    imgnet_transform = T.Compose([T.ToPILImage(), T.RandomResizedCrop(size=224), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
    place365_transform = T.Compose([T.RandomResizedCrop(size=224), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])

    # in-distribution
    if in_data_name == 'cub200':
        data_root = '/data/pdm102207/CUB/'
        metadata_root = '../../data/CUB/metadata/'
        train_in = CUB200_Dataset(data_root=data_root, metadata_root=os.path.join(metadata_root, 'train'),
                                  transform=dataset_transforms['train'], n_shot=n_shots)
        test_in = CUB200_Dataset(data_root=data_root, metadata_root=os.path.join(metadata_root, 'test'),
                                 transform=dataset_transforms['test'], n_shot=None)
    # OOD
    train_ood = None
    if ood_data_name == 'imgnet':
        train_ood = Imgnet224(ood=True, train=True, transform=imgnet_transform, in_class=[])
    elif ood_data_name == 'places365':
        folder = '/data/pdm102207/places365/places365_standard/train'
        train_ood = torchvision.datasets.ImageFolder(folder, transform=place365_transform)

    return train_in, test_in, train_ood