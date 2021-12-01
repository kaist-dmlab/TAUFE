import os
import torch
import pickle
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
from .load_80mTiny import TinyImages
import numpy as np
#from torchvision.utils import save_image

class LSUNImagesSample(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):

        file_path = file_path #'/home/pdm102207/AL_OOD/data/80mTiny/50K_sample.npy'

        def read_data(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f).astype(np.uint8)
            label = torch.zeros(data.shape[0])#.astype(np.uint8)

            print("data.shape:", data.shape)
            #print(data[0])
            #print(label.shape)

            return data, label

        self.data, self.label = read_data(file_path)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(data)
        return img, label

    def __len__(self):
        return len(self.label)

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = CIFAR10(root=file_path,download=download,train=train,transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

class MyCIFAR100(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar100 = CIFAR100(root=file_path,download=download,train=train,transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)

class MyTiny(Dataset):
    def __init__(self, file_path, transform):
        self.tiny = TinyImages(file_path, transform=transform)

    def __getitem__(self, index):
        data, target = self.tiny[index]
        return data, target, index

    def __len__(self):
        return len(self.tiny)

class MySVHN(Dataset):
    def __init__(self, file_path, download, transform):
        self.svhn = SVHN(file_path, download=False, transform=transform)

    def __getitem__(self, index):
        data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        return len(self.svhn)

class MyLSUN(Dataset):
    def __init__(self, file_path, download, transform):
        self.lsun = LSUN(file_path, download=False, transform=transform)

    def __getitem__(self, index):
        data, target = self.lsun[index]
        return data, target, index

    def __len__(self):
        return len(self.lsun)

def load_datasets(in_data_name, ood_data_name, index):
    # Transform
    if in_data_name == 'cifar10':
        T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif in_data_name == 'cifar100':
        T_normalize = T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100

    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])#
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    lsun_transform = T.Compose([T.RandomResizedCrop(size=32), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
    #lsun_test_transform = T.Compose([T.Resize(150), T.CenterCrop(size=128), T.ToTensor(), T_normalize])

    tiny_transform = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])
    #tiny_test_transform = T.Compose([T.ToPILImage(), T.ToTensor(), T_normalize])

    if index == False:
        # In-distribution
        if in_data_name == 'cifar10':
            file_path = '../../data/cifar10/'
            train_in = CIFAR10(file_path, train=True, download=False, transform=train_transform)
            test_in = CIFAR10(file_path, train=False, download=False, transform=test_transform)

        elif in_data_name == 'cifar100':
            file_path = '../../data/cifar100/'
            train_in = CIFAR100(file_path, train=True, download=False, transform=train_transform)
            test_in = CIFAR100(file_path, train=False, download=False, transform=test_transform)

        # OOD
        if ood_data_name == 'svhn':
            file_path = '../../data/svhn/'
            train_ood = SVHN(file_path, split='extra', download=True, transform=train_transform) #strong_transforms
            test_ood = SVHN(file_path, split='test', download=False, transform=test_transform)

        elif ood_data_name == 'lsun':
            file_path = '/data/pdm102207/lsun'
            classes='train'
            train_ood = LSUN(file_path, classes=classes, transform=lsun_transform)

            file_path = '../../data/svhn/'
            test_ood = SVHN(file_path, split='test', download=False, transform=test_transform)

        elif ood_data_name == 'lsun_sample':
            file_path = '../../data/lsun/lsun_x_sample_50000_np.pickle'
            lsun_transform = T.Compose([T.ToPILImage(), T.RandomResizedCrop(size=32), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
            train_ood = LSUNImagesSample(file_path, transform=lsun_transform)

            file_path = '../../data/svhn/'
            test_ood = SVHN(file_path, split='test', download=False, transform=test_transform)

        elif ood_data_name == '80mTiny':
            train_ood = TinyImages(transform=tiny_transform)

            file_path = '../../data/svhn/'
            test_ood = SVHN(file_path, split='test', download=True, transform=test_transform)

    if index == True:
        if in_data_name == 'cifar10':
            file_path = '../../data/cifar10/'
            train_in = MyCIFAR10(file_path, train=True, download=False, transform=train_transform)
            test_in = MyCIFAR10(file_path, train=False, download=False, transform=test_transform)

        elif in_data_name == 'cifar100':
            file_path = '../../data/cifar100/'
            train_in = MyCIFAR100(file_path, train=True, download=False, transform=train_transform)
            test_in = MyCIFAR100(file_path, train=False, download=False, transform=test_transform)

        # OOD
        if ood_data_name == 'svhn':
            file_path = '../../data/svhn/'
            train_ood = MySVHN(file_path, download=False, transform=train_transform)

            file_path = '../../data/lsun/'
            test_ood = LSUN(file_path, classes='train', transform=train_transform)

        elif ood_data_name == 'lsun':
            file_path = '/data/pdm102207/lsun'
            #classes = ['bedroom_train', 'bridge_train', 'church_outdoor_train', 'classroom_train','conference_room_train',
            #           'dining_room_train', 'kitchen_train', 'living_room_train', 'restaurant_train', 'tower_train']
            classes='train'
            train_ood = MyLSUN(file_path, classes=classes, transform=train_transform)

            file_path = '../../data/svhn/'
            test_ood = SVHN(file_path, download=False, transform=train_transform)  # strong_transforms

        elif ood_data_name == '80mTiny':
            train_ood = MyTiny(transform=tiny_transform)
            file_path = '../../data/svhn/'
            test_ood = MySVHN(file_path, split='test', download=True, transform=test_transform)

    return train_in, test_in, train_ood, test_ood