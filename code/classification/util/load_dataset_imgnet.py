import os
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from .load_Imgnet224 import Imgnet224
import numpy as np

def load_datasets(in_data_name, ood_data_name, resolution):
    # Transform
    if in_data_name == 'imgnet10':
        T_normalize = T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)) # imgnet-10

    if resolution == 64:
        train_transform = T.Compose([T.ToPILImage(), T.RandomResizedCrop(size=64), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        test_transform = T.Compose([T.ToPILImage(), T.Resize(80), T.CenterCrop(size=64), T.ToTensor(), T_normalize])
        place365_transform = T.Compose([T.RandomCrop(size=64), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        place365_test_transform = T.Compose([T.Resize(80), T.CenterCrop(size=64), T.ToTensor(), T_normalize])

    elif resolution == 128:
        train_transform = T.Compose([T.ToPILImage(), T.RandomResizedCrop(size=128), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        test_transform = T.Compose([T.ToPILImage(), T.Resize(150), T.CenterCrop(size=128), T.ToTensor(), T_normalize])
        place365_transform = T.Compose([T.RandomCrop(size=128), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        place365_test_transform = T.Compose([T.Resize(150), T.CenterCrop(size=128), T.ToTensor(), T_normalize])

    elif resolution == 224:
        train_transform = T.Compose([T.ToPILImage(),T.RandomResizedCrop(size=224),T.RandomHorizontalFlip(),T.ToTensor(),T_normalize])
        test_transform = T.Compose([T.ToPILImage(),T.Resize(256),T.CenterCrop(size=224),T.ToTensor(),T_normalize])
        place365_transform = T.Compose([T.RandomCrop(size=224),T.RandomHorizontalFlip(),T.ToTensor(),T_normalize])
        place365_test_transform = T.Compose([T.Resize(256), T.CenterCrop(size=224), T.ToTensor(), T_normalize])

    # in-class index
    in_class = np.arange(1000)
    np.random.shuffle(in_class)
    in_class = in_class[:10]
    print("in_class: ", in_class)

    # In-distribution
    if in_data_name == 'imgnet10':
        train_in = Imgnet224(ood=False, train=True, transform=train_transform, in_class=in_class)
        test_in = Imgnet224(ood=False, train=False, transform=test_transform, in_class=in_class)

    # OOD
    if ood_data_name == 'imgnet990':
        train_ood = Imgnet224(ood=True, train=True, transform=train_transform, in_class=in_class)

        folder = '/data/pdm102207/places365/places365_standard/val/'
        #valdir = os.path.join(folder, 'val')
        test_ood = torchvision.datasets.ImageFolder(folder, transform=place365_test_transform)

    elif ood_data_name == 'places365':
        folder = '/data/pdm102207/places365/places365_standard/train/'
        # valdir = os.path.join(folder, 'val')
        train_ood = torchvision.datasets.ImageFolder(folder, transform=place365_transform)

        test_ood = Imgnet224(ood=True, train=False, transform=test_transform, in_class=in_class)

    return train_in, test_in, train_ood, test_ood