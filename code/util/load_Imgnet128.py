import numpy as np
import torch
import os
import pickle
from PIL import Image

class Imgnet128(torch.utils.data.Dataset):

    def __init__(self, ood=False, train=True, transform=None, in_class = np.arange(10)):
        def read_data(data_folder, train, in_class):

            #val_class_list = os.listdir(data_folder+'val_blurred')

            if ood == False: # in-distribution !!!
                if train==False: # test set !!!
                    test_folder = 'val_blurred/'
                    class_list = np.array(os.listdir(data_folder + test_folder))
                    data = []
                    label = []
                    for i, c_name in enumerate(class_list[in_class]):
                        print(i)
                        files = os.listdir(data_folder + test_folder + c_name)
                        for f in files:
                            img = Image.open(data_folder + test_folder + c_name + '/' + f)
                            d = np.asarray(img)
                            data.append(d)
                            label.append(i)
                    print(len(data), len(label))
                    print(min(label), max(label))
                    return data, label
                else: # train set !!!
                    train_folder = 'train_blurred/'
                    class_list = np.array(os.listdir(data_folder + train_folder))
                    data = []
                    label = []
                    for i, c_name in enumerate(class_list[in_class]):
                        if i % 5 == 0:
                            print(i)
                        files = os.listdir(data_folder+train_folder+c_name)
                        for f in files:
                            img = Image.open(data_folder+train_folder+c_name+'/'+f)
                            d = np.asarray(img)
                            data.append(d)
                            label.append(i)
                    print(len(data), len(label))
                    print(min(label), max(label))
                    return data, label
            else: # OOD!!!!
                train_folder = 'train_blurred/'
                class_list = np.array(os.listdir(data_folder + train_folder))
                data = []
                label = []
                ood_class = np.delete(np.arange(1000), in_class)
                for i, c_name in enumerate(class_list):
                    if i%100 == 0:
                        print(i)
                    if i in ood_class:
                        files = os.listdir(data_folder + train_folder + c_name)
                        for f in files[:14]:
                            img = Image.open(data_folder + train_folder + c_name + '/' + f)
                            d = np.asarray(img)
                            data.append(d)
                            label.append(i)
                print(len(data), len(label))
                print(min(label), max(label))
                return data, label

        data_folder = '/data/pdm102207/Imgnet/'

        self.data, self.label = read_data(data_folder, train, in_class)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label)