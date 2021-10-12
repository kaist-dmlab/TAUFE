import numpy as np
import torch
import os
import pickle

class Imgnet64(torch.utils.data.Dataset):

    def __init__(self, ood=False, train=True, transform=None, in_class = np.arange(10)):
        def read_data(data_folder, train, dict_class):
            if ood == False:
                if train==False: #in-distribution, test set
                    file = open(os.path.join(data_folder, 'val_data'), 'rb')
                    dict = pickle.load(file)
                    d = dict['data']
                    l = np.array(dict['labels'])-1

                    idx = np.where(np.isin(l, in_class)==True)[0] # test_set from val_data & l<10

                    data = d[idx]
                    label = l[idx]

                    label = np.array([dict_class[l] for l in label])

                    data = np.fromstring(data, dtype='uint8').reshape((-1, 64, 64, 3))
                    print(data.shape, label.shape)
                    return data, label
                else: #in-distribution, train set
                    file_name = os.path.join(data_folder, 'train_data_batch_')
                    data = None
                    label = None
                    for i in range(10):
                        file = open(file_name+str(i+1), 'rb')
                        dict = pickle.load(file)
                        d = dict['data']
                        l = np.array(dict['labels'])-1

                        idx = np.where(np.isin(l, in_class)==True)[0]  # train_set from train_data & l<10
                        d = d[idx]
                        l = l[idx]

                        l = np.array([dict_class[ll] for ll in l])
                        print("label: ", l)

                        if i % 9 == 0:
                            print(i)

                        if i ==0:
                            data = d
                            label = l
                        else:
                            data = np.concatenate((data,d), axis=0)
                            label = np.concatenate((label, l), axis=0)
                    data = np.fromstring(data, dtype='uint8').reshape((-1, 64, 64, 3))
                    print(data.shape, label.shape)
                    return data, label
            else: # OOD!!!!
                file_name = os.path.join(data_folder, 'train_data_batch_')
                data = None
                label = None
                for i in range(10):
                    file = open(file_name + str(i+1), 'rb')
                    dict = pickle.load(file)
                    d = dict['data']
                    d = np.fromstring(d, dtype='uint8').reshape((-1, 64, 64, 3))
                    l = np.array(dict['labels'])-1

                    idx = np.where(np.isin(l, in_class)==False)[0]  # train_set from train_data & l<10
                    np.random.shuffle(idx)

                    d = d[idx][:1285]
                    l = l[idx][:1285]

                    print("label: ", l)

                    if i%9==0:
                        print(i)

                    if i == 0:
                        data = d
                        label = l
                    else:
                        data = np.concatenate((data, d), axis=0)
                        label = np.concatenate((label, l), axis=0)
                data = np.fromstring(data, dtype='uint8').reshape((-1, 64, 64, 3))
                print(data.shape, label.shape)
                return data, label

        data_folder = '/data/pdm102207/Imgnet64/'
        dict_class = {}
        for i in range(10):
            dict_class[in_class[i]] = i
        self.data, self.label = read_data(data_folder, train, dict_class)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label)