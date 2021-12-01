import numpy as np
import torch
import os

#from torchvision.utils import save_image

class TinyImagesSample(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        file_path = file_path #'/home/pdm102207/AL_OOD/data/80mTiny/50K_sample.npy'

        def read_data(filepath):
            filename = os.listdir(filepath)[0] #[0]
            print(filename)

            data = np.load(filepath+'/'+filename).astype(np.uint8)
            #print(data[0])

            label = np.zeros(data.shape[0]).astype(np.uint8)
            return data, label

        self.data, self.label = read_data(file_path)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(data)#.transpose(1, 2, 0))
            #save_image(img, '../../logs/80mTiny_sample_v2.png')

        return img, label

    def __len__(self):
        return len(self.label)