import cv2
import torch
from scipy import io as mat_io
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, mode, data_dir, metas, transform, limit):

        self.data_dir = data_dir
        self.data = []
        self.bbox = []
        self.target = []

        self.to_tensor = transforms.ToTensor()
        self.mode = mode
        self.resize_width = 224
        self.resize_height = 224

        self.transform = transform
        self.limit = limit

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx >= limit:
                    break
            #if idx %100 == 0:
            #    print(idx)

            if mode == 'train':
                #img_path = data_dir + 'cars_train/' + img_[5][0]
                #img = io.imread(img_path)
                #self.data.append(img)

                self.data.append(data_dir + 'cars_train/' + img_[5][0])
                # if self.mode == 'train':
                x_min = img_[0][0][0]
                y_min = img_[1][0][0]

                x_max = img_[2][0][0]
                y_max = img_[3][0][0]

                b = [x_min, y_min, x_max, y_max]
                #print(b)
                self.bbox.append(b)
                self.target.append(img_[4][0][0])
            elif mode == 'test':
                #img_path = data_dir + 'cars_test/' + img_[4][0]
                #img = io.imread(img_path)
                #self.data.append(img)

                self.data.append(data_dir + 'cars_test/' + img_[4][0])
                # if self.mode == 'train':
                x_min = img_[0][0][0]
                y_min = img_[1][0][0]

                x_max = img_[2][0][0]
                y_max = img_[3][0][0]

                b = [x_min, y_min, x_max, y_max]
                #print(b)
                self.bbox.append(b)
                #self.target.append(img_[4][0][0])

    def __getitem__(self, idx):
        image = io.imread(self.data[idx])
        #image = self.data[idx]
        bbox = np.array(self.bbox[idx]) #torch.tensor(self.bbox[idx], dtype=torch.float)
        #print(image.shape)
        #print(bbox)
        if len(image.shape) == 2:  # this is gray image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #img_resized = cv2.resize(image, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC)

        image, bbox = self.transform([image, bbox])
        if self.mode == 'train':
            return image, bbox
        elif self.mode == 'val' or self.mode == 'test':
            return image, bbox

    def __len__(self):
        return len(self.data)