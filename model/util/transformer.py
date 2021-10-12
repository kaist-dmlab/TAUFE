import numpy as np
from skimage import io, transform
import torch

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample[0], sample[1]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        bbox = bbox * [new_w / w, new_h / h, new_w / w, new_h / h]

        return img, bbox


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample[0], sample[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top: top + new_h,
                      left: left + new_w]

        bbox = bbox - [left, top, left, top]

        return img, bbox


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample[0], sample[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h)/2)
        left = int((w - new_w)/2)

        img = image[top: top + new_h,
              left: left + new_w]

        bbox = bbox - [left, top, left, top]

        return img, bbox

class Normalize(object):
    def __call__(self, sample):
        image, bbox = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = (image-0.5)/0.5
        return img, bbox


class ToTensor(object):
    def __call__(self, sample):
        image, bbox = sample[0], sample[1]

        #print(image.shape)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(bbox)
