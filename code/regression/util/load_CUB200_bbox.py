import munch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

#_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
#_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_IMAGE_MEAN_VALUE = [0.5, 0.5, 0.5]
_IMAGE_STD_VALUE = [0.5, 0.5, 0.5]
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, n_shot):
    """
    image_ids.txt has the structure
    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    """
    if n_shot is None:
        image_ids = []
        with open(metadata['image_ids']) as f:
            for line in f.readlines():
                image_ids.append(line.strip('\n'))
                '''
                if ood == False:
                    if int(line.split('.')[0]) <= 10:
                        image_ids.append(line.strip('\n'))
                    else:
                        break
                else:
                    if int(line.split('.')[0]) > 10:
                        image_ids.append(line.strip('\n'))
                '''
        return image_ids
    else:
        image_ids = []
        with open(metadata['image_ids']) as f:
            n_sample = 0
            class_now = 1
            for line in f.readlines():
                if int(line.split('.')[0])==class_now and n_sample < n_shot:
                    #print(int(line.split('.')[0]), n_sample)
                    image_ids.append(line.strip('\n'))
                    n_sample+=1
                if int(line.split('.')[0])!=class_now:
                    image_ids.append(line.strip('\n'))
                    n_sample=1
                    class_now = int(line.split('.')[0])
        return image_ids


def get_class_labels(metadata, n_shot):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    if n_shot is None:
        class_labels = {}
        with open(metadata.class_labels) as f:
            for line in f.readlines():
                image_id, class_label_string = line.strip('\n').split(',')
                class_labels[image_id] = int(class_label_string)
                '''
                if ood == False:
                    if int(line.split('.')) <= 10:
                        image_id, class_label_string = line.strip('\n').split(',')
                        class_labels[image_id] = int(class_label_string)
                    else:
                        break
                else:
                    if int(line.split('.')) > 10:
                        image_id, class_label_string = line.strip('\n').split(',')
                        class_labels[image_id] = int(class_label_string)
                '''
        return class_labels
    else:
        class_labels = {}
        with open(metadata.class_labels) as f:
            n_sample = 0
            class_now = 1
            for line in f.readlines():
                if int(line.split('.')[0]) == class_now and n_sample < n_shot:
                    image_id, class_label_string = line.strip('\n').split(',')
                    class_labels[image_id] = int(class_label_string)
                    n_sample += 1
                if int(line.split('.')[0]) != class_now:
                    #print(int(line.split('.')[0]))
                    image_id, class_label_string = line.strip('\n').split(',')
                    class_labels[image_id] = int(class_label_string)
                    n_sample = 1
                    class_now = int(line.split('.')[0])
        return class_labels

def get_bounding_boxes(metadata, n_shot):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    if n_shot is None:
        boxes = {}
        with open(metadata.localization) as f:
            for line in f.readlines():
                image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                #if image_id in boxes:
                #    boxes[image_id].append((x0, x1, y0, y1))
                #else:
                boxes[image_id] = np.array([x0, x1, y0, y1])
        return boxes
    else:
        boxes = {}
        with open(metadata.localization) as f:
            n_sample = 0
            class_now = 1
            for line in f.readlines():
                if int(line.split('.')[0]) == class_now and n_sample < n_shot:
                    image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                    x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                    #if image_id in boxes:
                    #    boxes[image_id].append((x0, x1, y0, y1))
                    #else:
                    boxes[image_id] = np.array([x0, x1, y0, y1])
                if int(line.split('.')[0]) != class_now:
                    image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                    x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                    #if image_id in boxes:
                    #    boxes[image_id].append((x0, x1, y0, y1))
                    #else:
                    boxes[image_id] = np.array([x0, x1, y0, y1])
                    n_sample = 1
                    class_now = int(line.split('.')[0])
        return boxes

def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


class CUB200_Dataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, n_shot):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root) #all paths to meta-data
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, n_shot) #proxy usually false
        self.image_labels = get_class_labels(self.metadata, n_shot)
        self.image_bbox = get_bounding_boxes(self.metadata, n_shot)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image_bbox = self.image_bbox[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = np.array(image.convert('RGB'))
        image, image_bbox = self.transform([image, image_bbox])
        return image, image_label, image_bbox, image_id

    def __len__(self):
        return len(self.image_ids)


def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))

    loaders = {
        split: DataLoader(
            CUB200_Dataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ),
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders
