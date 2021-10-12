# Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import pickle

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision
import torchvision.transforms as T

# Utils
from tqdm import tqdm

# Custom
import models.resnet224_bbox as resnet
from config import *
from util.utility import *
from util.load_Imgnet224 import Imgnet224
from util.load_CAR import CarsDataset
from util.transformer import Rescale, CenterCrop, RandomCrop, ToTensor, Normalize

# Seed
random.seed("Anonymized")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Load Data
_IMAGE_MEAN_VALUE = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225]

# transform image, labe, bbox altogether
dataset_transforms = dict(
    train=T.Compose([Rescale((256, 256)),RandomCrop(224),ToTensor(),Normalize()]),
    test=T.Compose([Rescale((224, 224)),ToTensor(),Normalize()])
)
Imgnet_normalize = T.Normalize(_IMAGE_MEAN_VALUE,_IMAGE_STD_VALUE)

imgnet224_transform = T.Compose([T.ToPILImage(),T.RandomResizedCrop(size=224),T.RandomHorizontalFlip(),T.ToTensor(),Imgnet_normalize])
place365_transform = T.Compose([T.RandomResizedCrop(size=224, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),Imgnet_normalize])

data_dir = '/data/anonymized/car/'
metadata_root = '/data/anonymized/car/devkit/'

# in_distribution
Car_train_in = CarsDataset(mode='train',data_dir=data_dir,metas=os.path.join(metadata_root, 'cars_train_annos.mat'),transform=dataset_transforms['train'],limit=NUM_TRAIN)
print("len(Car_train_in): ", len(Car_train_in))

Car_test = CarsDataset(mode='test',data_dir=data_dir,metas=os.path.join(metadata_root, 'cars_test_annos.mat'),transform=dataset_transforms['test'],limit=None)

# ood
imgnet224_train_ood = Imgnet224(ood = True, train=True, transform=imgnet224_transform, in_class=[])

folder = '/data/anonymized/places365/places365_standard/'
#traindir = os.path.join(folder, 'train')
valdir = os.path.join(folder, 'val')
place365_train_ood = torchvision.datasets.ImageFolder(valdir, transform=place365_transform)
#
def IoULoss(pred_box, gt_box):
    xymin = torch.max(pred_box, gt_box)  # [:,[0,1]]
    xymax = torch.min(pred_box, gt_box)  # [:,[2,3]]

    ixmin = xymin[:, 0]
    iymin = xymin[:, 1]
    ixmax = xymax[:, 2]
    iymax = xymax[:, 3]

    iw = torch.max(ixmax - ixmin + 1., torch.tensor(0.).cuda())
    ih = torch.max(iymax - iymin + 1., torch.tensor(0.).cuda())

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    unis = ((pred_box[:, 2] - pred_box[:, 0] + 1.) * (pred_box[:, 3] - pred_box[:, 1] + 1.) +
            (gt_box[:, 2] - gt_box[:, 0] + 1.) * (gt_box[:, 3] - gt_box[:, 1] + 1.) -
            inters)

    ious = inters/unis
    return 1-torch.mean(ious)

def DIoULoss(pred_box, gt_box):
    pred_center_x = (pred_box[:, 2] + pred_box[:, 0]) / 2
    pred_center_y = (pred_box[:, 3] + pred_box[:, 1]) / 2

    gt_center_x = (gt_box[:, 2] + gt_box[:, 0]) / 2
    gt_center_y = (gt_box[:, 3] + gt_box[:, 1]) / 2

    dist_center = (pred_center_x-gt_center_x)**2+(pred_center_y-gt_center_y)**2

    a = (pred_box[:, 0] - gt_box[:, 2]) ** 2 + (pred_box[:, 1] - gt_box[:, 3]) ** 2
    b = (pred_box[:, 0] - gt_box[:, 2]) ** 2 + (pred_box[:, 3] - gt_box[:, 1]) ** 2
    c = (pred_box[:, 2] - gt_box[:, 0]) ** 2 + (pred_box[:, 1] - gt_box[:, 3]) ** 2
    d = (pred_box[:, 2] - gt_box[:, 0]) ** 2 + (pred_box[:, 3] - gt_box[:, 1]) ** 2

    a = a.reshape((-1, a.size(0)))
    b = b.reshape((-1, b.size(0)))
    c = c.reshape((-1, c.size(0)))
    d = d.reshape((-1, d.size(0)))

    dist_diag = torch.cat((a,b,c,d),1)

    dist_diag_min = torch.max(dist_diag, 1).values

    return torch.mean(dist_center/dist_diag_min)

def train_epoch(models, criterion, optimizers, dataloaders, epoch, weight):
    models['backbone'].train()

    for data in tqdm(dataloaders['train_in'], leave=False, total=len(dataloaders['train_in'])):
        inputs = data[0].type(torch.FloatTensor).cuda()
        bboxs = data[1].type(torch.FloatTensor).cuda()

        optimizers['backbone'].zero_grad()

        out_c, out_r, features = models['backbone'](inputs)

        coord_loss_r = criterion["regression"](out_r, bboxs)
        iou_loss = IoULoss(out_r, bboxs)
        #diou_loss = DIoULoss(out_r, bboxs)

        loss = coord_loss_r + 0.4*iou_loss #+ 0.4*diou_loss # + weight * Blackhole_loss
        #print(coord_loss_r) #, iou_loss, diou_loss)

        loss.backward()
        optimizers['backbone'].step()

        del data

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, weight):
    print('>> Train a Model.')
    logs = []
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, weight)

        if epoch % 5 == 4:
            iou = test(models, dataloaders)
            print('epoch: {}, iou: {}'.format(epoch, iou))

            # save logs
            logs.append([epoch, iou])
            #np.savetxt('../logs_sup/Car_localization_Standard_n2000_per_epoch_v1.txt', logs, delimiter=',', fmt="%.4f")
    print('>> Finished.')

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters

    xymin = torch.max(pred_box, gt_box)#[:,[0,1]]
    ixmin = xymin[:,0]
    iymin = xymin[:,1]
    xymax = torch.min(pred_box, gt_box)#[:,[2,3]]
    ixmax = xymax[:,2]
    iymax = xymax[:,3]

    #print(xymin)
    #print(xymax)

    iw = torch.max(ixmax-ixmin+1., torch.tensor(0.).cuda())
    ih = torch.max(iymax-iymin+1., torch.tensor(0.).cuda())

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    unis = ((pred_box[:, 2] - pred_box[:, 0] + 1.) * (pred_box[:, 3] - pred_box[:, 1] + 1.) +
            (gt_box[:, 2] - gt_box[:, 0] + 1.) * (gt_box[:, 3] - gt_box[:, 1] + 1.) -
            inters)

    #print(gt_box, pred_box)
    #print(inters, unis)

    for i, u in enumerate(unis.cpu()):
        if u.data < 0:
            print(i)

    # 4. calculate the overlaps between pred_box and gt_box
    ious = inters / unis

    #print(ious)

    return torch.mean(ious)

#
def test(models, dataloaders):
    models['backbone'].eval()

    total_iou = 0

    f = nn.Softmax(dim=1)
    to_np = lambda x: x.data.cpu().tolist()
    with torch.no_grad():
        for (inputs, bbox) in dataloaders['test_in']:

            inputs = inputs.type(torch.FloatTensor).cuda()
            bbox = bbox.type(torch.FloatTensor).cuda()

            out_c, out_r, features = models['backbone'](inputs)

            smax, preds = torch.max(f(out_c.data), 1)
            pred_bbox = out_r.data

            batch_iou = get_iou(pred_bbox, bbox)
            total_iou += batch_iou.item()

    #acc = 100 * correct / total
    total_iou = total_iou/len(dataloaders['test_in'])

    return total_iou


def get_stratified_indices(labels, n):
    C = max(labels) + 1  # 10
    sampled_idx = []
    for c in range(C):
        idx = []
        for i, l in enumerate(labels):
            if l == c:
                idx += [i]
        random.shuffle(idx)
        sampled_idx += idx[:n]
    print("========Stratified Sampling========")
    print("Total: {}, Class: {}, per class: {}".format(len(sampled_idx), C, n))
    return sampled_idx


'''
#save as pickle
with open('../data/Imgnet224/imgnet224_train_in.txt', 'wb') as f:
    pickle.dump(imgnet224_train_in, f)
with open('../data/Imgnet224/imgnet224_train_ood.txt', 'wb') as f:
    pickle.dump(imgnet224_train_ood, f)
with open('../data/Imgnet224/imgnet224_test.txt', 'wb') as f:
    pickle.dump(imgnet224_test, f)


#read from pickle
with open('../data/Imgnet224/imgnet224_train_in.txt', 'rb') as f:
    imgnet224_train_in = pickle.load(f)
with open('../data/Imgnet224/imgnet224_train_ood.txt', 'rb') as f:
    imgnet224_train_ood = pickle.load(f)
with open('../data/Imgnet224/imgnet224_test.txt', 'rb') as f:
    imgnet224_test = pickle.load(f)
'''

train_in = Car_train_in
test_in = Car_test

# AL Main
if __name__ == '__main__':
    indices_in = list(range(len(train_in)))
    random.shuffle(indices_in)
    print("# of train_in samples", len(train_in))

    train_in_loader = DataLoader(train_in, batch_size=BATCH,
                                 sampler=SubsetRandomSampler(indices_in),
                                 pin_memory=True)

    test_in_loader = DataLoader(test_in, batch_size=BATCH)

    dataloaders = {'train_in': train_in_loader, 'test_in': test_in_loader}

    # Model
    resnet34 = resnet.ResNet34(num_classes=NUM_CLASS).cuda()
    # If use a Pre-train Model
    '''
    pretrained_model = torchvision.models.resnet34(pretrained=True)

    for name, param in resnet34.named_parameters():
        for n, p in pretrained_model.named_parameters():
            if name not in ['linear_classify.weight', 'linear_classify.weight', 'linear_reg.bias']:
                if n not in ['linear.weight', 'linear.bias']:
                    param = p
    '''
    models = {'backbone': resnet34}
    torch.backends.cudnn.benchmark = False

    # Loss, criterion and scheduler (re)initialization
    criterion_c = nn.CrossEntropyLoss(reduction='mean')
    #criterion_r = nn.MSELoss(reduction='mean')
    criterion_r = nn.SmoothL1Loss(reduction='mean').cuda()
    criterion = {"classification": criterion_c, "regression": criterion_r}
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,momentum=MOMENTUM, weight_decay=WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, WEIGHT)
    iou = test(models, dataloaders)
    print('Test iou: ', iou)