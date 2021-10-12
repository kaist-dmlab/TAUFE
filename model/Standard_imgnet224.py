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
# import visdom
from tqdm import tqdm

# Custom
import models.resnet224 as resnet
from config import *
from util.utility import *
from util.load_Imgnet224 import Imgnet224


# Seed
random.seed("Anonymized")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##
# Load Data
Imgnet_normalize = T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

imgnet224_transform = T.Compose([T.ToPILImage(),T.RandomResizedCrop(size=224),T.RandomHorizontalFlip(),T.ToTensor(),Imgnet_normalize])
imgnet224_transform_test = T.Compose([T.ToPILImage(),T.Resize(256),T.CenterCrop(size=224),T.ToTensor(),Imgnet_normalize])

place365_transform = T.Compose([T.RandomCrop(size=224, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),Imgnet_normalize])

in_class = np.arange(1000)
np.random.shuffle(in_class)
in_class = in_class[:10]
print("in_class: ", in_class)

# Training set, Divide Imgnet into Imgnet10 (in-distribution) vs Imgnet990 (ood)
imgnet224_train_in = Imgnet224(ood = False, train=True, transform=imgnet224_transform, in_class=in_class)
imgnet224_train_ood = Imgnet224(ood = True, train=True, transform=imgnet224_transform, in_class=in_class)

# Test set, Imgnet10
imgnet224_test = Imgnet224(ood = False, train=False, transform=imgnet224_transform_test, in_class=in_class)

# Place365 (ood)
folder = '/data/anonymized/places365/places365_standard/'
#traindir = os.path.join(folder, 'train')
valdir = os.path.join(folder, 'val')
place365_train_ood = torchvision.datasets.ImageFolder(valdir,transform=place365_transform)

def train_epoch(models, criterion, optimizers, dataloaders, epoch):
    models['backbone'].train()

    for data in tqdm(dataloaders['train_in'], leave=False, total=len(dataloaders['train_in'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

        loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs):
    print('>> Train a Model.')
    logs = []
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch)

        if epoch % 5 == 4:
            acc = test(models, dataloaders)
            print('Epoch: {}, Test acc: {}'.format(epoch, acc))

            # save logs
            logs.append([epoch, acc])
    #np.savetxt('../logs_sup/Imgnet224_10_990_Standard_n5000_per_epoch_v1.txt',logs, delimiter=',', fmt="%.4f")
    print('>> Finished.')

def test(models, dataloaders):
    models['backbone'].eval()

    total = 0
    correct = 0

    f = nn.Softmax(dim=1)
    to_np = lambda x: x.data.cpu().tolist()
    with torch.no_grad():
        for (inputs, labels) in dataloaders['test_in']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outs, features = models['backbone'](inputs)
            smax, preds = torch.max(f(outs.data), 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    return acc

def get_stratified_indices(labels, n):
    C = max(labels)+1 #10
    sampled_idx = []
    for c in range(C):
        idx = []
        for i, l in enumerate(labels):
            if l == c:
                idx+=[i]
        random.shuffle(idx)
        sampled_idx+=idx[:n]
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

train_in, train_ood = imgnet224_train_in, imgnet224_train_ood
test_in, test_ood = imgnet224_test, imgnet224_train_ood

# AL Main
if __name__ == '__main__':
    train_in_y = train_in.label
    #print(min(test_in.label), max(test_in.label))
    labeled_set_in = get_stratified_indices(train_in_y, SAMPLE_PER_CLASS)
    #print(len(labeled_set_in))

    indices_ood = list(range(len(train_ood.label)))
    random.shuffle(indices_ood)
    labeled_set_ood = indices_ood[:len(train_in)]

    train_in_loader = DataLoader(train_in, batch_size=BATCH,
                                 sampler=SubsetRandomSampler(labeled_set_in),
                                 pin_memory=True)

    train_ood_loader = DataLoader(train_ood, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set_ood),
                                  pin_memory=True)

    test_in_loader = DataLoader(test_in, batch_size=BATCH)
    test_ood_loader = DataLoader(test_ood, batch_size=BATCH)

    dataloaders = {'train_in': train_in_loader, 'train_ood': train_ood_loader,
                   'test_in': test_in_loader, 'test_ood': test_ood_loader}

    # Model
    resnet18 = resnet.ResNet34(num_classes=NUM_CLASS).cuda()
    models = {'backbone': resnet18}
    torch.backends.cudnn.benchmark = False

    # Loss, criterion and scheduler (re)initialization
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                               momentum=MOMENTUM, weight_decay=WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, EPOCH)
    acc = test(models, dataloaders)
    print('Final Test acc: ', acc)