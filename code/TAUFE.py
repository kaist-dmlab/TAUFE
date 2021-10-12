import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN

# Utils
from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from util.sampler import SubsetSequentialSampler
from util.utility import *
from util.load_80mTiny import TinyImages

# Seed
random.seed("Anonymized")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##
# Load Data
CIFAR10_normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_normalize = T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),CIFAR100_normalize])#CIFAR100_normalize
test_transform = T.Compose([T.ToTensor(),CIFAR100_normalize])
tiny_transform = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),CIFAR100_normalize])
svhn_transform = T.Compose([T.CenterCrop(size=32),T.ToTensor(),CIFAR100_normalize])

# CIFAR10
#cifar10_train = CIFAR10('../data/cifar10', train=True, download=True, transform=train_transform)
#cifar10_test = CIFAR10('../data/cifar10', train=False, download=True, transform=test_transform)

# CIFAR100
cifar100_train = CIFAR100('../data/cifar100', train=True, download=True, transform=train_transform)
cifar100_test = CIFAR100('../data/cifar100', train=False, download=True, transform=test_transform)

# 80mTiny
tiny_ood = TinyImages(transform=tiny_transform)

# SVHN
svhn = SVHN('../data/svhn', download=True, transform=svhn_transform)

def train_epoch(models, criterion, optimizers, dataloaders, epoch, weight):
    models['backbone'].train()

    for (data, ood) in zip(tqdm(dataloaders['train_in'], leave=False, total=len(dataloaders['train_in'])),
                           tqdm(dataloaders['train_ood'], leave=False, total=len(dataloaders['train_ood']))):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        inputs_ood = ood[0].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

        # for Blackhole
        scores_ood, features_ood = models['backbone'](inputs_ood)

        #if epoch > EPOCH_DETACH:
        #    After 120 epochs, stop the gradient from the loss prediction module propagated to the target code.
        #    features_ood = features_ood.detach()

        #Blackhole_loss = torch.mean(torch.mean(features_ood ** 2, dim=-1))
        #Blackhole_loss = torch.mean(features_ood ** 2)
        Blackhole_loss = torch.mean(features_ood)

        loss = m_backbone_loss + weight * Blackhole_loss
        #print(m_backbone_loss, Blackhole_loss)

        loss.backward()
        optimizers['backbone'].step()

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, weight):
    print('>> Train a Model.')
    logs = []
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, weight)

        acc, auroc, aupr, fpr, auroc2, aupr2, fpr2 = test_new(models, dataloaders)
        print('Epoch: {}, Test acc: {}, auroc_u: {}, aupr_u: {}, fpr_u: {}, auroc_e: {}, aupr_e: {}, fpr_e: {}'.format(epoch, acc, auroc, aupr, fpr, auroc2, aupr2, fpr2))
        # save logs
        logs.append([epoch, acc, auroc, aupr, fpr, auroc2, aupr2, fpr2])
    np.savetxt('../logs_rebuttal/cifar10_80mTiny_taufe_n2500_v1.txt', logs, delimiter=',', fmt="%.4f")
    print('>> Finished.')

#
def test(models, dataloaders):
    models['backbone'].eval()

    total = 0
    correct = 0
    uncertainty_in = []
    energy_in = []

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

#
def test_new(models, dataloaders):
    models['backbone'].eval()

    total = 0
    correct = 0
    in_scores = []
    in_scores_e = []

    f = nn.Softmax(dim=1)
    to_np = lambda x: x.data.cpu().tolist()
    with torch.no_grad():
        for (inputs, labels) in dataloaders['test_in']:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outs, _ = models['backbone'](inputs)
            smax, preds = torch.max(f(outs.data), 1)

            #accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # uncertainty
            in_scores += to_np(-smax)

            # energy
            energy_in = -torch.logsumexp(outs.data, dim=1)
            in_scores_e += to_np(energy_in)

    ood_scores = []
    ood_scores_e = []
    with torch.no_grad():
        for (inputs, labels) in dataloaders['test_ood']:
            inputs = inputs.cuda()
            outs, _ = models['backbone'](inputs)
            smax, _ = torch.max(f(outs.data), 1)

            # uncertainty
            ood_scores += to_np(-smax)

            # energy
            energy_ood = -torch.logsumexp(outs.data, dim=1)
            ood_scores_e += to_np(energy_ood)

            if len(ood_scores) >= len(in_scores):
                break
    # print(" (in_scores), (ood_scores): ")
    # print(in_scores[:20], ood_scores[:20])
    # in_scores = [1 for i in range(10)]
    # ood_scores = [0 for i in range(10)]

    auroc, aupr, fpr = get_measures(ood_scores[:len(in_scores)], in_scores)
    auroc2, aupr2, fpr2 = get_measures(ood_scores_e[:len(in_scores)], in_scores_e)

    acc = 100 * correct / total
    return acc, auroc, aupr, fpr, auroc2, aupr2, fpr2

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

train_in, train_ood = cifar100_train, tiny_ood # cifar100_train
test_in, test_ood = cifar100_test, svhn

##
# AL Main
if __name__ == '__main__':
    train_in_y = train_in.targets
    labeled_set_in = get_stratified_indices(train_in_y, SAMPLE_PER_CLASS)

    indices_ood = list(range(NUM_TRAIN_OOD))
    random.shuffle(indices_ood)
    labeled_set_ood = indices_ood[:NUM_CLASS*SAMPLE_PER_CLASS]

    train_in_loader = DataLoader(train_in, batch_size=BATCH,
                                 sampler=SubsetRandomSampler(labeled_set_in),
                                 pin_memory=True)

    train_ood_loader = DataLoader(train_ood, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set_ood),
                                  pin_memory=True)
    offset = np.random.randint(len(train_ood_loader.dataset))
    train_ood_loader.dataset.offset = offset
\
    test_in_loader = DataLoader(test_in, batch_size=BATCH)
    test_ood_loader = DataLoader(test_ood, batch_size=BATCH)

    dataloaders = {'train_in': train_in_loader, 'train_ood': train_ood_loader,
                   'test_in': test_in_loader, 'test_ood': test_ood_loader}

    # Model
    resnet18 = resnet.ResNet18(num_classes=NUM_CLASS).cuda()
    models = {'backbone': resnet18}
    torch.backends.cudnn.benchmark = False

    # Loss, criterion and scheduler initialization
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                               momentum=MOMENTUM, weight_decay=WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, WEIGHT)
    acc = test_new(models, dataloaders)
    print('Final Test acc: ', acc)