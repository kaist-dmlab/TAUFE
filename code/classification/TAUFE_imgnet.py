# Written by dongmin park @ KAIST
# dongminpark@kaist.ac.kr

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Custom
import time
import models.resnet64 as resnet64
import models.resnet128 as resnet128
import models.resnet224 as resnet224
from util.sampler import SubsetSequentialSampler
from util.utility import *

from util.arguments import parser
from util.load_dataset_imgnet import load_datasets

import pickle

# Seed
t = time.time
random.seed("0")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def train_epoch(models, criterion, optimizers, dataloaders, epoch, n_class, weight):
    models['backbone'].train()

    for i, (data, ood) in enumerate(zip(dataloaders['train_in'], dataloaders['train_ood'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        inputs_ood = ood[0].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

        # TAUFE
        scores_ood, features_ood = models['backbone'](inputs_ood)
        taufe_loss = torch.mean(features_ood ** 2)

        #if i == 0:
        #    print("m_backbone_loss: {}, TAUFE_loss: {}".format(m_backbone_loss, reg_loss))

        loss = m_backbone_loss + weight * taufe_loss

        loss.backward()
        optimizers['backbone'].step()

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, n_class,  weight):
    print('>> Train a Model.')

    logs = []
    best_acc = 0
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, n_class, weight)

        acc = test(models, dataloaders)
        print('Epoch: {}, Test acc: {}'.format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
        logs.append([epoch, acc])
    print('>> Finished.')
    print('Best Test acc: {}'.format(best_acc))

def test(models, dataloaders):
    models['backbone'].eval()

    total = 0
    correct = 0

    f = nn.Softmax(dim=1)
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

def test_v2(models, dataloaders):
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
            in_scores += to_np(-smax.data)

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
            ood_scores += to_np(-smax.data)

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
    #auroc2, aupr2, fpr2 = get_measures(ood_scores_e[:len(in_scores)], in_scores_e)

    acc = 100 * correct / total
    return acc, auroc, aupr, fpr #, auroc2, aupr2, fpr2

if __name__ == '__main__':
    # arguments
    args = parser.parse_args()
    print("args: ", args)

    resolution = 224 #64, 128
    train_in, test_in, train_ood, test_ood = load_datasets(args.in_data_name, args.ood_data_name, resolution)

    # save as pickle
    with open('../../data/Imgnet224/imgnet10_r224_train_in.txt', 'wb') as f:
        pickle.dump(train_in, f)
    with open('../../data/Imgnet224/imgnet990_r224_train_ood.txt', 'wb') as f:
        pickle.dump(train_ood, f)
    with open('../../data/Imgnet224/imgnet10_r224_test.txt', 'wb') as f:
        pickle.dump(test_in, f)
    with open('../../data/Imgnet224/places365_r224_test.txt', 'wb') as f:
        pickle.dump(test_ood, f)
    '''
    # read from pickle
    with open('../../data/Imgnet224/imgnet10_r224_train_in.txt', 'rb') as f:
        train_in = pickle.load(f)
    with open('../../data/Imgnet224/imgnet990_r224_train_ood.txt', 'rb') as f:
        train_ood = pickle.load(f)
    with open('../../data/Imgnet224/imgnet10_r224_test.txt', 'rb') as f:
        test_in = pickle.load(f)
    with open('../../data/Imgnet224/places365_r224_test.txt', 'rb') as f:
        test_ood = pickle.load(f)
    '''

    # Initialize a labeled dataset by randomly sampling
    indices_in = list(range(len(train_in)))
    random.shuffle(indices_in)
    labeled_set_in = indices_in[:args.n_samples]

    train_in_loader = DataLoader(train_in, batch_size=int(args.batch_size/2),
                                 sampler=SubsetRandomSampler(labeled_set_in),
                                 pin_memory=True)

    train_ood_loader = DataLoader(train_ood, batch_size=int(args.batch_size/2),
                                  pin_memory=True)

    print("len(train_in_loader), len(train_ood_loader):", len(train_in_loader), len(train_ood_loader))

    test_in_loader = DataLoader(test_in, batch_size=args.test_batch_size)
    test_ood_loader = DataLoader(test_ood, batch_size=args.test_batch_size)

    dataloaders = {'train_in': train_in_loader, 'train_ood': train_ood_loader,
                   'test_in': test_in_loader, 'test_ood': test_ood_loader}

    # Model
    resnet50 = resnet224.ResNet18(num_classes=args.n_class).cuda()
    resnet50 = nn.DataParallel(resnet50)
    models = {'backbone': resnet50}
    torch.backends.cudnn.benchmark = False

    # optimizer and scheduler initialization
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.wdecay)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestone)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, args.epochs, args.n_class, args.taufe_weight)
    acc, auroc, aupr, fpr = test_v2(models, dataloaders)
    print('Final Test acc: {}, auroc: {}, aupr: {}, fpr: {}'.format(acc, auroc, aupr, fpr))