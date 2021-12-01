# Written by dongmin park @ KAIST
# dongminpark@kaist.ac.kr

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import models.resnet as resnet
from util.sampler import SubsetSequentialSampler
from util.utility import *

from util.arguments import parser
from util.load_dataset import load_datasets
from util.load_80mTiny import TinyImages

# Seed
t = time.time
random.seed("0")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

criterion_taufe = nn.SmoothL1Loss(reduction='mean').cuda()
def train_epoch(models, criterion, optimizers, dataloaders, epoch, n_class):
    models['backbone'].train()

    for i, data in enumerate(dataloaders['train_in']):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        loss = torch.sum(target_loss) / target_loss.size(0)

        loss.backward()
        optimizers['backbone'].step()

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, n_class):
    print('>> Train a Model.')

    best_acc = 0
    logs = []
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, n_class)

        acc = test(models, dataloaders)
        #print('Epoch: {}, Test acc: {}'.format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
        logs.append([epoch, acc])
    print('>> Finished.')
    print("Best Test acc: {}".format(best_acc))

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

if __name__ == '__main__':
    # arguments
    args = parser.parse_args()
    print("args: ", args)

    with_index = False
    train_in, test_in, _, _ = load_datasets(args.in_data_name, args.ood_data_name, with_index)

    # Initialize a labeled dataset by randomly sampling
    indices_in = list(range(args.n_train))
    random.shuffle(indices_in)
    labeled_set_in = indices_in[:args.n_samples]
    print(labeled_set_in[:10])

    train_in_loader = DataLoader(train_in, batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(labeled_set_in),
                                 pin_memory=True)

    test_in_loader = DataLoader(test_in, batch_size=args.test_batch_size)

    dataloaders = {'train_in': train_in_loader, 'test_in': test_in_loader}

    # Model
    resnet18 = resnet.ResNet18(num_classes=args.n_class).cuda()
    resnet18 = nn.DataParallel(resnet18)
    models = {'backbone': resnet18}
    torch.backends.cudnn.benchmark = False

    # optimizer and scheduler initialization
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.wdecay)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestone)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, args.epochs, args.n_class)
    acc = test(models, dataloaders)
    print('Final Test acc: {}'.format(acc))