# Written by dongmin park @ KAIST
# dongminpark@kaist.ac.kr

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

# Custom
import models.resnet224_bbox as resnet
from util.utility import *
from util.arguments import parser
from util.load_datasets import load_datasets

# Seed
random.seed("0")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

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

    dist_diag = torch.cat((a, b, c, d), 1)

    dist_diag_min = torch.max(dist_diag, 1).values

    return torch.mean(dist_center / dist_diag_min)

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, weight, loss_type):
    models['backbone'].train()

    for (data, ood) in zip(dataloaders['train_in'],dataloaders['train_ood']):
        inputs = data[0].type(torch.FloatTensor).cuda()
        bboxs = data[2].type(torch.FloatTensor).cuda()

        inputs_ood = ood[0].cuda()

        optimizers['backbone'].zero_grad()

        out_c, out_r, features = models['backbone'](inputs)

        # Losses
        coord_loss_r = criterion["regression"](out_r, bboxs)

        if loss_type == 'L1':
            target_loss = coord_loss_r
        elif loss_type == 'L1-IoU':
            iou_loss = IoULoss(out_r, bboxs)
            target_loss = coord_loss_r + 0.4 * iou_loss
        if loss_type == 'D-IoU':
            iou_loss = IoULoss(out_r, bboxs)
            diou_loss = DIoULoss(out_r, bboxs)
            target_loss = coord_loss_r + 0.4 * iou_loss + 0.4 * diou_loss

        # TAUFE
        out_c_ood, out_r_ood, features_ood = models['backbone'](inputs_ood)
        taufe_loss = torch.mean(features_ood ** 2)

        loss = target_loss + weight*taufe_loss

        loss.backward()
        optimizers['backbone'].step()

        del data

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, weight, loss_type):
    print('>> Train a Model.')
    logs = []
    best_iou = 0
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, weight, loss_type)

        acc, iou = test(models, dataloaders)
        best_iou = max(best_iou, iou)
        print('epoch: {}, iou: {}'.format(epoch, iou))

        # save logs
        logs.append([epoch, iou])
        # np.savetxt('../logs_taufe.txt', logs, delimiter=',', fmt="%.4f")

    print('>> Finished.')
    print('Best Test iou: {}'.format(best_iou))

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

    # 4. calculate the overlaps between pred_box and gt_box
    ious = inters / unis

    #print(ious)

    return torch.mean(ious)

def test(models, dataloaders):
    models['backbone'].eval()

    total = 0
    correct = 0
    total_iou = 0

    f = nn.Softmax(dim=1)
    to_np = lambda x: x.data.cpu().tolist()
    with torch.no_grad():
        for (inputs, labels, bbox, id) in dataloaders['test_in']:

            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.cuda()
            bbox = bbox.type(torch.FloatTensor).cuda()

            out_c, out_r, features = models['backbone'](inputs)

            smax, preds = torch.max(f(out_c.data), 1)
            pred_bbox = out_r.data

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            batch_iou = get_iou(pred_bbox, bbox)
            total_iou += batch_iou.item()

    acc = 100 * correct / total
    total_iou = total_iou/len(dataloaders['test_in'])

    return acc, total_iou


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

# AL Main
if __name__ == '__main__':
    args = parser.parse_args()
    print("args: ", args)

    # Load Data
    train_in, test_in, train_ood = load_datasets(args.in_data_name, args.ood_data_name, args.n_shots)

    indices_in = list(range(len(train_in)))
    random.shuffle(indices_in)
    #print("# of train_in samples", len(train_in))

    indices_ood = list(range(len(train_ood)))
    random.shuffle(indices_ood)
    labeled_set_ood = indices_ood[:len(train_in)]
    #print("# of train_ood samples", len(train_ood))

    train_in_loader = DataLoader(train_in, batch_size=int(args.batch_size/2),
                                 sampler=SubsetRandomSampler(indices_in),
                                 pin_memory=True)

    train_ood_loader = DataLoader(train_ood, batch_size=int(args.batch_size/2),
                                  pin_memory=True)

    test_in_loader = DataLoader(test_in, batch_size=args.test_batch_size)

    dataloaders = {'train_in': train_in_loader, 'train_ood': train_ood_loader,
                   'test_in': test_in_loader}

    # Model
    model = resnet.ResNet50(num_classes=args.n_class).cuda()
    models = {'backbone': model}
    torch.backends.cudnn.benchmark = False

    # Loss, criterion and scheduler (re)initialization
    criterion_c = nn.CrossEntropyLoss(reduction='mean')
    # criterion_r = nn.MSELoss(reduction='mean')
    criterion_r = nn.SmoothL1Loss(reduction='mean').cuda()
    criterion = {"classification": criterion_c, "regression": criterion_r}
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr, momentum=args.momentum,
                               weight_decay=args.wdecay)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestone)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, args.epochs, args.taufe_weight, args.loss_type)
    iou = test(models, dataloaders)
    print('Final Test iou: ', iou)