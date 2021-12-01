import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Undesirable Features Deactivation')
parser.add_argument('--in-data-name', type=str, default='cifar10') # cifar10, cifar100, imgnet10
parser.add_argument('--ood-data-name', type=str, default='lsun') # lsun, 80mTiny, svhn, imgnet990, places365
parser.add_argument('--n-train', type=int, default=50000, metavar='N',
                    help='# of training samples')
parser.add_argument('--n-class', type=int, default=10, metavar='N',
                    help='# of classes')
parser.add_argument('--n-samples', type=int, default=500, metavar='N',
                    help='# of samples per class')
parser.add_argument('--taufe-weight', type=float, default=0.1, metavar='ZeroReg',
                    help='taufe weight (default: 0.1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--wdecay', type=float, default=5e-4, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--milestone', type=list, default=[100, 150])
args = parser.parse_args()