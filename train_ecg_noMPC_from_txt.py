#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = " "

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import h5py
import numpy as np
from torchsummary import summary
from sklearn.metrics import r2_score, mean_squared_error
import math
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--is_comet", help="Set is Comet", action='store_true')
parser.add_argument("-m", "--model_type", help="model name(shallow, normal, ann, mpc, cnn2d)", type=str, default='cnnavg')
parser.add_argument("-mpc", "--mpc", help="shallow model", action='store_true')
parser.add_argument("-lt", "--loss_type", help="use sgd as optimizer", type=str, default='adam')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=3)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=1e-2)
parser.add_argument("-eps", "--eps", help="Set epsilon of adam", type=float, default=1e-7)
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-sc", "--scaler", help="Set random seed", type=str, default='max100')
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=80)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=20)
parser.add_argument("-mom", "--momentum", help="Set momentum", type=float, default=0.9)

args = parser.parse_args()

max_x = 0
if args.is_comet:
    experiment = Experiment(api_key="eIskxE43gdgwOiTV27APVUQtB", project_name='secure-ecg', workspace="wonderit")
else:
    experiment = None

def scale(arr, m, s):
    arr = arr - m
    arr = arr / (s + 1e-7)
    return arr


def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m
    return arr


def scale_minmax(arr, min, max):
    arr = (arr - min) / (max - min)
    return arr


def scale_maxabs(arr, maxabs):
    arr = arr / maxabs
    return arr


def scale_robust(arr, q1, q3):
    print('q1 : ', q1, 'q1 : ', q3)
    arr = (arr - q1) / (q3-q1)
    return arr


def return_maxabs_min_max(arr, q1, q3):
    print('q1 : ', q1, 'q1 : ', q3)
    arr = (arr - q1) / (q3-q1)
    return arr


# def rescale_minmax(arr, min, max):
#     arr = arr - m
#     arr = arr / (s + 1e-7)
#     return arr
#

# 5500 criteria
# MEAN = 59.3
# STD = 10.6
# mean_x = 1.547
# std_x = 156.820


# 5500 new criteria
# MEAN = 61.6
# STD = 9.8
# 5500 new criteria !!!
# MEAN = 61.4
# STD = 9.7

MEAN = 61.9
STD = 10.8
#
# MEAN = 62.0
# STD = 11.0
# mean_x = 1.693
# std_x = 155.617

# x mean, std:  1.693 155.617
# y mean, std:  61.6 9.8


# 11000 criteria
# MEAN = 61.5
# STD = 9.9
# mean_x = 1.784
# std_x = 154.998

# 22000 criteria
# MEAN = 61.93
# STD = 10.91
# mean_x = 1.733
# std_x = 156.279
# mean_x = 1.914
# std_x = 156.413

_ = torch.manual_seed(args.seed)

result_path = os.path.join('result_torch', 'text_{}scaler_{}_{}_eps{}_ep{}_bs{}_{}-{}_lr{}_mom{}'.format(
    args.scaler,
    args.model_type,
    args.loss_type,
    args.eps,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr,
    args.momentum
))



batches = 5000 / args.batch_size
log_batches = int(batches / args.log_interval)

DATAPATH = '../data/ecg/text_demo_5500'
train_file_suffix = 'train'
test_file_suffix = 'test'

file_name_train_x = 'X{}'.format(train_file_suffix)
file_name_train_y = 'y{}'.format(train_file_suffix)
file_name_test_x = 'X{}'.format(test_file_suffix)
file_name_test_y = 'y{}'.format(test_file_suffix)

print('Converting to TorchDataset...')

train_x = np.loadtxt('{}/{}'.format(DATAPATH, file_name_train_x), delimiter=',')
test_x = np.loadtxt('{}/{}'.format(DATAPATH, file_name_test_x), delimiter=',')

total_x = np.vstack((train_x, test_x))

train_y = np.loadtxt('{}/{}'.format(DATAPATH, file_name_train_y), delimiter=',')
test_y = np.loadtxt('{}/{}'.format(DATAPATH, file_name_test_y), delimiter=',')

train_x = train_x.reshape(train_x.shape[0], 3, 500)
test_x = test_x.reshape(test_x.shape[0], 3, 500)


# train_y = scale(train_y, MEAN, STD)
# test_y = scale(test_y, MEAN, STD)
#
# print('train_x m, s: ', train_x.mean(), train_x.std())
#
# if args.scaler == 'minmax':
#     train_x = scale_minmax(train_x, total_x.min(), total_x.max())
#     test_x = scale_minmax(test_x, total_x.min(), total_x.max())
# elif args.scaler == 'maxabs':
#     train_x = scale_maxabs(train_x, np.max(np.abs(total_x)))
#     test_x = scale_maxabs(test_x, np.max(np.abs(total_x)))
# elif args.scaler == 'robust':
#     train_x = scale_robust(train_x, np.quantile(total_x, 0.25), np.quantile(total_x, 0.75))
#     test_x = scale_robust(test_x, np.quantile(total_x, 0.25), np.quantile(total_x, 0.75))
# elif args.scaler == 'standard':
#     train_x = scale(train_x, mean_x, std_x)
#     test_x = scale(test_x, mean_x, std_x)

class ECGDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)

# train_dataset, test_dataset = torch.utils.data.random_split(data, [args.n_train_items, args.n_test_items])
train_dataset = ECGDataset(train_x, train_y, transform=False)
test_dataset = ECGDataset(test_x, test_y, transform=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print('Torch Dataset Train/Test split finished...')

print('Data Sharing complete')

class CANN(nn.Module):
    def __init__(self):
        super(CANN, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        # self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        # self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=self.padding_size)
        # self.fc1 = nn.Linear(2856, 16)     # 4 layer of CNN
        # self.fc1 = nn.Linear(2892, 16)     # 3 layer of CNN
        self.fc1 = nn.Linear(2928, 16)     # 2 layer of CNN
        # self.fc1 = nn.Linear(2964, 16)     # 1 layer of CNN
        # self.fc1 = nn.Linear(150, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        # x = self.avgpool1(x)  # 32
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv2(x))
        # y = self.avgpool1(y)
        # x = F.relu(self.conv2(x))
        # y = self.avgpool1(y)
        # x = self.conv3(x)
        y = F.relu(self.conv2(x))
        # y = self.avgpool1(y)
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class CNNAVG(nn.Module):
    def __init__(self):
        super(CNNAVG, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(372, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.max_x = max_x

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = self.avgpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        y = F.relu(self.conv3(x))

        y = self.avgpool3(y)
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc1(y))

        y = F.relu(self.fc2(y))

        y = self.fc3(y)
        return y

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.channel_size = 3
        self.fc1 = nn.Linear(self.channel_size * 500, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(self.channel_size * 500, 16)
        self.fc5 = nn.Linear(16, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        # x = F.relu(self.conv1(x))  # 32
        # x = F.relu(self.conv2(x))  # 32
        # x = self.avgpool1(x)  # 32
        #
        # # y = F.relu(self.conv2(x))
        # x1 = F.relu(self.conv2(x))
        #
        # # Comment Temp
        # c1 = torch.cat((x, x1), dim=1)  # 64
        # x2 = F.relu(self.conv3(c1))  # 32
        # y = torch.cat((x, x1, x2), dim=1)  # 96
        # # downsizing
        # y = F.relu(self.conv10(y))  # 24
        # # y = self.avgpool1(y)
        # # y = F.relu(self.conv10(y))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_forMPC(nn.Module):
    def __init__(self):
        super(CNN_forMPC, self).__init__()
        self.kernel_size = 7
        self.padding_size = 3
        self.channel_size = 6
        # self.channel_size = 32
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv22 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size * 3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv6 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv7 = nn.Conv1d(self.channel_size * 3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv8 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv9 = nn.Conv1d(self.channel_size*2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv10 = nn.Conv1d(self.channel_size * 3, 1, kernel_size=1)
        self.fc1 = nn.Linear(125, 16)
        # self.fc1 = nn.Linear(2976, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = F.relu(self.conv2(x))  # 32
        x = self.avgpool1(x)  # 32

        # y = F.relu(self.conv2(x))
        x1 = F.relu(self.conv22(x))

        # Comment Temp
        c1 = torch.cat((x, x1), dim=1)  # 64
        x2 = F.relu(self.conv3(c1))  # 32
        y = torch.cat((x, x1, x2), dim=1)  # 96
        # downsizing
        y = F.relu(self.conv4(y))  # 24
        y = self.avgpool2(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3, x4), dim=1)
        # Comment Temp

        # y = F.relu(self.conv7(y))
        # y = self.avgpool1(y)
        #
        # x5 = F.relu(self.conv8(y))
        # c3 = torch.cat((y, x5), dim=1)
        # x6 = F.relu(self.conv9(c3))
        # y = torch.cat((y, x5, x6), dim=1)

        y = F.relu(self.conv10(y))

        # print('shape before flatten', y.shape)
        # Flatten
        y = y.view(y.shape[0], -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y


class CNN2D_SHALLOW(nn.Module):
    def __init__(self):
        super(CNN2D_SHALLOW, self).__init__()
        self.kernel_size = (1, 7)
        self.padding_size = 0
        self.channel_size = 6
        self.conv1 = nn.Conv2d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv2d(6, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.fc1 = nn.Linear(2964, 16)
        # self.fc1 = nn.Linear(2976, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        print('0', x.shape)
        x = F.relu(self.conv1(x))
        y = F.relu(self.conv2(x))  # 32
        print('1', y.shape)
        y = x.view(y.shape[0], -1)
        print('1', y.shape) # 32
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y


class CNN2D_forMPC(nn.Module):
    def __init__(self):
        super(CNN2D_forMPC, self).__init__()
        self.kernel_size = (1, 7)
        self.padding_size = (1, 3)
        self.channel_size = 6
        # self.channel_size = 32
        self.conv1 = nn.Conv2d(12, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv2d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv2d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv2d(self.channel_size * 3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv5 = nn.Conv2d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv6 = nn.Conv2d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv10 = nn.Conv2d(self.channel_size * 3, 1, kernel_size=1)
        self.fc1 = nn.Linear(127, 16)
        # self.fc1 = nn.Linear(2976, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        # print('0', x.shape)
        x = F.relu(self.conv1(x))  # 32
        # print('1', x.shape)
        x = F.relu(self.conv2(x))  # 32
        # print('2', x.shape)

        x = self.avgpool1(x)  # 32
        x1 = F.relu(self.conv2(x))
        # print('3', x.shape)
        c1 = torch.cat((x, x1[:, :, :x.shape[2]]), dim=1)  # 64
        # print('4', x.shape)
        x2 = F.relu(self.conv3(c1))  # 32
        # print('4', x.shape)
        y = torch.cat((x, x1[:, :, :x.shape[2]], x2[:, :, :x.shape[2]]), dim=1)  # 96
        # downsizing
        y = F.relu(self.conv4(y))  # 24
        y = self.avgpool1(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3[:, :, :y.shape[2]]), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3[:, :, :y.shape[2]], x4[:, :, :y.shape[2]]), dim=1)

        y = F.relu(self.conv10(y))
        # print('7', y.shape)
        y = y.view(y.shape[0], -1)
        # print('8', y.shape)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y

class ML4CVD_shallow(nn.Module):
    def __init__(self):
        super(ML4CVD_shallow, self).__init__()
        self.kernel_size = 7
        self.padding_size = 3
        self.channel_size = 32
        self.conv1 = nn.Conv1d(12, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size * 3, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(24, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv6 = nn.Conv1d(48, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv7 = nn.Conv1d(72, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv8 = nn.Conv1d(16, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv9 = nn.Conv1d(32, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.fc1 = nn.Linear(2976, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 32
        x = F.relu(self.conv2(x)) # 32
        x = self.avgpool1(x) # 32
        x1 = F.relu(self.conv2(x))
        c1 = torch.cat((x, x1), dim=1) # 64
        x2 = F.relu(self.conv3(c1)) # 32
        y = torch.cat((x, x1, x2), dim=1) # 96
        # downsizing
        y = F.relu(self.conv4(y)) # 24
        y = self.avgpool1(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3, x4), dim=1)

        y = F.relu(self.conv7(y))
        y = self.avgpool1(y)

        x5 = F.relu(self.conv8(y))
        c3 = torch.cat((y, x5), dim=1)
        x6 = F.relu(self.conv9(c3))
        y = torch.cat((y, x5, x6), dim=1)

        # Flatten
        y = y.view(y.size(0), -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y


class ML4CVD(nn.Module):
    def __init__(self):
        super(ML4CVD, self).__init__()
        self.kernel_size = 71
        self.padding_size = 35
        self.channel_size = 32
        self.conv1 = nn.Conv1d(12, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size * 3, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(24, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv6 = nn.Conv1d(48, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv7 = nn.Conv1d(72, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv8 = nn.Conv1d(16, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv9 = nn.Conv1d(32, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.fc1 = nn.Linear(30000, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = F.relu(self.conv2(x))  # 32
        x = self.avgpool1(x)  # 32
        x1 = F.relu(self.conv2(x))
        c1 = torch.cat((x, x1), dim=1)  # 64
        x2 = F.relu(self.conv3(c1))  # 32
        y = torch.cat((x, x1, x2), dim=1)  # 96
        # downsizing
        y = F.relu(self.conv4(y))  # 24
        y = self.avgpool1(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3, x4), dim=1)

        y = F.relu(self.conv7(y))
        y = self.avgpool1(y)

        x5 = F.relu(self.conv8(y))
        c3 = torch.cat((y, x5), dim=1)
        x6 = F.relu(self.conv9(c3))
        y = torch.cat((y, x5, x6), dim=1)

        # Flatten
        y = y.view(y.size(0), -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y



def report_scores(X, y, trained_model):
    y_true = []
    y_pred = []
    # y_score = []

    # DON'T NORMALIZE X
    # X = scale(X, mean_x, std_x)
    # print('Example : X - ', X[0, 0:3], 'y - ', y[0])
    # print(X.shape, y.shape)

    # reshaped_X = X.reshape(X.shape[0], 3, 500)


    with torch.no_grad():
        scores = trained_model(torch.from_numpy(X).float())
        #
        # output rescale
        scores = rescale(scores, MEAN, STD)
        y = rescale(y, MEAN, STD)

        mse_loss = mean_squared_error(y, scores)

        y_true.extend(list(y))
        y_pred.extend(scores)

    return y_true, y_pred, mse_loss


def train(args, model, private_train_loader, optimizer, epoch, test_loader):
    training_step = log_batches * (epoch-1)
    model.train()
    data_count = 0
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(private_train_loader):  # <-- now it is a private dataset
        # if target.min() < 25 or target.max() > 140:
        #     continue

        start_time = time.time()

        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]

        # Reshape
        # output = output.view(-1)
        # target = target.view(-1)

        target = target.view(target.shape[0], 1)
        # r2 : 0.67 with smooth l1 loss
        # loss = F.smooth_l1_loss(output, target).sum() / batch_size

        # r2 : 0.7  with logcosh loss
        # loss = (torch.log(torch.cosh(output - target))).sum() / batch_size

        # r2 : 0.646 w mse loss
        loss = ((output - target) ** 2).sum() / batch_size
        # loss = ((output - target) ** 2).sum()

        loss.backward()

        optimizer.step()

        epoch_loss = loss.item()
        data_count += 1

        if batch_idx % args.log_interval == 0:
            # loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                       100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))

            if args.is_comet:
                training_step = training_step + 1
                y_true_train, y_pred_train, train_mse_loss = report_scores(train_x, train_y, model)
                _, train_r2 = r_squared_mse(y_true_train, y_pred_train, train_mse_loss)
                print('step : ', training_step)
                experiment.log_metric("train_mse", train_mse_loss, epoch=epoch, step=training_step)
                experiment.log_metric("train_r2", train_r2, epoch=epoch, step=training_step)
                # test during training
                test(args, model, test_loader, epoch, batch_idx, training_step)

def test(args, model, private_test_loader, epoch, batch=999, step=0):
    model.eval()
    test_loss = 0
    data_count = 0
    pred_list = []
    target_list = []
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()

            output = model(data)
            #
            # output rescale
            output = rescale(output, MEAN, STD)
            target = rescale(target, MEAN, STD)

            # Reshape
            # output = output.view(-1)
            # target = target.view(-1)

            target = target.view(target.shape[0], 1)

            test_loss += ((output - target) ** 2).sum()
            # test_loss += torch.log(torch.cosh(output - target)).sum()

            data_count += len(output)
            pred_list.extend(output.numpy())
            target_list.extend(target.numpy())
            # print('rmse:', torch.sqrt(((output - target) ** 2).sum() / args.batch_size))
            # print('r2score:', r2_score(target_list, pred_list))

    # test_loss = test_loss.get().float_precision()
    # print('Test set: Loss: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
    #            100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))
    print('\nTest set: Loss: avg MSE ({:.4f})\tTime: {:.3f}s'.format(test_loss / data_count, time.time() - start_time))


    # # output rescale
    # target_list = rescale(target_list, MEAN, STD)
    # pred_list = rescale(pred_list, MEAN, STD)

    rm, test_r2 = r_squared_mse(target_list, pred_list)

    scatter_plot(target_list, pred_list, epoch, rm, batch)


    if args.is_comet:
        experiment.log_metric("test_mse", test_loss / data_count, epoch=epoch, step=step)
        experiment.log_metric("test_r2", test_r2, epoch=epoch, step=step)

    # if epoch % args.log_interval == 0:
    #     scatter_plot(target_list, pred_list, epoch, rm, batch)

def scatter_plot(y_true, y_pred, epoch, message, batch):
    result = np.column_stack((y_true,y_pred))

    if not os.path.exists('{}/{}'.format(result_path, 'csv')):
        os.makedirs('{}/{}'.format(result_path, 'csv'))

    if not os.path.exists('{}/{}'.format(result_path, 'scatter')):
        os.makedirs('{}/{}'.format(result_path, 'scatter'))

    pd.DataFrame(result).to_csv("{}/csv/{}.csv".format(result_path, epoch), index=False)

    plt.scatter(y_pred, y_true, s=3)
    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    # plt.savefig("{}/scatter/{}.png".format(result_path, epoch))

    if args.is_comet:
        experiment.log_figure(figure=plt, figure_name='{}_{}.png'.format(epoch, batch))
    else:
        plt.savefig("{}/scatter/{}_{}.png".format(result_path, epoch, batch))
    plt.clf()
    # plt.show()


def r_squared_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred)
    # bounds_check = np.min(y_pred) > MIN_MOISTURE_BOUND
    # bounds_check = bounds_check&(np.max(y_pred) < MAX_MOISTURE_BOUND)

    print('Scoring - std', np.std(y_true), np.std(y_pred))
    print('Scoring - median', np.median(y_true), np.median(y_pred))
    print('Scoring - min', np.min(y_true), np.min(y_pred))
    print('Scoring - max', np.max(y_true), np.max(y_pred))
    print('Scoring - mean', np.mean(y_true), np.mean(y_pred))
    print('Scoring - MSE: ', mse, 'RMSE: ', math.sqrt(mse))
    print('Scoring - R2: ', r2)
    # print(y_pred)


    result_message = 'r2:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r2, mse, np.std(y_true), np.std(y_pred))
    return result_message, r2

def save_model(model, path):

    torch.save(model.state_dict(), path)

def transform_array(torch_array, ):
    p = torch_array.detach().numpy()
    print('p:', p.shape, p.ndim)
    # if conv1d
    if p.ndim == 3:
        p = p.reshape(p.shape[0], -1)
    p = np.transpose(p)
    p = np.round_(p, 7)
    return p

def save_model_to_txt(model, path, ep):

    conv1_weight = transform_array(model.conv1.weight)
    conv1_bias = transform_array(model.conv1.bias)

    conv2_weight = transform_array(model.conv2.weight)
    conv2_bias = transform_array(model.conv2.bias)


    conv3_weight = transform_array(model.conv3.weight)
    conv3_bias = transform_array(model.conv3.bias)

    fc1_weight = transform_array(model.fc1.weight)
    fc1_bias = transform_array(model.fc1.bias)

    fc2_weight = transform_array(model.fc2.weight)
    fc2_bias = transform_array(model.fc2.bias)

    fc3_weight = transform_array(model.fc3.weight)
    fc3_bias = transform_array(model.fc3.bias)

    np.savetxt('{}ecg_P1_{}_0_W0.bin'.format(path, ep), conv1_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b0.bin'.format(path, ep), conv1_bias, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_W1.bin'.format(path, ep), conv2_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b1.bin'.format(path, ep), conv2_bias, fmt='%1.7f')

    np.savetxt('{}ecg_P1_{}_0_W2.bin'.format(path, ep), conv3_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b2.bin'.format(path, ep), conv3_bias, fmt='%1.7f')

    np.savetxt('{}ecg_P1_{}_0_W3.bin'.format(path, ep), fc1_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b3.bin'.format(path, ep), fc1_bias, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_W4.bin'.format(path, ep), fc2_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b4.bin'.format(path, ep), fc2_bias, fmt='%1.7f')

    np.savetxt('{}ecg_P1_{}_0_W5.bin'.format(path, ep), fc3_weight, fmt='%1.7f')
    np.savetxt('{}ecg_P1_{}_0_b5.bin'.format(path, ep), fc3_bias, fmt='%1.7f')


if args.model_type in ['shallow', 'ann', 'cnn2d', 'cann', 'cnnavg']:

    if args.model_type == 'shallow':
        model = CNN_forMPC()
    elif args.model_type == 'cnn2d':
        model = CNN2D_SHALLOW()
    elif args.model_type == 'cann':
        model = CANN()
    elif args.model_type == 'cnnavg':
        model = CNNAVG()
    else:
        model = ANN()

    # if args.model_type == 'cnn2d':
    #     summary(model, input_size=(3, 500, 1), batch_size=args.batch_size)
    # else:
    #     summary(model, input_size=(3, 500), batch_size=args.batch_size)
else:
    model = ML4CVD()
    summary(model, input_size=(12, 5000), batch_size=args.batch_size)

print(model)

# save_model_to_txt(model, "{}/models/".format(result_path), 0)
# exit(0)
# model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
# for 12channel

# for 1 channel
# summary(model, input_size =(1, 12, 5000), batch_size=args.batch_size)
# exit()

if args.loss_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)  # 4.58
elif args.loss_type == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr)  # 4.58
elif args.loss_type == 'lbfgs':
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr)  # 4.58
elif args.loss_type == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 4.58
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

for epoch in range(1, args.epochs + 1):

    # Save model
    if not os.path.exists('{}/{}'.format(result_path, 'models')):
        os.makedirs('{}/{}'.format(result_path, 'models'))

    if epoch == 1:
        save_model_to_txt(model, "{}/models/".format(result_path), epoch-1)
    train(args, model, train_loader, optimizer, epoch, test_loader)
    test(args, model, test_loader, epoch, epoch * batches)
    if epoch % args.log_interval == 0:
        save_model(model, "{}/models/ep{}.h5".format(result_path, epoch))
        save_model_to_txt(model, "{}/models/".format(result_path), epoch-1)


