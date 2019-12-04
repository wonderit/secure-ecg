#!/usr/bin/env python3

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
import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
#
# class Arguments():
#     def __init__(self):
#         self.batch_size = 32
#         self.epochs = 1
#         self.lr = 1e-4   # 0.00002
#         self.seed = 1234
#         self.log_interval = 1  # Log info at each batch
#         self.precision_fractional = 3
#
#         # We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
#         self.n_train_items = 300
#         self.n_test_items = 30
#
# args = Arguments()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compressed", help="Compress ecg data", action='store_true')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=1)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=1e-4)
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=80)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=20)

args = parser.parse_args()

_ = torch.manual_seed(args.seed)

model_type = 'original'
if args.compressed:
    model_type = 'compressed'

result_path = 'result_torch/{}_ep{}_bs{}_{}:{}_lr{}'.format(
    model_type,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr
)

# import syft as sy  # import the Pysyft library
# hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning
#
# # simulation functions
# def connect_to_workers(n_workers):
#     return [
#         sy.VirtualWorker(hook, id=f"worker{i+1}")
#         for i in range(n_workers)
#     ]
# def connect_to_crypto_provider():
#     return sy.VirtualWorker(hook, id="crypto_provider")
#
# workers = connect_to_workers(n_workers=2)
# crypto_provider = connect_to_crypto_provider()


DATAPATH = '../data/ecg/raw/2019-11-19'
# DATA_LENGTH = 100
# BATCH_SIZE = 10
# TRAIN_RATIO = 0.8
ecg_key_string_list = [
    "strip_I",
    "strip_II",
    "strip_III",
    "strip_aVR",
    "strip_aVL",
    "strip_aVF",
    "strip_V1",
    "strip_V2",
    "strip_V3",
    "strip_V4",
    "strip_V5",
    "strip_V6",
]

hdf5_files = []
count = 0
for f in glob.glob("{}/*.hd5".format(DATAPATH)):
    count += 1
    if count > (args.n_train_items + args.n_test_items):
        break
    hdf5_files.append(f)

print('Data Loading finished (row:{})'.format(len(hdf5_files)))


class ECGDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = x.reshape([12, 5000])
            x = x.reshape([12, 12//12, 500, 5000 // 500]).mean(3).mean(1)
            # print('x', x.shape, x[0, :9])
            # plt.plot(x[4, :])
            # plt.show()
            # # x = s
            # # print('s', s.shape, s[0, :3])
            # plt.plot(x[4, :])
            # plt.show()
            # scaler = MinMaxScaler(feature_range=(-1, 1))
            # x = scaler.fit_transform(x.numpy())
            # x = torch.from_numpy(x)
            # #
            # # print('x', x.shape, x[0, :9])
            #
            #
            # plt.plot(x[4, :])
            # plt.show()

            # exit()

        return x, y

    def __len__(self):
        return len(self.data)

print('Converting to TorchDataset...')

x_all = []
y_all = []
for hdf_file in hdf5_files:
    f = h5py.File(hdf_file, 'r')
    y_all.append(f['continuous']['VentricularRate'][0])
    x_list = list()
    for (i, key) in enumerate(ecg_key_string_list):
        x = f['ecg_rest'][key][:]
        x_list.append(x)
    x_list = np.stack(x_list)
    x_list = x_list.reshape(12, -1)
    x_all.append(x_list)

x = np.asarray(x_all)
y = np.asarray(y_all)

print(x.shape, y.shape)

# print('x', x.shape, x[0, 0, :9])
# plt.plot(x[4, 4, :])
# plt.show()

# scaler = NDStandardScaler()
# x = scaler.fit_transform(x)
# x = torch.from_numpy(x)


# keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
# would matter if you wanted to normalize over a different axis.

# print('x', x.shape, x[0, 0, :9])
# plt.plot(x[4, 4, :])
# plt.show()

if args.compressed:
    data = ECGDataset(x, y, transform=True)
else:
    data = ECGDataset(x, y, transform=False) # 4.58
# train_size = int(TRAIN_RATIO * len(data))
# test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [args.n_train_items, args.n_test_items])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print('Torch Dataset Train/Test split finished...')

def get_private_data_loaders(precision_fractional, workers, crypto_provider):
    def one_hot_of(index_tensor):
        """
        Transform to one hot tensor

        Example:
            [0, 3, 9]
            =>
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

        """
        onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
        onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
        return onehot_tensor

    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        """
        return (
            tensor
                .fix_precision(precision_fractional=precision_fractional)
                .share(*workers, crypto_provider=crypto_provider, requires_grad=True)
        )

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True, transform=transformation),
    #     batch_size=args.batch_size
    # )

    private_train_loader = [
        (secret_share(data), secret_share(target))
        for i, (data, target) in enumerate(train_loader)
        if i < args.n_train_items / args.batch_size
    ]

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, download=True, transform=transformation),
    #     batch_size=args.test_batch_size
    # )

    private_test_loader = [
        (secret_share(data), secret_share(target.float()))
        for i, (data, target) in enumerate(test_loader)
        if i < args.n_test_items / args.batch_size
    ]

    return private_train_loader, private_test_loader

#
# private_train_loader, private_test_loader = get_private_data_loaders(
#     precision_fractional=args.precision_fractional,
#     workers=workers,
#     crypto_provider=crypto_provider
# )

print('Data Sharing complete')

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

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    data_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):  # <-- now it is a private dataset
        # if target.min() < 25 or target.max() > 140:
        #     continue

        start_time = time.time()

        optimizer.zero_grad()


        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        loss = ((output - target) ** 2).sum() / batch_size

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            # loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                       100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))


def test(args, model, private_test_loader, epoch):
    model.eval()
    test_loss = 0
    data_count = 0
    pred_list = []
    target_list = []
    with torch.no_grad():
        for data, target in test_loader:
            start_time = time.time()

            output = model(data)
            test_loss += ((output - target) ** 2).sum()
            data_count += len(output)
            pred_list.extend(output[:, 0].numpy())
            target_list.extend(target.numpy())
            # print('rmse:', torch.sqrt(((output - target) ** 2).sum() / args.batch_size))
            # print('r2score:', r2_score(target_list, pred_list))

    # test_loss = test_loss.get().float_precision()
    # print('Test set: Loss: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
    #            100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))
    print('\nTest set: Loss: avg MSE ({:.4f})\tTime: {:.3f}s'.format(test_loss / data_count, time.time() - start_time))

    rm = r_squared_mse(target_list, pred_list)

    if epoch % args.log_interval == 0:
        scatter_plot(target_list, pred_list, epoch, rm)
    # if epoch == args.epochs:


def scatter_plot(y_true, y_pred, epoch, message):
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
    plt.savefig("{}/scatter/{}.png".format(result_path, epoch))
    plt.clf()
    # plt.show()


def r_squared_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred,
                             sample_weight=sample_weight,
                             multioutput=multioutput)
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
    # exit()

    result_message = 'r2:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r2, mse, np.std(y_true), np.std(y_pred))
    return result_message

def save_model(model, path):

    torch.save(model.state_dict(), path)

if args.compressed:
    model = ML4CVD_shallow()
    summary(model, input_size=(12, 500), batch_size=args.batch_size)
else:
    model = ML4CVD()
    summary(model, input_size=(12, 5000), batch_size=args.batch_size)

print(model)

# model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
# for 12channel

# for 1 channel
# summary(model, input_size =(1, 12, 5000), batch_size=args.batch_size)
# exit()
# optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 4.58
# optimizer = optimizer.fix_precision()

for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)
    test(args, model, test_loader, epoch)
    # Save model
    if not os.path.exists('{}/{}'.format(result_path, 'models')):
        os.makedirs('{}/{}'.format(result_path, 'models'))
    if epoch % args.log_interval == 0:
        save_model(model, "{}/models/ep{}.h5".format(result_path, epoch))
