#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import syft as sy  # import the Pysyft library
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

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compressed", help="Compress ecg data", action='store_true')
parser.add_argument("-m", "--model_type", help="model name(shallow, normal, ann, mpc)", type=str, default='shallow')
parser.add_argument("-mpc", "--mpc", help="shallow model", action='store_true')
parser.add_argument("-sgd", "--sgd", help="use sgd as optimizer", action='store_true')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=1)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=2e-4)
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=32)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=16)
parser.add_argument("-pf", "--precision_fractional", help="Set precision fractional", type=int, default=3)

args = parser.parse_args()
MEAN = 59.3
STD = 10.6
_ = torch.manual_seed(args.seed)

# model_type = 'original'
# if args.compressed:
#     model_type = 'comp'

loss_type = 'adam'
if args.sgd:
    loss_type = 'sgd'

result_path = 'secure-result_torch/{}_{}_ep{}_bs{}_{}:{}_lr{}'.format(
    args.model_type,
    loss_type,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr
)
hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning

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


alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)

DATAPATH = '../data/ecg/raw/2019-11-19'
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

def scale(arr, std, mean):
    arr -= mean
    arr /= (std + 1e-7)
    return arr

def rescale(arr, std, mean):
    arr = arr * std
    arr = arr + mean

    return arr

class ECGDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        # if self.transform:
        #     x = x.reshape([12, 5000])
        #     x = x.reshape([12, 12//12, 500, 5000 // 500]).mean(3).mean(1)
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
    if args.model_type in ['shallow', 'ann']:
        x_list = x_list.reshape([12, 12 // 12, 500, 5000 // 500]).mean(3).mean(1)
    x_all.append(x_list)

x = np.asarray(x_all)
y = np.asarray(y_all)

print(x.shape, y.shape)

y = scale(y, MEAN, STD)



class ANN2(nn.Module):
    def __init__(self):
        super(ANN2, self).__init__()
        # self.fc1 = nn.Linear(12 * 500, 128)
        self.rnn = nn.RNN(500, 20, 3)
        # self.fc1 = nn.Linear(60, 128)
        self.fc1 = nn.Linear(240, 16)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        # x = x.view(-1, 3 * 500)
        output, hidden = self.rnn(x, torch.zeros(3, 12, 20))
        x = output.view(output.shape[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_forMPC(nn.Module):
    def __init__(self):
        super(CNN_forMPC, self).__init__()
        self.kernel_size = 7
        self.padding_size = 3
        self.channel_size = 6
        # self.channel_size = 32
        self.conv1 = nn.Conv1d(12, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size * 3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
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

class ML4CVD_shallow(nn.Module):
    def __init__(self):
        super(ML4CVD_shallow, self).__init__()
        self.kernel_size = 7
        self.padding_size = 3
        self.channel_size = 12
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
        self.conv10 = nn.Conv1d(36, 1, kernel_size=1)
        self.fc1 = nn.Linear(250, 16)
        # self.fc1 = nn.Linear(2976, 16)
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
        # y = F.relu(self.conv4(y))  # 24
        # y = self.avgpool1(y)

        # x3 = F.relu(self.conv5(y))
        # c2 = torch.cat((y, x3), dim=1)
        # x4 = F.relu(self.conv6(c2))
        # y = torch.cat((y, x3, x4), dim=1)
        #
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

class ML4CVD(nn.Module):
    def __init__(self):
        super(ML4CVD, self).__init__()
        self.kernel_size = 71
        self.padding_size = 35
        self.channel_size = 32
        self.conv1 = nn.Conv1d(12, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
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


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(12 * 500, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 12 * 500)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):  # <-- now it is a private dataset
        start_time = time.time()

        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        # Reshape
        output = output.view(-1)
        target = target.view(-1)

        loss = ((output - target) ** 2).sum().refresh() / batch_size
        # loss = ((output - target) ** 2).sum() / batch_size

        loss.backward()

        optimizer.step()
        # step(optimizer)

        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
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
        for data, target in private_test_loader:
            start_time = time.time()

            output = model(data)

            # pred = output.argmax(dim=1)
            # correct += pred.eq(target.view_as(pred)).sum()
            # test_loss += ((output - target) ** 2).sum()
            data_count += len(data)
            pred_list.append(output.detach().clone().get().float_precision().numpy()[:, 0])
            target_list.append(target.detach().clone().get().float_precision().numpy())

    # test_loss = test_loss.get().float_precision()
    # print('Test set: Loss: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
    #            100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))
    # print('\nTest set: Loss: avg MSE ({:.4f})\tTime: {:.3f}s'.format(test_loss / data_count, time.time() - start_time))


    target_list = np.array(target_list).reshape(-1)
    pred_list = np.array(pred_list).reshape(-1)
    print(target_list.shape, pred_list.shape)
    print('example before rescale', target_list[0], pred_list[0])

    # output rescale
    target_list = rescale(target_list, MEAN, STD)
    pred_list = rescale(pred_list, MEAN, STD)

    print('example after rescale', target_list[0], pred_list[0])

    rm = r_squared_mse(target_list, pred_list)

    if epoch % args.log_interval == 0:
        scatter_plot(target_list, pred_list, epoch, rm)

        # Save model
        if not os.path.exists('{}/{}'.format(result_path, 'models')):
            os.makedirs('{}/{}'.format(result_path, 'models'))

        if epoch % args.log_interval == 0:
            save_model(model.get().float_precision(), "{}/models/ep{}.h5".format(result_path, epoch))


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

def step(opt, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    # if closure is not None:
    #     loss = closure()

    for group in opt.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            # if grad.is_sparse:
            #     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = opt.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data).float()
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            print(-step_size)
            print(exp_avg)
            print(denom)
            print(p.data)
            # torch.addcdiv()
            p.data.add_(-step_size.mul(exp_avg.div(denom)))
            # p.data.addcdiv_(-step_size, exp_avg, denom)

    return loss


if args.model_type in ['shallow', 'ann']:

    if args.model_type == 'shallow':
        model = CNN_forMPC()
    else:
        model = ANN2()
    summary(model, input_size=(12, 500), batch_size=args.batch_size)
else:
    model = ML4CVD()
    summary(model, input_size=(12, 5000), batch_size=args.batch_size)
#
# print('model sharing start')
# model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
# print('model sharing end')
#
#
# if args.sgd:
#     optimizer = optim.SGD(model.parameters(), lr=args.lr)
# else:
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 4.58
#
# # optimizer = optim.SGD(model.parameters(), lr=args.lr)
# # optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optimizer.fix_precision()
#
# for epoch in range(1, args.epochs + 1):
#     train(args, model, private_train_loader, optimizer, epoch)
#     test(args, model, private_test_loader, epoch)
#     # save_model(model, 'secure-models/model.h5')


# We encode everything
# model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)

data = torch.from_numpy(x).fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)
target = torch.from_numpy(y).fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)
model = model.fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)

opt = optim.SGD(params=model.parameters(),lr=0.1).fix_precision()

for iter in range(args.epochs):
    # 1) erase previous gradients (if they exist)
    opt.zero_grad()

    # 2) make a prediction
    pred = model(data)

    # Reshape
    pred = pred.view(-1)
    target = target.view(-1)

    # 3) calculate how much we missed
    loss = ((pred - target)**2).sum()

    # 4) figure out which weights caused us to miss
    loss.backward()

    # 5) change those weights
    opt.step()

    # 6) print our progress
    print('loss' , loss.get().float_precision())
    print('pred : ', rescale(pred.detach().clone().get().float_precision(), STD, MEAN))
    print('target: ', rescale(target.detach().clone().get().float_precision(), STD, MEAN))