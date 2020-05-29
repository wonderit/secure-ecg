#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
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
parser.add_argument("-m", "--model_type", help="model name(shallow, normal, ann, cann, pc, cnn2d)", type=str, default='cnnavg')
parser.add_argument("-mpc", "--mpc", help="shallow model", action='store_true')
parser.add_argument("-lt", "--loss_type", help="use sgd as optimizer", type=str, default='sgd')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=4)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=10)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=4e-4)#4e-4
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=20)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=20)
parser.add_argument("-pf", "--precision_fractional", help="Set precision fractional", type=int, default=3)
parser.add_argument("-mom", "--momentum", help="Set momentum", type=float, default=0.9)

args = parser.parse_args()
MEAN = 59.3
STD = 10.6

# 5500 criteria
mean_x = 1.547
std_x = 156.820

_ = torch.manual_seed(args.seed)

result_path = 'secure-result_torch/{}_{}_ep{}_bs{}_{}:{}_lr{}_mom{}'.format(
    args.model_type,
    args.loss_type,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr,
    args.momentum
)

import syft as sy  # import the Pysyft library
hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning

# simulation functions
def connect_to_workers(n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]
def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")

workers = connect_to_workers(n_workers=2)
crypto_provider = connect_to_crypto_provider()


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

train_y = np.loadtxt('{}/{}'.format(DATAPATH, file_name_train_y), delimiter=',')
test_y = np.loadtxt('{}/{}'.format(DATAPATH, file_name_test_y), delimiter=',')

train_x = train_x.reshape(train_x.shape[0], 3, 500)
test_x = test_x.reshape(test_x.shape[0], 3, 500)

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

print('Converting to TorchDataset...')

train_data = ECGDataset(train_x, train_y, transform=False)
test_data = ECGDataset(test_x, test_y, transform=False)

print('Torch Dataset Train/Test split finished...')

def get_private_data_loaders(precision_fractional, workers, crypto_provider, train_dataset, test_dataset):

    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        """
        return (
            tensor
                .fix_precision(precision_fractional=precision_fractional)
                .share(*workers, crypto_provider=crypto_provider, requires_grad=True)
        )

    # train_dataset, test_dataset = torch.utils.data.random_split(data, [args.n_train_items, args.n_test_items])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    private_train_loader = [
        (secret_share(data), secret_share(target))
        for i, (data, target) in enumerate(train_loader)
        if i < args.n_train_items / args.batch_size
    ]

    private_test_loader = [
        (secret_share(data), secret_share(target))
        for i, (data, target) in enumerate(test_loader)
        if i < args.n_test_items / args.batch_size
    ]

    return private_train_loader, private_test_loader


private_train_loader, private_test_loader = get_private_data_loaders(
    precision_fractional=args.precision_fractional,
    workers=workers,
    crypto_provider=crypto_provider,
    train_dataset=train_data,
    test_dataset=train_data,
)

print('Data Sharing complete')



class CNNAVG(nn.Module):
    def __init__(self):
        super(CNNAVG, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        # self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=self.padding_size)
        # self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=self.padding_size)
        # self.fc1 = nn.Linear(2856, 16)     # 4 layer of CNN
        # self.fc1 = nn.Linear(2892, 16)     # 3 layer of CNN
        # self.fc1 = nn.Linear(1410, 16)     # 2 layer of CNN
        # self.fc1 = nn.Linear(2964, 16)     # 1 layer of CNN
        self.fc1 = nn.Linear(342, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.max_x = max_x

    def forward(self, x):
        x = self.conv1(x)  # 32
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

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):  # <-- now it is a private dataset
        start_time = time.time()

        optimizer.zero_grad()

        # print('data', data)
        # print('command', data.handle_func_command)
        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        # Reshape
        # output = output.view(batch_size, -1)
        # target = target.view(batch_size, -1)
        target = target.view(target.shape[0], 1)
        # loss = ((output - target) ** 2).sum()
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
            target = target.view(target.shape[0], 1)
            # pred = output.argmax(dim=1)
            # correct += pred.eq(target.view_as(pred)).sum()
            test_loss += ((output - target) ** 2).sum()
            data_count += len(data)

            if args.epochs == epoch:
                pred_list.extend(output.copy().get().float_precision().numpy()[:, 0])
                target_list.extend(target.copy().get().float_precision().numpy()[:, 0])

    test_loss = test_loss.get().float_precision()
    # print('Test set: Loss: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
    #            100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))
    # print('\nTest set: Loss: avg MSE ({:.4f})\tTime: {:.3f}s'.format(test_loss / data_count, time.time() - start_time))

    if args.epochs == epoch:
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

    r2 = r2_score(y_true, y_pred, sample_weight=None, multioutput=None)
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

if args.model_type in ['shallow', 'ann', 'cnn2d', 'cann', 'cnnavg']:

    if args.model_type == 'shallow':
        model = CNN_forMPC()
    elif args.model_type == 'cnn2d':
        model = CNN2D_forMPC()
    elif args.model_type == 'cann':
        model = CANN()
    elif args.model_type == 'cnnavg':
        model = CNNAVG()
    else:
        model = ANN()

    if args.model_type == 'cnn2d':
        summary(model, input_size=(12, 500, 1), batch_size=args.batch_size)
    else:
        summary(model, input_size=(3, 500), batch_size=args.batch_size)
else:
    model = ML4CVD()
    summary(model, input_size=(12, 5000), batch_size=args.batch_size)



# TODO

print('Save initial weight, bias start')
exit()
# print(model.conv1.weight)
# print(model.conv1.bias)
def transform_array(torch_array) :
    p = torch_array.detach().numpy()
    p = p.reshape(6, -1)
    p = np.transpose(p)
    p = np.round_(p, 7)
    return p

# conv1_weight = transform_array(model.conv1.weight)
# conv1_bias = transform_array(model.conv1.bias)
# np.savetxt('cann_conv1_weight.txt', conv1_weight, fmt='%1.7f')
# np.savetxt('cann_conv1_bias.txt', conv1_bias, fmt='%1.7f')

# conv2_weight = transform_array(model.conv2.weight)
# conv2_bias = transform_array(model.conv2.bias)
# np.savetxt('cann_conv2_weight.txt', conv2_weight, fmt='%1.7f')
# np.savetxt('cann_conv2_bias.txt', conv2_bias, fmt='%1.7f')
#

# conv3_weight = transform_array(model.conv3.weight)
# conv3_bias = transform_array(model.conv3.bias)
# np.savetxt('cann_conv3_weight.txt', conv3_weight, fmt='%1.7f')
# np.savetxt('cann_conv3_bias.txt', conv3_bias, fmt='%1.7f')

print('model sharing start')
model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
print('model sharing end')


if args.loss_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 4.58
elif args.loss_type == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr)  # 4.58
elif args.loss_type == 'lbfgs':
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr)  # 4.58
elif args.loss_type == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 4.58
else:
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

# optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision()



for epoch in range(1, args.epochs + 1):
    train(args, model, private_train_loader, optimizer, epoch)
    test(args, model, private_test_loader, epoch)
    # save_model(model, 'secure-models/model.h5')