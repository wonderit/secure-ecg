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
parser.add_argument("-m", "--model_type", help="model name(shallow, normal, ann, mpc, cnn2d)", type=str, default='cann')
parser.add_argument("-mpc", "--mpc", help="shallow model", action='store_true')
parser.add_argument("-lt", "--loss_type", help="use sgd as optimizer", type=str, default='sgd')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=1)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=1e-3)
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=80)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=20)
parser.add_argument("-mom", "--momentum", help="Set momentum", type=float, default=0.9)

args = parser.parse_args()

if args.is_comet:
    experiment = Experiment(api_key="eIskxE43gdgwOiTV27APVUQtB", project_name='secure-ecg', workspace="wonderit")
else:
    experiment = None

def scale(arr, m, s):
    arr = arr - m
    arr = arr / (s + 1e-7)
    return arr

# LOSS = 'hinge'

def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m
    return arr

MEAN = 59.3
STD = 10.6
_ = torch.manual_seed(args.seed)

result_path = os.path.join('result_torch', 'text{}_{}_ep{}_bs{}_{}-{}_lr{}_mom{}'.format(
    args.model_type,
    args.loss_type,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr,
    args.momentum
))

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


# train_y = scale(train_y, MEAN, STD)
# test_y = scale(test_y, MEAN, STD)
train_x = scale(train_x, 1.547, 156.820)
test_x = scale(test_x, 1.547, 156.820)


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

    if args.is_comet:
        experiment.log_metric("test_mse", test_loss / data_count, epoch=epoch)

    # # output rescale
    # target_list = rescale(target_list, MEAN, STD)
    # pred_list = rescale(pred_list, MEAN, STD)

    rm = r_squared_mse(target_list, pred_list)

    if epoch % args.log_interval == 0:
        scatter_plot(target_list, pred_list, epoch, rm)

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
    # plt.savefig("{}/scatter/{}.png".format(result_path, epoch))

    if args.is_comet:
        experiment.log_figure(figure=plt, figure_name='{}.png'.format(epoch))
    else:
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

    if args.is_comet:
        experiment.log_metric("test_r2", r2)

    result_message = 'r2:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r2, mse, np.std(y_true), np.std(y_pred))
    return result_message

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

def load_model(model, path):
    print('path', path)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


class CNNAVG(nn.Module):
    def __init__(self):
        super(CNNAVG, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2, count_include_pad=False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2, count_include_pad=False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2, count_include_pad=False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2, count_include_pad=False)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        # self.fc1 = nn.Linear(2856, 16)     # 4 layer of CNN
        # self.fc1 = nn.Linear(2892, 16)     # 3 layer of CNN
        # self.fc1 = nn.Linear(1410, 16)     # 2 layer of CNN
        # self.fc1 = nn.Linear(2964, 16)     # 1 layer of CNN
        self.fc1 = nn.Linear(342, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)  # 32
        x = self.avgpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        y = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.avgpool3(x)
        # x = self.conv3(x)
        # y = F.relu(self.conv5(x))
        y = self.avgpool4(y)
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


if args.model_type in ['shallow', 'ann', 'cnn2d', 'cann', 'cnnavg']:

    if args.model_type == 'cnnavg':
        model = CNNAVG()

print(model)
model = load_model(model, "{}/models/ep{}.h5".format(result_path, args.epochs))
test(args, model, test_loader, args.epochs)


