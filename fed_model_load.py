#%%

import os
os.environ['CUDA_VISIBLE_DEVICES'] = " "

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from scipy import stats
import numpy as np
from torchsummary import summary
from sklearn.metrics import mean_squared_error
import math
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--is_comet", help="Set is Comet", action='store_true')
parser.add_argument("-m", "--model_type", help="model name(shallow, normal, ann, mpc, cnn2d)", type=str, default='cnnavg')
parser.add_argument("-mpc", "--mpc", help="shallow model", action='store_true')
parser.add_argument("-lts", "--loss_type_saved", help="use sgd as optimizer", type=str, default='sgd')
parser.add_argument("-lt", "--loss_type", help="use sgd as optimizer", type=str, default='sgd')
parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=3)
parser.add_argument("-es", "--epochs_saved", help="Set epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
parser.add_argument("-lr", "--lr", help="Set learning rate", type=float, default=1e-2)
parser.add_argument("-eps", "--eps", help="Set epsilon of adam", type=float, default=1e-7)
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-sc", "--scaler", help="Set random seed", type=str, default='max30_federated')
parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=5)
parser.add_argument("-mom", "--momentum", help="Set momentum", type=float, default=0.9)
parser.add_argument("-fr", "--federated_ratio", help="federated_ratio", type=float, default=0.1)

args = parser.parse_args(args=[])

#%%

max_x = 0

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

MEAN = 61.9
STD = 10.9

_ = torch.manual_seed(args.seed)

result_path = os.path.join('result_torch', 'from_fed{}_{}_fr{}_{}_{}_eps{}_ep{}_bs{}_lr{}_mom{}'.format(
    args.loss_type_saved,
    args.scaler,
    args.federated_ratio,
    args.model_type,
    args.loss_type,
    args.eps,
    args.epochs,
    args.batch_size,
    args.lr,
    args.momentum
))


result_path_saved = os.path.join('result_torch', 'fed_{}_fr{}_{}_{}_eps{}_ep{}_bs{}_lr{}_mom{}'.format(
    args.scaler,
    args.federated_ratio,
    args.model_type,
    args.loss_type_saved,
    args.eps,
    args.epochs_saved,
    args.batch_size,
    args.lr,
    args.momentum
))

#%%



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

result_array = []

#%%

# federated_instance_count =  int(train_x.shape[0] * args.federated_ratio)
federated_instance_count = 0
train_x = train_x[federated_instance_count:, :, :]
train_y = train_y[federated_instance_count:]

batches = int((5000-federated_instance_count) / args.batch_size)
log_batches = int(batches / args.log_interval)

#%%

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

#%%

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


class CNNMAX(nn.Module):
    def __init__(self):
        super(CNNMAX, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(372, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = self.maxpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        y = F.relu(self.conv3(x))
        y = self.maxpool3(y)
        y = y.view(y.shape[0], -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

def report_scores(X, y, trained_model):
    y_true = []
    y_pred = []


    with torch.no_grad():
        scores = trained_model(torch.from_numpy(X).float())
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

        print('batch_idx', batch_idx, batches)

        if batch_idx % args.log_interval == 0:
            # loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                       100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))

            # if args.is_comet:
            #     training_step = training_step + 1
            #     y_true_train, y_pred_train, train_mse_loss = report_scores(train_x, train_y, model)
            #     _, train_r = r_mse(y_true_train, y_pred_train, train_mse_loss)
            #     print('step : ', training_step)
            #     experiment.log_metric("train_mse", train_mse_loss, epoch=epoch, step=training_step)
            #     experiment.log_metric("train_r", train_r, epoch=epoch, step=training_step)

            # test during training
            test(args, model, test_loader, epoch, batch_idx, training_step)

        if batch_idx % batches == 0:
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

    rm, test_r = r_mse(target_list, pred_list)

    if batch == batches:
        scatter_plot(target_list, pred_list, epoch, rm, batch)
    #
    #
    # if args.is_comet:
    #     experiment.log_metric("test_mse", test_loss / data_count, epoch=epoch, step=step)
    #     experiment.log_metric("test_r", test_r, epoch=epoch, step=step)

    if batch % args.log_interval == 0:
        result = dict()
        result['mse_test'] = test_loss.numpy() / data_count
        result['r_test'] = test_r
        result_array.append(result)

#%%

def scatter_plot(y_true, y_pred, epoch, message, batch):
    result = np.column_stack((y_true,y_pred))

    if not os.path.exists('{}/{}'.format(result_path, 'csv')):
        os.makedirs('{}/{}'.format(result_path, 'csv'))

    if not os.path.exists('{}/{}'.format(result_path, 'scatter')):
        os.makedirs('{}/{}'.format(result_path, 'scatter'))

    pd.DataFrame(result).to_csv("{}/csv/{}.csv".format(result_path, epoch), index=False)

    import matplotlib.lines as mlines
    fig, ax = plt.subplots()
    line = mlines.Line2D([0, 1], [0, 1], color='red')

    ax.scatter(y_pred, y_true, s=3)

    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    # set axes range
    plt.xlim(30, 110)
    plt.ylim(30, 110)

    # plt.savefig("{}/scatter/{}.png".format(result_path, epoch))

    if args.is_comet:
        experiment.log_figure(figure=plt, figure_name='{}_{}.png'.format(epoch, batch))
    else:
        plt.savefig("{}/scatter/{}_{}.png".format(result_path, epoch, batch), dpi=600)
    plt.clf()
    # plt.show()



def r_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    # r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred)
    # bounds_check = np.min(y_pred) > MIN_MOISTURE_BOUND
    # bounds_check = bounds_check&(np.max(y_pred) < MAX_MOISTURE_BOUND)

    y_true = np.array(y_true, dtype=np.float)
    y_true = y_true.flatten()
    y_pred = np.array(y_pred, dtype=np.float)
    y_pred = y_pred.flatten()

    r = stats.pearsonr(y_true, y_pred)[0]
    r2 = r**2

    print('Scoring - std', np.std(y_true), np.std(y_pred))
    print('Scoring - median', np.median(y_true), np.median(y_pred))
    print('Scoring - min', np.min(y_true), np.min(y_pred))
    print('Scoring - max', np.max(y_true), np.max(y_pred))
    print('Scoring - mean', np.mean(y_true), np.mean(y_pred))
    print('Scoring - MSE: ', mse, 'RMSE: ', math.sqrt(mse))
    print('Scoring - R2: ', r2)
    # print(y_pred)


    result_message = 'r:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r, mse, np.std(y_true), np.std(y_pred))
    return result_message, r

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

#%%

if args.model_type in ['shallow', 'ann', 'cnn2d', 'cann', 'cnnavg', 'cnnmax']:

    if args.model_type == 'cnnavg':
        model = CNNAVG()
    elif args.model_type == 'cnnmax':
        model = CNNMAX()

    # if args.model_type == 'cnn2d':
    #     summary(model, input_size=(3, 500, 1), batch_size=args.batch_size)
    # else:
    #     summary(model, input_size=(3, 500), batch_size=args.batch_size)
# else:
#     model = ML4CVD()
#     summary(model, input_size=(12, 5000), batch_size=args.batch_size)

print(model)
print(model.conv1.weight)

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


#%%

# Load Model from previous saved model
model.load_state_dict(torch.load("{}/models/ep{}.h5".format(result_path_saved, args.epochs_saved)))
model.eval()
print(model.conv1.weight)

#%%


for epoch in range(1, args.epochs + 1):

    # Save model
    if not os.path.exists('{}/{}'.format(result_path, 'models')):
        os.makedirs('{}/{}'.format(result_path, 'models'))

    if epoch == 1:
        save_model_to_txt(model, "{}/models/".format(result_path), epoch-1)
    train(args, model, train_loader, optimizer, epoch, test_loader)
    # test(args, model, test_loader, epoch, epoch * batches)
    if epoch % args.log_interval == 0:
        save_model(model, "{}/models/ep{}.h5".format(result_path, epoch))

#%%

import csv
csv_file = "from_fed_{}_{}_{}_to_{}.csv".format(args.federated_ratio, args.model_type, args.loss_type_saved, args.loss_type)
csv_columns = ['mse_test', 'r_test']
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in result_array:
            writer.writerow(data)
except IOError:
    print("I/O error")


