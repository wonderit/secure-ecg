import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import glob
import h5py
import numpy as np
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import math

import time

class Arguments():
    def __init__(self):
        self.batch_size = 20
        self.epochs = 100
        self.lr = 1e-6   # 0.00002
        self.seed = 1
        self.log_interval = 1 # Log info at each batch
        self.precision_fractional = 3

        # We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
        self.n_train_items = 1600
        self.n_test_items = 400

args = Arguments()

_ = torch.manual_seed(args.seed)
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


DATAPATH = '/Users/wonsuk/projects/data/ecg/raw/2019-11-19'
# DATA_LENGTH = 100
# BATCH_SIZE = 10
TRAIN_RATIO = 0.8
ecg_key_string_list = [
    "strip_I",
    "strip_II",
    "strip_III",
    "strip_V1",
    "strip_V2",
    "strip_V3",
    "strip_V4",
    "strip_V5",
    "strip_V6",
    "strip_aVF",
    "strip_aVL",
    "strip_aVR"
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
            import matplotlib.pyplot as plt

            x = x.reshape([12, 5000])
            s = x.reshape([12, 12//12, 250, 5000 // 250]).mean(3).mean(1)
            # print('x', x.shape, x[0, :9])
            # plt.plot(x[4, :])
            # plt.show()
            x = s
            # print('s', s.shape, s[0, :3])
            # plt.plot(s[4, :])
            # plt.show()
            scaler = MinMaxScaler(feature_range=(-1, 1))
            x = scaler.fit_transform(x.numpy())
            x = torch.from_numpy(x)

        return x, y

    def __len__(self):
        return len(self.data)


print('Converting to TorchDataset...')

x_all = []
y_all = []
# for 12 data
for hdf_file in hdf5_files:
    f = h5py.File(hdf_file, 'r')
    y_all.append(f['continuous']['VentricularRate'][0] / 100)
    # x = np.zeros(shape=(5000))
    x_list = list()
    for (i, key) in enumerate(ecg_key_string_list):
        x = f['ecg_rest'][key][:]
        x_list.append(x)
    x_list = np.stack(x_list)
    x_list = x_list.reshape(12, -1)
    x_all.append(x_list)

# for 12 data
# for hdf_file in hdf5_files:
#     f = h5py.File(hdf_file, 'r')
#     y_all.append(f['continuous']['VentricularRate'][0] / 100)
#     x = np.zeros(shape=(12, 5000))
#     for (i, key) in enumerate(ecg_key_string_list):
#         x[i][:] = f['ecg_rest'][key][:]
#     x_all.append(x)

# for data only 1
# for hdf_file in hdf5_files:
#     f = h5py.File(hdf_file, 'r')
#     y_all.append(f['continuous']['VentricularRate'][0])
#     # x = np.zeros(shape=(5000))
#     x_list = list()
#     for (i, key) in enumerate(ecg_key_string_list):
#
#         x = f['ecg_rest'][key][:]
#         x_list.append(x)
#     x_list = np.stack(x_list)
#     # x_list = x_list.reshape(12, -1)
#     x_all.append(x_list)

x = np.asarray(x_all)
# x = x.reshape(-1, 1, 12, 5000)
y = np.asarray(y_all)

print(x.shape, y.shape)

data = ECGDataset(x, y, transform=True)
# data = ECGDataset(x, y, transform=False) # 4.58

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
        if i < n_test_items / args.batch_size
    ]

    return private_train_loader, private_test_loader

#
# private_train_loader, private_test_loader = get_private_data_loaders(
#     precision_fractional=args.precision_fractional,
#     workers=workers,
#     crypto_provider=crypto_provider
# )

print('Data Sharing complete')

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=(1, 10), stride=2, padding=1),
            # nn.Conv2d(4, 4, kernel_size=(1, 10), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=(1, 5), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=(1, 5), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(77 * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 4 * 77)
        x = self.linear_layers(x)
        return x

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.channel_size = 16
        self.conv1 = nn.Conv1d(12, self.channel_size, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(self.channel_size, 1, kernel_size=1, stride=1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(928, 200)
        self.fc2 = nn.Linear(246, 128)
        self.fc3 = nn.Linear(128, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        # x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernal_size = 16
        self.conv1 = nn.Conv1d(12, self.kernal_size, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=5, stride=2)
        # self.conv1 = nn.Conv1d(12, self.kernal_size, kernel_size=101, stride=2)
        # self.conv2 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=101, stride=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=51, stride=2)
        self.conv4 = nn.Conv1d(self.kernal_size, 1, kernel_size=1)
        self.fc1 = nn.Linear(562, 128)
        self.fc2 = nn.Linear(60, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        # x = x.view(-1, 600000)
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        # x = self.maxpool1(x.float())
        # x = F.relu(self.conv3(x))
        # x = self.maxpool1(x.float())
        # x = F.relu(self.conv3(x))
        # x = self.maxpool1(x.float())
        x = F.relu(self.conv4(x))
        # x = x.view(-1, )
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 562)
        # x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x).float())
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        return x

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    data_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):  # <-- now it is a private dataset
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
    r_squared_mse(target_list, pred_list)

    if epoch == args.epochs:
        scatter_plot(target_list, pred_list)

def scatter_plot(y_true, y_pred):
    import pandas as pd
    result = np.column_stack((y_true,y_pred))
    pd.DataFrame(result).to_csv("result/result_ep{}_bs{}_tr{}_test{}.csv".format(args.epochs,
                                                                                 args.batch_size,
                                                                                 args.n_train_items,
                                                                                 args.n_test_items), index=False)

    import matplotlib.pyplot as plt
    plt.scatter(y_true, y_pred)

    plt.savefig("result/result_ep{}_bs{}_tr{}_test{}.png".format(args.epochs,
                                                                                 args.batch_size,
                                                                                 args.n_train_items,
                                                                                 args.n_test_items))
    plt.show()


def r_squared_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred,
                  sample_weight=sample_weight, multioutput=multioutput)
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
    print('Scoring - MSE: ', mse, 'RMSE: ', math.sqrt(mse), 'R2 : ', r2)
    # print(y_pred)
    # exit()

def save_model(model, path):

    torch.save(model.state_dict(), path)

model = ANN()

print(model)
# model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
# for 12channel
summary(model, input_size =(12, 250), batch_size=args.batch_size)

# for 1 channel
# summary(model, input_size =(1, 12, 5000), batch_size=args.batch_size)
# exit()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 4.58
# optimizer = optimizer.fix_precision()

for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)
    test(args, model, test_loader, epoch)
    # Save model
    save_model(model, 'models/CNN_250x12_lr01e5_epoch10.h5')