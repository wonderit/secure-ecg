# Data split
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
from torchsummary import summary
torch.manual_seed(7)

DATAPATH = '/Users/wonsuk/projects/data/ecg/raw/2019-11-19'
DATA_LENGTH = 100
BATCH_SIZE = 10
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
    if count > DATA_LENGTH:
        break
    hdf5_files.append(f)

print('Data Loading finished (row:{})'.format(len(hdf5_files)))


class ECGDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).int()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


print('Converting to TorchDataset...')

x_all = []
y_all = []
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

data = ECGDataset(np.asarray(x_all), np.asarray(y_all))

train_size = int(TRAIN_RATIO * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for batch_idx, (data, target) in enumerate(dataloader_train):
    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))
    print(target)
hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
secure_worker = sy.VirtualWorker(id="secure_worker", hook=hook)

# Set model
# A Toy Model
class CNN(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernal_size = 4
        self.conv1 = nn.Conv1d(12, self.kernal_size, kernel_size=101)
        self.conv2 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=101)
        self.maxpool1 = nn.AvgPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=51)
        self.conv4 = nn.Conv1d(self.kernal_size, 1, kernel_size=1)
        # self.fc1 = nn.Linear(12 * 5000, 1)
        # self.fc2 = nn.Linear(2000, 1)
        # # self.fc2 = nn.Linear(1000, 1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=4, kernel_size=101),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=101),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=51),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=51),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(562, 1)

    def forward(self, x):
        # x = x.view(BATCH_SIZE, -1)
        print('check shape')
        print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        # x = F.sigmoid(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # return x
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = x.reshape(x.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = F.sigmoid(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernal_size = 4
        self.conv1 = nn.Conv1d(12, self.kernal_size, kernel_size=101)
        self.conv2 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=101)
        self.maxpool1 = nn.AvgPool1d(kernel_size=2)
        self.fc1 = nn.Linear(4800, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x = x.view(-1, 28 * 28)

        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
model = Net()

summary(model, input_size =(12, 5000), batch_size=BATCH_SIZE)
# exit()

model = model.fix_precision().share(bob, alice, crypto_provider=secure_worker, requires_grad=True)

opt = optim.SGD(params=model.parameters(), lr=0.1).fix_precision()

for iter in range(1):
    # 1) erase previous gradients (if they exist)
    opt.zero_grad()

    for batch_idx, (data, target) in enumerate(dataloader_train):
        data = data.fix_precision().share(bob, alice, crypto_provider=secure_worker, requires_grad=True)
        target = target.fix_precision().share(bob, alice, crypto_provider=secure_worker, requires_grad=True)

        # 2) make a prediction
        print(data.shape)
        pred = model(data)

        # 3) calculate how much we missed
        loss = ((pred - target) ** 2).sum()

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()

        # 6) print our progress
        print(loss.get().float_precision())