import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import glob
import h5py
import numpy as np

import time

class Arguments():
    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 10
        self.epochs = 10
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1 # Log info at each batch
        self.precision_fractional = 3

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

# We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
n_train_items = 100
n_test_items = 100


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
    if count > n_train_items:
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

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
        if i < n_train_items / args.batch_size
    ]

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, download=True, transform=transformation),
    #     batch_size=args.test_batch_size
    # )

    private_test_loader = [
        (secret_share(data), secret_share(target.float()))
        for i, (data, target) in enumerate(test_loader)
        if i < n_test_items / args.test_batch_size
    ]

    return private_train_loader, private_test_loader


private_train_loader, private_test_loader = get_private_data_loaders(
    precision_fractional=args.precision_fractional,
    workers=workers,
    crypto_provider=crypto_provider
)

print('Data Sharing complete')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernal_size = 4
        self.conv1 = nn.Conv1d(12, self.kernal_size, kernel_size=101)
        self.conv2 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=101)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(self.kernal_size, self.kernal_size, kernel_size=51)
        self.conv4 = nn.Conv1d(self.kernal_size, 1, kernel_size=1)
        self.fc1 = nn.Linear(562, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        # x = x.view(-1, 600000)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x.float())
        x = F.relu(self.conv3(x.long()))
        x = self.maxpool1(x.float())
        x = F.relu(self.conv3(x.long()))
        x = self.maxpool1(x.float())
        x = self.conv4(x.long())
        x = x.view(-1, 562)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x).float())
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        return x

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):  # <-- now it is a private dataset
        start_time = time.time()

        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        loss = ((output - target) ** 2).sum().refresh() / batch_size

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                       100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))


def test(args, model, private_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()

            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader) * args.test_batch_size,
                        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))

model = Net()
model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)

optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision()

for epoch in range(1, args.epochs + 1):
    train(args, model, private_train_loader, optimizer, epoch)
    test(args, model, private_test_loader)