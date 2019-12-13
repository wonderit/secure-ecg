import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import h5py
import numpy as np
from torchvision import datasets, transforms
import syft as sy
hook = sy.TorchHook(torch)
client = sy.VirtualWorker(hook, id="client")
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
epochs = 10
n_test_batches = 200
MEAN = 59.3
STD = 10.6

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 50
        self.epochs = epochs
        self.lr = 0.001
        self.log_interval = 100
        self.loss_type = 'adam'
        self.n_train_items = 100
        self.n_test_items = 100

args = Arguments()


result_path = 'secure-result_torch/{}_{}_ep{}_bs{}_{}:{}_lr{}'.format(
    args.model_type,
    args.loss_type,
    args.epochs,
    args.batch_size,
    args.n_train_items,
    args.n_test_items,
    args.lr
)



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
    if args.model_type in ['shallow', 'ann', 'cnn2d']:
        x_list = x_list.reshape([12, 12 // 12, 500, 5000 // 500]).mean(3).mean(1)

        if args.model_type == 'cnn2d':
            x_list = x_list.reshape(x_list.shape[0], x_list.shape[1], 1)

    x_all.append(x_list)

x = np.asarray(x_all)
y = np.asarray(y_all)

print(x.shape, y.shape)

y = scale(y, MEAN, STD)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True)

print(train_loader.dataset.data.shape, train_loader.dataset.targets.shape)
print(test_loader)
# exit()
private_test_loader = []
for data, target in test_loader:
    private_test_loader.append((
        data.fix_precision().share(alice, bob, crypto_provider=crypto_provider),
        target.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    ))

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(391, 128)
        self.conv1d = nn.Conv1d(1, 1, 3)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28 * 28)
        x = F.relu(self.conv1d(x))
        x = self.avgpool1(x)  # 32
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))

model = Net2()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)

model.fix_precision().share(alice, bob, crypto_provider=crypto_provider)


def test(args, model, test_loader):
    model.eval()
    n_correct_priv = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader[:n_test_batches]:
            output = model(data)
            pred = output.argmax(dim=1)
            n_correct_priv += pred.eq(target.view_as(pred)).sum()
            n_total += args.test_batch_size
            # This 'test' function performs the encrypted evaluation. The model weights, the data inputs, the prediction and the target used for scoring are all encrypted!

            # However as you can observe, the syntax is very similar to normal PyTorch testing! Nice!

            # The only thing we decrypt from the server side is the final score at the end of our 200 items batches to verify predictions were on average good.
            n_correct = n_correct_priv.copy().get().float_precision().long().item()

            print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
                n_correct, n_total,
                100. * n_correct / n_total))

test(args, model, private_test_loader)
