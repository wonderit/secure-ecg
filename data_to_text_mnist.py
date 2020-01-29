#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
import os
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=100)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=100)

args = parser.parse_args()
_ = torch.manual_seed(args.seed)
transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transformation),
        batch_size=args.n_train_items
    )

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transformation),
        batch_size=args.n_test_items
    )

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

# hdf5_files = []
# count = 0
# for f in glob.glob("{}/*.hd5".format(DATAPATH)):
#     count += 1
#     if count > (args.n_train_items + args.n_test_items):
#         break
#     hdf5_files.append(f)

# print('Data Loading finished (row:{})'.format(len(hdf5_files)))


print('Converting to TorchDataset...')

x_all = []
y_all = []
# for hdf_file in hdf5_files:
#     f = h5py.File(hdf_file, 'r')
#     y_all.append(f['continuous']['VentricularRate'][0])
#     x_list = list()
#     for (i, key) in enumerate(ecg_key_string_list):
#         x = f['ecg_rest'][key][:]
#         x_list.append(x)
#     x_list = np.stack(x_list)
#     x_list = x_list.reshape([3, 12 // 12, 500, 5000 // 500]).mean(3).mean(1)
#     x_all.append(x_list)

train_x = [
        (data.numpy())
        for i, (data, target) in enumerate(train_loader)
        if i < args.n_train_items / args.n_train_items
    ]


train_y = [
        (one_hot_of(target).numpy())
        for i, (data, target) in enumerate(train_loader)
        if i < args.n_train_items / args.n_train_items
    ]


test_x = [
        (data.numpy())
        for i, (data, target) in enumerate(test_loader)
        if i < args.n_test_items / args.n_test_items
    ]


test_y = [
        (one_hot_of(target).numpy())
        for i, (data, target) in enumerate(test_loader)
        if i < args.n_test_items / args.n_test_items
    ]


train_x = np.asarray(train_x).squeeze(0)
train_y = np.asarray(train_y).squeeze(0)
test_x = np.asarray(test_x).squeeze(0)
test_y = np.asarray(test_y).squeeze(0)


train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)
total_lengths = [args.n_train_items, args.n_test_items]
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
data_dir = '../data/mnist/text_demo_{}'.format(sum(total_lengths))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_file_suffix = 'train'
test_file_suffix = 'test'

file_name_train_x = 'X{}'.format(train_file_suffix)
file_name_train_y = 'y{}'.format(train_file_suffix)
file_name_test_x = 'X{}'.format(test_file_suffix)
file_name_test_y = 'y{}'.format(test_file_suffix)

np.savetxt('{}/{}'.format(data_dir, file_name_train_x), train_x, delimiter=',', fmt='%1.1f')
np.savetxt('{}/{}'.format(data_dir, file_name_train_y), train_y, delimiter=',', fmt='%1.0f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_x), test_x, delimiter=',', fmt='%1.1f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_y), test_y, delimiter=',', fmt='%1.0f')