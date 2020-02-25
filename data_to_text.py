#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
import glob
import h5py
import numpy as np
import argparse
from scipy import stats
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=5000)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=500)

args = parser.parse_args()
_ = torch.manual_seed(args.seed)
# for 5500
MEAN = 59.3
STD = 10.6
# x mean, std:  1.4867225111111129 155.86417997666166
# y mean, std:  61.906166666666664 10.701683448826586

# x mean, std:  1.693 155.617
# y mean, std:  61.6 9.8

# for 22000
# MEAN = 61.93
# STD = 10.91

DATAPATH = '../data/ecg/raw/2019-11-19'
ecg_key_string_list = [
    "strip_I",
    "strip_II",
    "strip_III",
    # "strip_aVR",
    # "strip_aVL",
    # "strip_aVF",
    # "strip_V1",
    # "strip_V2",
    # "strip_V3",
    # "strip_V4",
    # "strip_V5",
    # "strip_V6",
]

hdf5_files = []
count = 0
for f in glob.glob("{}/*.hd5".format(DATAPATH)):
    count += 1
    if count > (args.n_train_items + args.n_test_items * 2):
        break
    hdf5_files.append(f)

print('Data Loading finished (row:{})'.format(len(hdf5_files)))

def scale(arr, m, s):
    arr -= m
    arr /= (s + 1e-7)
    return arr

def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m

    return arr

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
    x_list = x_list.reshape([3, 12 // 12, 500, 5000 // 500]).mean(3).mean(1)
    x_all.append(x_list)

x = np.asarray(x_all)
y = np.asarray(y_all)

x = x.reshape(x.shape[0], -1)

zscore_of_y = np.abs(stats.zscore(y))

x = x[np.where(zscore_of_y <= 3)]
y = y[np.where(zscore_of_y <= 3)]


total_lengths = [args.n_train_items, args.n_test_items]

x = x[:sum(total_lengths), :]
y = y[:sum(total_lengths)]


print('x mean, std: ', np.round(x.mean(), 3), np.round(x.std(), 3))
print('y mean, std: ', np.round(y.mean(), 1), np.round(y.std(), 1))
print(x.shape, y.shape)
y = scale(y, np.round(y.mean(), 1), np.round(y.std(), 1))

# indices = sample(range(sum(total_lengths)), args.n_test_items)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = args.n_test_items / sum(total_lengths))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
data_dir = '../data/ecg/text_demo_new_{}'.format(sum(total_lengths))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_file_suffix = 'train'
test_file_suffix = 'test'

file_name_train_x = 'X{}'.format(train_file_suffix)
file_name_train_y = 'y{}'.format(train_file_suffix)
file_name_test_x = 'X{}'.format(test_file_suffix)
file_name_test_y = 'y{}'.format(test_file_suffix)

np.savetxt('{}/{}'.format(data_dir, file_name_train_x), train_x, delimiter=',', fmt='%1.1f')
np.savetxt('{}/{}'.format(data_dir, file_name_train_y), train_y, delimiter='\n', fmt='%f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_x), test_x, delimiter=',', fmt='%1.1f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_y), test_y, delimiter='\n', fmt='%f')