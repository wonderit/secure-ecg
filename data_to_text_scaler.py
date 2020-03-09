#!/usr/bin/env python3

import torch
import glob
import h5py
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=5000)
parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=500)
parser.add_argument("-sc", "--scaler", help="Set random seed", type=str, default='raw')
parser.add_argument("-x", "--is_remove_outlier_x", help="Set remove outlier x", action='store_true')
parser.add_argument("-y", "--is_remove_outlier_y", help="Set remove outlier y", action='store_true')

args = parser.parse_args()
_ = torch.manual_seed(args.seed)
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
    if count > (args.n_train_items + args.n_test_items * 4):
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


def scale_div(arr, denom):
    arr = arr / denom
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

for i in range(x.shape[1]):
    part_x = x[:, i, :]
    print('train_x m, s: ', part_x.mean(), part_x.std())
    print('train_x min, max: ', part_x.min(), part_x.max())

    if args.scaler == 'minmax':
        part_x = scale_minmax(part_x, part_x.min(), part_x.max())
    elif args.scaler == 'maxabs':
        part_x = scale_maxabs(part_x, np.max(np.abs(part_x)))
    elif args.scaler == 'robust':
        part_x = scale_robust(part_x, np.quantile(part_x, 0.25), np.quantile(part_x, 0.75))
    elif args.scaler == 'standard':
        part_x = scale(part_x, part_x.mean(), part_x.std())
    elif args.scaler == 'div10':
        part_x = scale_div(part_x, 10)
    elif args.scaler == 'div100':
        part_x = scale_div(part_x, 100)


    x[:, i, :] = part_x


x = x.reshape(x.shape[0], -1)

if args.is_remove_outlier_x:
    zscore_of_x = np.abs(stats.zscore(x))
    x_condition = np.copy(x)
    x_condition[np.where(zscore_of_x <= 3)] = 0
    x_condition[np.where(zscore_of_x > 3)] = 1
    x_condition = np.sum(x_condition, axis=1)
    x = x[np.where(x_condition == 0)]
    y = y[np.where(x_condition == 0)]
    print('train_x m, s: ', x.mean(), x.std())
    print('train_x min, max: ', x.min(), x.max())
    print('x_length : ', x.shape[0])

if args.is_remove_outlier_y:
    zscore_of_y = np.abs(stats.zscore(y))
    x = x[np.where(zscore_of_y <= 3)]
    y = y[np.where(zscore_of_y <= 3)]
    print('x_length : ', x.shape[0])

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
data_dir = '../data/ecg/text_demo_{}'.format(sum(total_lengths))

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