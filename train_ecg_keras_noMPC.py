#!/usr/bin/env python3

from keras.layers.convolutional import Conv1D, AveragePooling1D
from keras.layers import Dense, Flatten, concatenate
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras import losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import glob
import h5py
import os

def scale(arr, std, mean):
    arr -= mean
    arr /= (std + 1e-7)
    return arr

def rescale(arr, std, mean):
    arr = arr * std
    arr = arr + mean

    return arr

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def create_model(model_type, input_shape, loss_function):
    print('input_shape', input_shape)
    if model_type.startswith('cnn'):
        data_input = Input(shape=(input_shape[1], input_shape[2]))
        kernel_size = 7
        if input_shape[1] > 500:
            kernel_size = 71
        # Conv Block 1
        x = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(data_input)
        x = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = AveragePooling1D(pool_size=2, strides=2)(x)

        # residual block 1
        x1 = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
        c1 = concatenate([x, x1])
        x2 = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(c1)
        x = concatenate([x, x1, x2])

        # Conv Block 2
        x = Conv1D(filters=24, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = AveragePooling1D(pool_size=2, strides=2)(x)

        # residual block 2
        x1 = Conv1D(filters=24, kernel_size=kernel_size, padding='same', activation='relu')(x)
        c1 = concatenate([x, x1])
        x2 = Conv1D(filters=24, kernel_size=kernel_size, padding='same', activation='relu')(c1)
        x = concatenate([x, x1, x2])

        # Conv Block 3
        x = Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = AveragePooling1D(pool_size=2, strides=2)(x)

        # residual block 3
        x1 = Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu')(x)
        c1 = concatenate([x, x1])
        x2 = Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu')(c1)
        x = concatenate([x, x1, x2])

        # Flatten
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        y = Dense(1)(x)
        model = Model(inputs=data_input, outputs=y)
        model.compile(loss=losses.logcosh, optimizer=Adam(lr=args.learning_rate), metrics=[r2_keras])
        # model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=args.learning_rate), metrics=[r2_keras])
    else:
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(24, activation='linear'))
        model.compile(loss=loss_function, optimizer='adam', metrics=[r2_score])

    return model


def save(model):
    # serialize model to JSON
    model_json = model.to_json()
    model_export_path_folder = 'models_keras/{}_{}_{}_{}'.format(args.model, args.n_train_items + args.n_test_items,
                                                                 args.batch_size, args.epochs)
    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    # Visualize Model
    plot_model(model, to_file='{}/model.png'.format(model_export_path_folder), show_shapes=True)

    model_export_path_template = '{}/{}_1.{}'
    model_export_path = model_export_path_template.format(model_export_path_folder, args.loss_function, 'json')
    with open(model_export_path, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(
        model_export_path_template.format(model_export_path_folder, args.loss_function, 'h5'))
    print("Saved model to disk")


def test(y_true, y_pred, model):
    r2 = r2_score(y_true, y_pred, multioutput=None)
    mse = mean_squared_error(y_true, y_pred,
                             sample_weight=None,
                             multioutput=None)

    print('Scoring - std', np.std(y_true), np.std(y_pred))
    print('Scoring - median', np.median(y_true), np.median(y_pred))
    print('Scoring - min', np.min(y_true), np.min(y_pred))
    print('Scoring - max', np.max(y_true), np.max(y_pred))
    print('Scoring - mean', np.mean(y_true), np.mean(y_pred))
    print('Scoring - MSE: ', mse, 'RMSE: ', np.sqrt(mse))
    print('Scoring - R2: ', r2)

    model_export_path_folder = 'models_keras/{}_{}_{}_{}'.format(args.model, args.n_train_items + args.n_test_items,
                                                                 args.batch_size, args.epochs)
    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)
    result = np.column_stack((y_true, y_pred))

    model_export_path_template = '{}/result.{}'
    model_export_path = model_export_path_template.format(model_export_path_folder, 'csv')
    pd.DataFrame(result).to_csv(model_export_path, index=False)

    plt.scatter(y_pred, y_true, s=3)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.savefig(model_export_path_template.format(model_export_path_folder, 'png'))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-w", "--width", help="Compress ecg data by width", type=int, default=500)
    parser.add_argument("-e", "--epochs", help="Set epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", help="Set learning rate", type=float, default=2e-4)
    parser.add_argument("-lf", "--loss_function", help="Select model type.", default="mse")
    parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
    parser.add_argument("-li", "--log_interval", help="Set log interval", type=int, default=1)
    parser.add_argument("-tr", "--n_train_items", help="Set log interval", type=int, default=80)
    parser.add_argument("-te", "--n_test_items", help="Set log interval", type=int, default=20)

    args = parser.parse_args()

    test_rate = args.n_test_items / (args.n_train_items + args.n_test_items)

    if args.width > 500:
        print('Custom width :', args.width)

    DATAPATH = '../data/ecg/raw/2019-11-19'
    # DATA_LENGTH = 100
    # BATCH_SIZE = 10
    # TRAIN_RATIO = 0.8

    MEAN = 59.3
    STD = 10.6
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

        if args.width < 5000:
            x_list = x_list.reshape([args.width, 5000 // args.width, 12, 12 // 12]).mean(3).mean(1)
        else:
            x_list = x_list.reshape([5000, 12])
        x_all.append(x_list)

    x = np.asarray(x_all)
    y = np.asarray(y_all)

    # print(x.shape, y.shape)

    y = scale(y, MEAN, STD)

    print('Data Loading... Finished.')
    print('Data Splitting...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=args.seed)

    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)
    print('Data Splitting... finsihed')

    model = create_model(args.model, x_train.shape, 'mse')

    history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    # save and visualize model
    save(model)

    y_pred = model.predict(x_test)

    y_test = rescale(y_test, MEAN, STD)
    y_pred = rescale(y_pred, MEAN, STD)
    test(y_test, y_pred, model)

    # Loss
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    train_progress_figure_path_folder = 'result/train_progress_paper'
    if not os.path.exists(train_progress_figure_path_folder):
        os.makedirs(train_progress_figure_path_folder)
    plt.savefig('{}/{}_{}_{}_{}.png'.format(train_progress_figure_path_folder,
                                               args.model,
                                               args.n_train_items + args.n_test_items,
                                               args.batch_size, args.epochs))
