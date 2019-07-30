#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : processing.py
# time      : 4/6/2019 14:20
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import numpy as np
import pandas as pd
import itertools
from imblearn.over_sampling import RandomOverSampler

def load_data(dataset_name, radius, k_neighbor, class_num):
    '''
    :param dataset_name: str, name of the dataset.
    :param radius: float, radius of the environment.
    :param k_neighbor: int, k neighboring atoms.
    :param class_num: int, class number of atom category.
    :return: numpy array of x, y and ddg.
    '''
    if k_neighbor != 0:
        data = np.load('../datasets_array/%s/k_neighbor/%s_r_%.2f_neighbor_%d_class_%d.npz'
                       % (dataset_name, dataset_name, radius, k_neighbor, class_num))
    else:
        data = np.load('../datasets_array/%s/radius/%s_r_%.2f_neighbor_%d_class_%d.npz'
                       % (dataset_name, dataset_name, radius, k_neighbor, class_num))
    x = data['x']
    y = data['y']
    ddg = data['ddg']
    assert x.shape[0] == ddg.shape[0]
    assert x.shape[0] == y.shape[0]
    return x,y,ddg

def sort_row(x, method):
    '''
    :param x: 3D tensor of this dataset, the axis are: data_num, row_num and col_nm.
    :param method: str, sorting method = [0,1,2]<-->sort by[dist, octant, random].
    :return: 3D tensor after sort.
    '''
    data_num, row_num, col_num = x.shape
    if method == 'distance':
        return x
    elif method == 'octant':
        x_new = np.zeros(x.shape)
        for i in range(x.shape[0]):
            data = pd.DataFrame(x[i])
            octant1 = data[(data[1] >= 0) & (data[2] >= 0) & (data[3] >= 0)]
            octant2 = data[(data[1] < 0) & (data[2] > 0) & (data[3] > 0)]
            octant3 = data[(data[1] < 0) & (data[2] < 0) & (data[3] > 0)]
            octant4 = data[(data[1] > 0) & (data[2] < 0) & (data[3] > 0)]
            octant5 = data[(data[1] > 0) & (data[2] > 0) & (data[3] < 0)]
            octant6 = data[(data[1] < 0) & (data[2] > 0) & (data[3] < 0)]
            octant7 = data[(data[1] < 0) & (data[2] < 0) & (data[3] < 0)]
            octant8 = data[(data[1] > 0) & (data[2] < 0) & (data[3] < 0)]
            temp_array = np.vstack((octant1, octant2, octant3, octant4, octant5, octant6, octant7, octant8))
            x_new[i] = temp_array
        return x_new
    elif method == 'permutation1':
        indices = np.load('../global/permutation1/indices_%d.npy' % row_num)
    elif method == 'permutation2':
        indices = np.load('../global/permutation2/indices_%d.npy' % row_num)
    for i in range(data_num):
        x[i] = x[i][indices]
    return x

def shuffle_data(x, y, ddg, random_seed):
    indices = [i for i in range(x.shape[0])]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    ddg = ddg[indices]
    return x,y,ddg

def split_val(x_train, y_train, ddg_train, ddg_test, random_seed):

    p_train_indices, n_train_indices = ddg_train >= 0, ddg_train < 0
    x_p_train, x_n_train = x_train[p_train_indices], x_train[n_train_indices]
    y_p_train, y_n_train = y_train[p_train_indices], y_train[n_train_indices]
    ddg_p_train, ddg_n_train = ddg_train[p_train_indices], ddg_train[n_train_indices]

    num_p_test, num_n_test = sum(ddg_test >= 0), sum(ddg_test < 0)

    x_p_val, x_n_val = x_p_train[:num_p_test], x_n_train[:num_n_test]
    y_p_val, y_n_val = y_p_train[:num_p_test], y_n_train[:num_n_test]
    ddg_p_val, ddg_n_val = ddg_p_train[:num_p_test], ddg_n_train[:num_n_test]

    x_p_train, x_n_train = x_p_train[num_p_test:], x_n_train[num_n_test:]
    y_p_train, y_n_train = y_p_train[num_p_test:], y_n_train[num_n_test:]
    ddg_p_train, ddg_n_train = ddg_p_train[num_p_test:], ddg_n_train[num_n_test:]

    x_val, y_val, ddg_val = np.vstack((x_p_val, x_n_val)), np.vstack((y_p_val, y_n_val)), np.hstack((ddg_p_val, ddg_n_val))
    x_train, y_train, ddg_train = np.vstack((x_p_train,x_n_train)), np.vstack((y_p_train,y_n_train)),\
                                  np.hstack((ddg_p_train,ddg_n_train))
    ## shuffe data.
    x_train_new, y_train_new, ddg_train_new = shuffle_data(x_train, y_train, ddg_train, random_seed=random_seed)
    assert x_train_new.shape[0] + x_val.shape[0] == x_train.shape[0]
    assert x_val.shape[0] == ddg_test.shape[0]

    return x_train_new, y_train_new, ddg_train_new, x_val, y_val, ddg_val

def oversampling(x_train, y_train):
    train_num, train_row, train_col = x_train.shape
    x_train = x_train.reshape((train_num, train_row * train_col))
    y_train = y_train.reshape(train_num)

    ros = RandomOverSampler()
    x_train_new, y_train_new = ros.fit_sample(x_train, y_train)
    x_train = x_train_new.reshape(-1,train_row,train_col)
    y_train = y_train_new.reshape(-1,1)
    positive_indices, negative_indices = y_train.reshape(-1, ) == 1, y_train.reshape(-1, ) == 0
    assert x_train[positive_indices].shape[0] == x_train[negative_indices].shape[0]
    assert x_train[positive_indices].shape[0] == x_train[negative_indices].shape[0]
    return x_train, y_train

def normalize(x_train, x_test, x_val, method = 'norm'):
    num_train, row_train, col_train = x_train.shape
    num_val, row_val, col_val = x_val.shape
    num_test, row_test, col_test = x_test.shape
    x_train = x_train.reshape((num_train * row_train, col_train))
    x_val = x_val.reshape((num_val*row_val, col_val))
    x_test = x_test.reshape((num_test * row_test, col_test))

    if method == 'norm':
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train -= mean
        x_train /= std
        x_val -= mean
        x_val /= std
        x_test -= mean
        x_test /= std
    elif method == 'max':
        max_ = x_train.max(axis=0)
        x_train /= max_
        x_val /= max_
        x_test /= max_
    x_train = x_train.reshape(num_train,row_train,col_train)
    x_val = x_val.reshape(num_val,row_val,col_val)
    x_test = x_test.reshape(num_test,row_test,col_test)
    return x_train, x_val, x_test

def reshape_tensor(x_train, x_test, x_val):
    ## reshape array to Input shape
    train_data_num, row_num, col_num = x_train.shape
    test_data_num, val_data_num = x_test.shape[0], x_val.shape[0]
    x_train = x_train.reshape(train_data_num, row_num, col_num, 1)
    x_test = x_test.reshape(test_data_num, row_num, col_num, 1)
    x_val = x_val.reshape(val_data_num, row_num, col_num, 1)

    return x_train, x_test, x_val

def split_delta_r(x_train):
    x_train, delta_r_train = x_train[:, :, :-5], x_train[:, 0, -5:]
    x_train = x_train[:, :, :, np.newaxis]
    return x_train


if __name__ == '__main__':
    data = np.load('../datasets_array/S1925/k_neighbor/S1925_r_50.00_neighbor_50_class_5.npz')
    x = data['x']
    print('x_shape:',x.shape)
    # print(x[0,0:5,:])
    x_new, pchange = multi_channel(x)