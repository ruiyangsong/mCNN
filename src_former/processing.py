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

def multi_channel(x):
    """
    :param x: 二通道的data_array.
    :return: 多通道的data_array
    """
    ## 为方便正规化、切分验证集等操作，多通道表示在最后完成.
    data_num, row, col = x.shape

    if col == 16:
        # S2648,不含RSA,['dist'||'x','y', 'z'||'ph'||'temperature'||'C', 'H', 'O', 'N', 'other']
        # group_num = 5
        #print('列属性数目为：',col)
        pchange = x[:, 0, 11:]
        x_new = np.zeros(data_num * 120 * row * 11).reshape(data_num, 120, row, 11)
        nums = itertools.permutations([0, 1, 2, 3, 4])
        dict = {'0': [0], '1': [1, 2, 3], '2': [4], '3': [5], '4': [6,7, 8, 9, 10]}
        count = 0
        for sequence in nums:
            keys = [str(i) for i in sequence]
            for k in range(data_num):
                temp = x[k]
                x_new[k, count, :, :] = temp[:,dict[keys[0]] + dict[keys[1]] + dict[keys[2]] + dict[keys[3]] + dict[keys[4]]]
            count += 1
        # print('x_new shape:', x_new.shape,'pcahnge shape:',pchange.shape)
        return x_new, pchange

    elif col == 17:
        # S1932,['dist'||'x', 'y', 'z'||'rsa'||'ph','temperature'||'C', 'H', 'O', 'N', 'other']
        # group_num = 5
        pchange = x[:,0,12:]
        #print('列属性数目为：', col)
        x_new = np.zeros(data_num*120*row*12).reshape(data_num,120,row,12)
        nums = itertools.permutations([0,1, 2, 3, 4])
        dict = {'0':[0],'1':[1,2,3],'2':[4],'3':[5,6],'4':[7,8,9,10,11]}
        count = 0
        for sequence in nums:
            keys = [str(i) for i in sequence]
            for k in range(data_num):
                temp = x[k]
                x_new[k,count,:,:] = temp[:,dict[keys[0]] + dict[keys[1]] + dict[keys[2]] + dict[keys[3]] + dict[keys[4]]]
            count+=1
        #print('x_new shape:', x_new.shape,'pcahnge shape:',pchange.shape)
        return x_new, pchange

def octant(x):
    x_new = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if i % 500 == 0:
            print('-----%dth mutation is being processed.'%i)
        data = pd.DataFrame(x[i])
        octant1 = data[(data[1] >= 0) & (data[2] >= 0) & (data[3] >= 0)]
        octant2 = data[(data[1] < 0) & (data[2] > 0) & (data[3] > 0)]
        octant3 = data[(data[1] < 0) & (data[2] < 0) & (data[3] > 0)]
        octant4 = data[(data[1] > 0) & (data[2] < 0) & (data[3] > 0)]
        octant5 = data[(data[1] > 0) & (data[2] > 0) & (data[3] < 0)]
        octant6 = data[(data[1] < 0) & (data[2] > 0) & (data[3] < 0)]
        octant7 = data[(data[1] < 0) & (data[2] < 0) & (data[3] < 0)]
        octant8 = data[(data[1] > 0) & (data[2] < 0) & (data[3] < 0)]
        temp_array = np.vstack((octant1,octant2,octant3,octant4,octant5,octant6,octant7,octant8))
        x_new[i] = temp_array
    return x_new
def reshape_tensor(x_train, x_test, x_val, nn_model):
    ## reshape array to Input shape
    if nn_model == 0:
        x_train = x_train.reshape((-1, x_train[1] * x_train[2]))
        x_test = x_test.reshape((-1, x_test[1] * x_test[2]))
        x_val = x_val.reshape((-1, x_val.shape[1], x_val.shape[2]))
    elif nn_model == 1:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    elif nn_model ==2:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))

    return x_train, x_test, x_val

def normalize(x_train, x_val, x_test, method=0):
    num_train, row_train, col_train = x_train.shape
    num_val, row_val, col_val = x_val.shape
    num_test, row_test, col_test = x_test.shape
    x_train = x_train.reshape((num_train * row_train, col_train))
    x_val = x_val.reshape((num_val*row_val, col_val))
    x_test = x_test.reshape((num_test * row_test, col_test))

    if method == 0:
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train -= mean
        x_train /= std
        x_val -= mean
        x_val /= std
        x_test -= mean
        x_test /= std
    elif method == 1:
        max_ = x_train.max(axis=0)
        x_train /= max_
        x_val /= max_
        x_test /= max_
    x_train = x_train.reshape((num_train,row_train,col_train))
    x_val = x_val.reshape((num_val,row_val,col_val))
    x_test = x_test.reshape((num_test,row_test,col_test))
    return x_train, x_val, x_test

def split_val(x_train,y_train,ddg_train,x_test,y_test,k):
    ddg_train = ddg_train.reshape(-1,1) ## vstack不支持 0D
    ## block this for efficiency
    # print('-' * 10, '验证集切分之前')
    # assert x_train[y_train.reshape(-1, ) == 1].shape[0] == y_train[y_train.reshape(-1, ) == 1].shape[0]
    # print('训练样本正类个数:', y_train[y_train.reshape(-1, ) == 1].shape[0])
    # assert x_train[y_train.reshape(-1, ) == 0].shape[0] == y_train[y_train.reshape(-1, ) == 0].shape[0]
    # print('训练样本负类个数:', y_train[y_train.reshape(-1, ) == 0].shape[0])
    # assert x_test[y_test.reshape(-1, ) == 1].shape[0] == y_test[y_test.reshape(-1, ) == 1].shape[0]
    # print('test样本正类个数:', y_test[y_test.reshape(-1, ) == 1].shape[0])
    # assert x_test[y_test.reshape(-1, ) == 0].shape[0] == y_test[y_test.reshape(-1, ) == 0].shape[0]
    # print('test样本负类个数:', y_test[y_test.reshape(-1, ) == 0].shape[0])

    ## 切分验证集
    #--------------------------------------
    # ##方法1
    # val_num = x_train.shape[0] // (k - 1)
    # x_val = x_train[:val_num]
    # y_val = y_train[:val_num]
    # x_train = x_train[val_num:]
    # y_train = y_train[val_num:]
    ##方法2
    #从训练数据中选出正类和负类数据
    x_p_train = x_train[y_train.reshape(-1, ) == 1]
    y_p_train = y_train[y_train.reshape(-1, ) == 1]
    ddg_p_train = ddg_train[y_train.reshape(-1, ) == 1]
    x_n_train = x_train[y_train.reshape(-1, ) == 0]
    y_n_train = y_train[y_train.reshape(-1, ) == 0]
    ddg_n_train = ddg_train[y_train.reshape(-1, ) == 0]
    #计算测试数据中正类和负类个数
    test_p_num = y_test[y_test.reshape(-1, ) == 1].shape[0]
    test_n_num = y_test[y_test.reshape(-1, ) == 0].shape[0]
    #按照测试数据的个数从训练数据中选出验证数据。
    x_p_val = x_p_train[:test_p_num]
    y_p_val = y_p_train[:test_p_num]
    ddg_p_val = ddg_p_train[:test_p_num]
    x_n_val = x_n_train[:test_n_num]
    y_n_val = y_n_train[:test_n_num]
    ddg_n_val = ddg_n_train[:test_n_num]
    #将验证数据组合起来
    x_val = np.vstack((x_p_val, x_n_val))
    y_val = np.vstack((y_p_val, y_n_val))
    ddg_val = np.vstack((ddg_p_val, ddg_n_val))
    #将剩余的训练数据组合起来
    x_p_train = x_p_train[test_p_num:]
    y_p_train = y_p_train[test_p_num:]
    ddg_p_train = ddg_p_train[test_p_num:]
    x_n_train = x_n_train[test_n_num:]
    y_n_train = y_n_train[test_n_num:]
    ddg_n_train = ddg_n_train[test_n_num:]
    x_train = np.vstack((x_p_train,x_n_train))
    y_train = np.vstack((y_p_train,y_n_train))
    ddg_train = np.vstack((ddg_p_train,ddg_n_train))
    #再将ddg转换成 0D
    ddg_train = ddg_train.reshape(-1)
    ddg_val = ddg_val.reshape(-1)

    ## block this for efficiency
    # print('-' * 10, '验证集切分之后')
    # assert x_train[y_train.reshape(-1, ) == 1].shape[0] == y_train[y_train.reshape(-1, ) == 1].shape[0]
    # print('训练样本正类个数:', y_train[y_train.reshape(-1, ) == 1].shape[0])
    # assert x_train[y_train.reshape(-1, ) == 0].shape[0] == y_train[y_train.reshape(-1, ) == 0].shape[0]
    # print('训练样本负类个数:', y_train[y_train.reshape(-1, ) == 0].shape[0])
    # assert x_val[y_val.reshape(-1, ) == 1].shape[0] == y_val[y_val.reshape(-1, ) == 1].shape[0]
    # print('val样本正类个数:', y_val[y_val.reshape(-1, ) == 1].shape[0])
    # assert x_val[y_val.reshape(-1, ) == 0].shape[0] == y_val[y_val.reshape(-1, ) == 0].shape[0]
    # print('val样本负类个数:', y_val[y_val.reshape(-1, ) == 0].shape[0])
    # assert x_test[y_test.reshape(-1, ) == 1].shape[0] == y_test[y_test.reshape(-1, ) == 1].shape[0]
    # print('test样本正类个数:', y_test[y_test.reshape(-1, ) == 1].shape[0])
    # assert x_test[y_test.reshape(-1, ) == 0].shape[0] == y_test[y_test.reshape(-1, ) == 0].shape[0]
    # print('test样本负类个数:', y_test[y_test.reshape(-1, ) == 0].shape[0])

    return x_train,y_train,ddg_train,x_val,y_val,ddg_val

if __name__ == '__main__':
    data = np.load('../datasets_array/S1925/k_neighbor/S1925_r_50.00_neighbor_50_class_5.npz')
    x = data['x']
    print('x_shape:',x.shape)
    # print(x[0,0:5,:])
    x_new, pchange = multi_channel(x)

    x_new = octant(x)
    print(x_new.shape)
    print(pchange.shape)
    # print(x_new[0, 0:5, :])