#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : sampling.py
# time      : 4/6/2019 14:13
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from shuffle_data import shuffle_data

def oversampling(x_train, y_train):
    ## 过采样
    # print('-' * 10, '对切分后的训练集进行过采样...')
    train_num,train_row,train_col = x_train.shape
    x_train = x_train.reshape((train_num,train_row*train_col))
    y_train = y_train.reshape(train_num)

    ros = RandomOverSampler()
    x_train_new, y_train_new = ros.fit_sample(x_train, y_train)

    # ddg_train_new = []
    # for each_data in x_train_new:
    #     for i in range(train_num):
    #         if all(x_train[i] == each_data):
    #             ddg_train_new.append(ddg_train[i])
    #         else:
    #             print('WRONG!!!!!!!!!!!!!!!!!!!!')
    # ddg_train = np.array(ddg_train_new)

    x_train = x_train_new.reshape((-1,train_row,train_col))
    y_train = y_train_new.reshape((-1,1))

    ## block this for efficiency
    # print('-' * 10, '过采样之后')
    # assert x_train[y_train.reshape(-1, ) == 1].shape[0] == y_train[y_train.reshape(-1, ) == 1].shape[0]
    # print('训练样本正类个数:', y_train[y_train.reshape(-1, ) == 1].shape[0])
    # assert x_train[y_train.reshape(-1, ) == 0].shape[0] == y_train[y_train.reshape(-1, ) == 0].shape[0]
    # print('训练样本负类个数:', y_train[y_train.reshape(-1, ) == 0].shape[0])

    return x_train, y_train

def undersampling(x,y,under_threshold):
    """
    :function: 对样本进行欠采样操作.
    :param x: datasets.
    :param y: label.
    :under_threshold: threshold of unbalanced rate,欠采样时考虑的不均衡比的阈值,超过此阈值才进行欠采样操作.
    :return: balanced datasets and labels.
    """
    x_p = x[y.reshape(-1, ) == 1]
    x_n = x[y.reshape(-1, ) == 0]
    p_num = x_p.shape[0]
    n_num = x_n.shape[0]
    if p_num >= n_num:
        rate = p_num / n_num
    else:
        rate = n_num / p_num
    print('-' * 10, '正负类数据量分别为: %d,%d数据不均衡比为：%.2f, 数据集形状为%r' % (p_num, n_num, rate, x.shape))
    if rate >= under_threshold:
        print('-' * 10, '进行随机欠采样操作')
        if p_num > n_num:
            x_p = x_p[np.random.randint(low=x_p.shape[0], size=int(x_p.shape[0] / rate))]
        else:
            x_n = x_n[np.random.randint(low=x_n.shape[0], size=int(x_n.shape[0] / rate))]
        # x_p = x_p[np.random.randint(low=x_p.shape[0],size=int(x_p.shape[0]/rate))] if p_num > n_num else x_n = x_n[np.random.randint(low=x_n.shape[0],size=int(x_n.shape[0]/rate))] # 取 1/3
        p_num = x_p.shape[0]
        n_num = x_n.shape[0]
        print('此时正负类数据量分别为：%d, %d' % (p_num, n_num))
        x = np.vstack((x_p, x_n))
        y = np.vstack((np.ones((p_num, 1), dtype=np.float32), np.zeros((n_num, 1), dtype=np.float32)))
        ## 打乱数据
        x,y = shuffle_data(x,y)

    return x,y