#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : read_array.py
# time      : 4/6/2019 14:36
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import numpy as np

def load_data(dataset_name, radius, k_neighbor, class_num,dist,nn_model=1):
    """
    :param dataset_name: dataset_name, exm: S1932
    :param radius: 邻域半径.
    :param k_neighbor: k近邻.
    :param class_num: 原子类别数.
    :param dist: 是否加载dist数据集.0不加载,1加载.
    :return: x,y
    """
    print('-' * 10, 'loading data...')
    if dist == 1:
        # print('-' * 10, 'dataset_name is: %s_x_dist_r_%.2f_neighbor_%d_class_%d.npz' % (
        #     dataset_name, radius, k_neighbor, class_num))
        data = np.load('../datasets_array/%s/dist/%s_dist_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num))
        x = data['x']
        y = data['y']
        data_num = x.shape[0]
        ddg = np.zeros(data_num)  # 当数据中没有ddg时，默认取为0
        if nn_model >= 2:
            ddg = data['ddg']
    elif k_neighbor != 0:
        # print('-' * 10, 'dataset_name is: %s_x_r_%.2f_neighbor_%d_class_%d.npz' % (
        #     dataset_name, radius, k_neighbor, class_num))
        data = np.load('../datasets_array/%s/k_neighbor/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num))
        x = data['x']
        y = data['y']
        data_num = x.shape[0]
        ddg = np.zeros(data_num)  # 当数据中没有ddg时，默认取为0
        if nn_model >= 2:
            ddg = data['ddg']
    else:
        # print('-' * 10, 'dataset_name is: %s_x_r_%.2f_neighbor_%d_class_%d.npz' % (
        #     dataset_name, radius, k_neighbor, class_num))
        data = np.load('../datasets_array/%s/radius/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num))
        x = data['x']
        y = data['y']
        data_num = x.shape[0]
        ddg = np.zeros(data_num)  # 当数据中没有ddg时，默认取为0
        if nn_model >= 2:
            ddg = data['ddg']
    # assert x.shape[0] == ddg.shape[0]
    # assert x.shape[0] == y.shape[0]
    return x,y,ddg
