#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : transform_data_array.py
# time      : 3/20/2019 21:03
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import numpy as np
from scipy.spatial.distance import cdist

def transform_data_array(dataset_array):
    """
    :function: 计算原子两两之间的距离矩阵,并用此矩阵替代原始 dataset_x_neighbork.npy 中的坐标部分.
    :param data_array_path: PATH of dataset_x_neighbork.npy, which is np_array.
    :return: dataset_x_dist_neighbork.npy.
    """
    dataset_x = []
    [data_num,row_num,col_num] = dataset_array.shape
    dataset_coord = dataset_array[:,:,1:4]
    for data in dataset_coord:
        dataset_x.append(cdist(data,data,metric='euclidean'))
    dataset_x = np.array(dataset_x).reshape((-1,row_num))
    dataset_x = np.hstack((dataset_array[:,:,0].reshape((-1,1)),dataset_x))

    dataset_x = np.hstack((dataset_x,dataset_array.reshape((-1,col_num))[:,4:]))
    dataset_x = dataset_x.reshape((data_num,row_num,col_num-3+row_num))
    print('-'*10,'将坐标换成距离矩阵后的 dataset_x 的形状为：',dataset_x.shape)

    return dataset_x

if __name__ == '__main__':
    data_array_path = '../datasets_array/dataset_x_neighbor30.npy'
    dataset_array = np.load(data_array_path)
    dataset_x_dist_neighbor30 = transform_data_array(dataset_array)
    np.save('../datasets_array/dataset_x_dist_neighbor30.npy', dataset_x_dist_neighbor30)
