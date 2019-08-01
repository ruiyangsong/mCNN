#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : shuffle_data.py
# time      : 3/31/2019 15:06
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import os
import numpy as np

def shuffle_data(x,y,ddg):
    indices = [i for i in range(x.shape[0])]
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    ddg = ddg[indices]
    return x,y,ddg

def sort_atom(x,method = 2):
    '''
    x: 3D tensor, the axis are: sample_num, row_num and col_nm.
    method = [0,1,2]<-->sort by[dist, octant, random].
    '''
    data_num, row_num, col_num = x.shape
    if method == 2:
        indices = np.load('./global/k_neighbor1/indices_%d.npy'%row_num)
        for i in range(data_num):
            x[i] = x[i][indices]
    else:
        pass
    return x



# def shuffle_local_data(local_path):
#     """
#     :param local_path: 存储的未打乱顺序的张量路径.
#     :return: 打乱顺序的张量.
#     """
#     folder_list = os.listdir(local_path)  # ['ProNIT', 'S1932', 'S2648']
#     all_tensor_path_list = []
#     for folder in folder_list:
#         folder_path = local_path + folder + '/'
#         tensor_name_list = os.listdir(folder_path)
#         tensor_path_list = [folder_path + tensor_name for tensor_name in tensor_name_list]
#         for tensor_path in tensor_path_list:
#             all_tensor_path_list.append(tensor_path)
#
#     pass


if __name__ == '__main__':
    local_path = '../datasets_array/S1932/'
