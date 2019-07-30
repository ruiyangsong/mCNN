#!～/anaconda3/envs/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : generate_dataset.py
# time      : 3/11/2019 21:10
# author    : ruiyang
# email     : ww_sry@163.com

import sys
import os
import numpy as np
import pandas as pd
from transform_data_array import transform_data_array

def get_df_dataset(df_coord, atom_class=5):
    '''
    :param df_coord: pandas DataFrame stored neighboring atom coords of the whole dataset.
    :param atom_class: class number of atoms.
    atom_class = 5 -- [C, H, O, N, other], by default
               = 1 -- [All in one]
               = 2 -- [Hydrophilic, Hydrophobic]
               = 8 -- [Pharmacophore]
    :return: pandas DataFrame appends vector $\Deata_r$.
    '''
    # python dictionary which records aa's atom number of each class.
    # more properties of different aas can be foun here: https://zh.wikipedia.org/wiki/%E6%B0%A8%E5%9F%BA%E9%85%B8.
    aa_atom_dict = {'A': {'C': 3, 'H': 7, 'O': 2, 'N': 1},
                    'R': {'C': 6, 'H': 14, 'O': 2, 'N': 4},
                    'N': {'C': 4, 'H': 8, 'O': 3, 'N': 2},
                    'D': {'C': 4, 'H': 7, 'O': 4, 'N': 1},
                    'C': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
                    'Q': {'C': 5, 'H': 10, 'O': 3, 'N': 2},
                    'E': {'C': 5, 'H': 9, 'O': 4, 'N': 1},
                    'G': {'C': 2, 'H': 5, 'O': 2, 'N': 1},
                    'H': {'C': 6, 'H': 9, 'O': 2, 'N': 3},
                    'I': {'C': 6, 'H': 13, 'O': 2, 'N': 1},
                    'L': {'C': 6, 'H': 13, 'O': 2, 'N': 1},
                    'K': {'C': 6, 'H': 14, 'O': 2, 'N': 2},
                    'M': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 1},
                    'F': {'C': 9, 'H': 11, 'O': 2, 'N': 1},
                    'P': {'C': 5, 'H': 9, 'O': 2, 'N': 1},
                    'S': {'C': 3, 'H': 7, 'O': 3, 'N': 1},
                    'T': {'C': 4, 'H': 9, 'O': 3, 'N': 1},
                    'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2},
                    'Y': {'C': 9, 'H': 11, 'O': 3, 'N': 1},
                    'V': {'C': 5, 'H': 11, 'O': 2, 'N': 1}
                    }

    aa_vec_dict = {} # aa_vec_dict = {'aa_name':[vec_of_atom_number by class], ...}
    ## Calc aa_vec_dict.
    for aa_name in aa_atom_dict.keys():
        aa_vec = list(aa_atom_dict[aa_name].values())
        if len(aa_vec) == 4:
            aa_vec.append(0)
        aa_vec_dict[aa_name] = aa_vec

    print('Begin append atom_vec and delta_r, shape of df_coord:', df_coord.shape, 'columns are:\n', df_coord.columns)

    if atom_class == 5:
        temp_array = np.zeros(len(df_coord))
        df_dataset = pd.DataFrame({'key':df_coord.key, 'pdb': df_coord.pdb, 'wild_type': df_coord.wild_type,
                                   'chain': df_coord.chain, 'position': df_coord.position, 'inode': df_coord.inode,
                                   'rsa': df_coord.rsa,'mutant': df_coord.mutant, 'ph': df_coord.ph,
                                   'temperature': df_coord.temperature,'ddg': df_coord.ddg,
                                   'full_name': df_coord.full_name, 'name': df_coord.name, 'dist': df_coord.dist,
                                   'x': df_coord.x, 'y': df_coord.y, 'z': df_coord.z, 'C': temp_array, 'H': temp_array,
                                   'O': temp_array, 'N': temp_array, 'other': temp_array, 'dC': temp_array,
                                   'dH': temp_array, 'dO': temp_array, 'dN': temp_array, 'dother': temp_array})

        df_dataset.loc[df_dataset.name == 'C', 'C'] = 1
        df_dataset.loc[df_dataset.name == 'H', 'H'] = 1
        df_dataset.loc[df_dataset.name == 'O', 'O'] = 1
        df_dataset.loc[df_dataset.name == 'N', 'N'] = 1
        df_dataset.loc[(df_dataset.name != 'C') & (df_dataset.name != 'H') & (df_dataset.name != 'O') &
                       (df_dataset.name != 'N'), 'other'] = 1
        del df_coord
        print('Append atom_vec done, begin append delta_r.')
        #print('data types of df_dataset\'s columns:\n', df_dataset.dtypes)

        ## processing by different types of delta_r.
        df_key_delta_r = df_dataset.loc[:, ['wild_type', 'mutant']]
        df_key_delta_r.drop_duplicates(keep='first', inplace=True)
        num_type_delta_r = len(df_key_delta_r)
        print('Number of different types of delta_r: ', num_type_delta_r) # 302
        for i in range(num_type_delta_r):
            if i % 100 == 0:
                print('Processing %d-th delta_r' % (i))
            wild_type = df_key_delta_r.iloc[i, 0]
            mutant = df_key_delta_r.iloc[i, 1]
            delta_r = np.array(aa_vec_dict[mutant]) - np.array(aa_vec_dict[wild_type])
            df_dataset.loc[(df_dataset.wild_type == wild_type) & (df_dataset.mutant == mutant),
                           ['dC', 'dH', 'dO', 'dN', 'dother']] = delta_r

    elif atom_class == 1:
        pass
    elif atom_class == 2:
        pass
    elif atom_class == 8:
        pass

    print('Append delta_r done, shape of df_dataset:', df_dataset.shape, 'columns are:\n', df_dataset.columns)
    return df_dataset


def generate_dataset(num_mutation, df_dataset, cutoff_step=0, k_neighbor=30):
    '''
    :param num_mutation: number of mutations in this dataset.
    :param df_dataset: pandas DataFrame which contains delta_r calculated by function get_df_dataset.
    :param cutoff_step: float, distance to scan the matrix (2D Tensor). 0 is the default option.
    :param k_neighbor: int, k neighboring atoms.
    :return: 3D Tensor which stored the numerical representation of this dataset.
    '''
    print('Begin normalize height and array storage.')
    y = []
    ddg_value = []

    if k_neighbor == 0:
        height = max(df_dataset.key.value_counts())
    else:
        height = k_neighbor
    if np.sum(df_dataset.rsa.values) == 0:
        print('Without RSA information in this dataset.')
        width = 16
    else:
        print('There are RSA information in this dataset.')
        width = 17

    data_array = np.zeros((height, width))  # the final 3D array for storing this dataset.
    if cutoff_step == 0:
        print('This dataset contains %d mutations, every mutation is 2D array with shape: %d * %d'
              % (num_mutation, height, width))
        for key in range(num_mutation):
            if key % 500 == 0:
                print('processing %d-th mutation' % key)
            df_key = df_dataset[df_dataset['key'] == key]
            ddg = df_key.iloc[0, df_key.columns.get_loc('ddg')]
            # print(ddg)
            if width == 16:
                df_key = df_key.loc[:, ['dist', 'x', 'y', 'z', 'ph', 'temperature', 'C', 'H', 'O', 'N', 'other',
                                        'dC', 'dH', 'dO', 'dN', 'dother']]
            else:
                df_key = df_key.loc[:, ['dist', 'x', 'y', 'z', 'rsa', 'ph', 'temperature', 'C', 'H', 'O', 'N', 'other',
                                        'dC', 'dH', 'dO', 'dN', 'dother']]

            key_array = df_key.values
            # print('dtype of key_array is:' ,temp_array.dtype) #float
            gap = height - key_array.shape[0]
            if gap == 0:
                data_array = np.vstack((data_array, key_array))
            else:
                gap_array = np.zeros((gap, width))
                key_array = np.vstack((key_array, gap_array))
                assert key_array.shape[0] == height
                data_array = np.vstack((data_array, key_array))
                assert data_array.shape[0] % height == 0

            ddg_value.append(ddg)
            if ddg >= 0:
                y.append(1)
            else:
                y.append(0)
        ddg_value = np.array(ddg_value) # 0D Tensor
        y = np.array(y)
        y = y.reshape(-1, 1)

        data_array = data_array.reshape(-1, height, width)
        data_array = data_array[1:]

        print('normalize height and array representation is done, tensor shape as fellows:')
        print('data_array shape', data_array.shape, 'y shape', y.shape, 'ddg_value shape', ddg_value.shape)

    return data_array, y, ddg_value

def save_data_array(x,y,ddg_value,dataset_name, radius, k_neighbor, class_num):
    path_k_neighbor = '../datasets_array/' + dataset_name + '/k_neighbor/'
    path_radius = '../datasets_array/' + dataset_name + '/radius/'
    if not os.path.exists(path_k_neighbor):
        os.mkdir(path_k_neighbor)
    if not os.path.exists(path_radius):
        os.mkdir(path_radius)

    if k_neighbor != 0:
        np.savez('../datasets_array/%s/k_neighbor/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num), x=x,y=y,ddg=ddg_value)
    else:
        np.savez('../datasets_array/%s/radius/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num), x=x,y=y,ddg=ddg_value)
    print('The 3D array which stored numerical representation has stored in local hard drive.')

if __name__ == '__main__':
    dataset_name, radius, k_neighbor, class_num = sys.argv[1:]
    radius = float(radius)
    k_neighbor = int(k_neighbor)
    class_num = int(class_num)

    dataset_path = '../datasets/%s/' % dataset_name
    mutation_csv_path = dataset_path + '%s_new.csv' % dataset_name  # 描述突变信息的csv文件路径

    ## ==================== test function: get_df_dataset ====================
    coord_csv_path = dataset_path + dataset_name + '_r_%.2f_neighbor_%d_onlyAtom_centerCA.csv' % (radius, k_neighbor)

    df_dataset = get_df_dataset(coord_csv_path, atom_class=class_num) #包含了 原子类别向量 和 pchange 的df

    # dataset_csv_path = dataset_path + dataset_name + '_r_%.2f_neighbor_%d_onlyAtom_centerCA_%dclassVec.csv' % (radius, k_neighbor, class_num)
    # df_dataset.to_csv(dataset_csv_path, index=False)
    # ==================== test function: generate_dataset ====================
    # f = open(dataset_csv_path, 'r')
    # df_dataset = pd.read_csv(f)
    # f.close()

    ##计算并保存x,y
    x, y, ddg_value = generate_dataset(mutation_csv_path, df_dataset, cutoff_step=0, k_neighbor = k_neighbor)
    save_data_array(x, y, ddg_value, dataset_name, radius, k_neighbor, class_num)

    ## 计算并保存dist_array
    # print('-' * 10, '正在将坐标数据转换成距离矩阵...')
    # x_dist = transform_data_array(x)
    # y_dist = y
    # save_data_array(x_dist, y_dist,ddg_value, dataset_name, radius, k_neighbor, class_num, dist=1)


 # ##单独转换时使用此段代码
    # path = '../datasets_array/S1932/radius/'
    # tensor_name_list = os.listdir(path)
    # radius = 4
    # for data in tensor_name_list:
    #
    #     data_path = path+data
    #     print(data_path)
    #     arraytemp = np.load(data_path)
    #     x = arraytemp['x']
    #     y = arraytemp['y']
    #     print('-' * 10, '正在将坐标数据转换成距离矩阵...')
    #     x_dist = transform_data_array(x)
    #     y_dist = y
    #     save_data_array(x_dist, y_dist, dataset_name, radius, k_neighbor, class_num, dist=1)
    #     radius+=2
