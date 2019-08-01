#!～/anaconda3/env/bioinfo/bin/python
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
from shuffle_data import shuffle_data

def get_df_dataset(path_coord_csv, atom_class=5):
    """
    :参数:    path_coord_csv: 包含突变邻域坐标的csv文件的绝对路径(
                    r'E:\projects\dissertation\mCSM\mCSM-dataset\S2648\S2648_radius_10_onlyAtom_centerCA.csv').
              atom_class: 原子分类数
                            atom_class = 5 -- [C, H, O, N, other], by default
                                       = 1 -- [all in one]
                                       = 2 -- [亲水, 疏水]
                                       = 8 -- [药效团]
    :函数功能: 向包含邻域原子坐标的文件中追加“原子类别向量” 和 “pchange”.
    :调用:    df_dataset = get_df_dataset(path, atom_class).
    :return:  包含类别向量，pchange向量（追加到了每个原子）和坐标的DataFrame
    """
    # ==================== 只记录了氨基酸原子的字典 ====================
    # 为了描述突变氨基酸前后的性质变化，除了原子类别的变化以外，还可以参考不同氨基酸的其他属性，此处先只考虑氨基酸的原子类别和数目
    # 其他属性参照wiki: https://zh.wikipedia.org/wiki/%E6%B0%A8%E5%9F%BA%E9%85%B8
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

    aa_vec_dict = {}  # 氨基酸对应的”类别、个数“向量. aa_vec_dict = {'aa_name':[vec_of_atom_number by class]...}
    # 计算 aa_vec_dict.
    for aa_name in aa_atom_dict.keys():
        aa_name_vec = list(aa_atom_dict[aa_name].values())
        if len(aa_name_vec) == 4:
            aa_name_vec.append(0)
        aa_vec_dict[aa_name] = aa_name_vec

    f_coord = open(path_coord_csv, 'r')
    df_coord = pd.read_csv(f_coord)
    f_coord.close()

    print('-'*10,'开始追加原子类别向量，追加前df_coord形状为：', df_coord.shape,'\n列属性为:',df_coord.columns)

    if atom_class == 5:

        temp_array = np.zeros(len(df_coord))
        df_dataset = pd.DataFrame({'key':df_coord.key, 'pdb': df_coord.pdb, 'wild_type': df_coord.wild_type, 'chain': df_coord.chain,
                                   'position': df_coord.position, 'inode': df_coord.inode, 'rsa': df_coord.rsa,
                                   'mutant': df_coord.mutant, 'ph': df_coord.ph, 'temperature': df_coord.temperature,
                                   'ddg': df_coord.ddg, 'full_name': df_coord.full_name, 'name': df_coord.name,
                                   'dist': df_coord.dist, 'x': df_coord.x, 'y': df_coord.y, 'z': df_coord.z,
                                   'C': temp_array, 'H': temp_array, 'O': temp_array, 'N': temp_array, 'other': temp_array,
                                   'dC': temp_array, 'dH': temp_array, 'dO': temp_array, 'dN': temp_array, 'dother': temp_array})

        df_dataset.loc[df_dataset.name == 'C', 'C'] = 1
        df_dataset.loc[df_dataset.name == 'H', 'H'] = 1
        df_dataset.loc[df_dataset.name == 'O', 'O'] = 1
        df_dataset.loc[df_dataset.name == 'N', 'N'] = 1
        df_dataset.loc[(df_dataset.name != 'C') & (df_dataset.name != 'H') & (df_dataset.name != 'O') &
                       (df_dataset.name != 'N'), 'other'] = 1
        del df_coord

        print('-'*10,'原子类别向量追加完成，此时df_dataset形状为（包括非数值列属性）：', df_dataset.shape)
        print('-'*10,'下面进行pchange向量的追加.')
        #print('df_dataset的列数据类型为：\n', df_dataset.dtypes)

        # ========== 计算突变前后氨基酸变化向量append到每个原子后面（此处仅考虑了原子类别变化，后期可以扩展此处实现更多其他的方法） ==========

        ## 一行一行处理太慢了！！！
        # for i in range(len(df_dataset)):
        #     print('目前处理到第 %d 行'%i)
        #     wild_type = df_dataset.iloc[i, 1]
        #     mutant = df_dataset.iloc[i, 5]
        #     pchange = np.array(aa_vec_dict[mutant]) - np.array(aa_vec_dict[wild_type])
        #     df_dataset.iloc[i, 20:] = pchange

        ## 进行批量处理
        df_key_pchange = df_dataset.loc[:, ['wild_type', 'mutant']]  # 存储了此数据集中不同的 pchange 的 wild_type 和 mutant.
        df_key_pchange.drop_duplicates(keep='first', inplace=True)
        # print(len(df_key_pchange)) # 302
        for i in range(len(df_key_pchange)):
            if i % 20 == 0:
                print('目前处理到第 %d 种pchange,共 %d 种' % (i, len(df_key_pchange)))
            wild_type = df_key_pchange.iloc[i, 0]
            mutant = df_key_pchange.iloc[i, 1]
            pchange = np.array(aa_vec_dict[mutant]) - np.array(aa_vec_dict[wild_type])
            df_dataset.loc[(df_dataset.wild_type == wild_type) & (df_dataset.mutant == mutant),
                           ['dC', 'dH', 'dO', 'dN', 'dother']] = pchange


    elif atom_class == 1:
        pass
    elif atom_class == 2:
        pass
    elif atom_class == 8:
        pass

    print('-'*10,'pchange追加完成，此时df形状为（包括非数值列属性）：', df_dataset.shape)
    print('-'*10,'下面进行高度归一化和张量存储.')

    return df_dataset


def generate_dataset(path_mutation_csv, df_dataset, cutoff_step=0, k_neighbor=30):
    """
    :param path_mutation_csv: 描述突变信息的原csv文件的绝对路径(r'E:\projects\dissertation\mCSM\mCSM-dataset\S2648\S2648_new.csv')
    :param df_dataset: 包含了突变邻域 原子的坐标, 原子类别向量 和 pchange 向量的 DataFrame.
    :param cutoff_step: 根据距离中心[坐标为：(0,0,0)]的远近来讲数组标准化，默认为0，即在最后补0进行标准化.
    :函数功能: 由包含了坐标和原子类别向量的DataFrame以及突变的其他信息(PH, TEMPERATURE)，由给定的cutoff_step生成用于训练的数据集，
              数据集存储在numpy数组中(张量).对于不同长度的突变进行了标准化.
    :return:  返回包含数据集的numpy数组(张量), 注：每一列没有进行标准化.
    """
    f_mutation = open(path_mutation_csv, 'r')
    df_mutation = pd.read_csv(f_mutation)
    f_mutation.close()
    mutation_num = len(df_mutation)

    size_list = []
    y = []
    ddg_value = []
    if k_neighbor == 0:
    # ==================== 计算单点突变邻域中最多的原子数是多少 ====================
        print('-'*10,'计算数据的height...\n')
        for i in range(mutation_num):
            row = df_mutation.iloc[i, :] #每一个突变信息
            inode = ' '
            try:
                int(row.POSITION)
            except:
                # print(type(row.POSITION))# str
                inode = row.POSITION[-1:]
                row.POSITION = row.POSITION[0:-1]

            [key, pdb, wild_type, chain, position, inode, mutant] = [row.key, row.PDB[0:4], row.WILD_TYPE, row.CHAIN,
                                                                     int(row.POSITION), inode, row.MUTANT]
            temp_df = df_dataset[(df_dataset['key']==key) & (df_dataset['pdb']==pdb) & (df_dataset['wild_type']==wild_type)
                                 & (df_dataset['chain']==chain) & (df_dataset['position']==position)
                                 & (df_dataset['inode']==inode) & (df_dataset['mutant']==mutant)]
            size_list.append(len(temp_df))
        height = max(size_list)  # 此数据集的高

    else:
        height = k_neighbor
    if np.sum(df_dataset.rsa.values) == 0:
        print('此数据集不包含RSA信息！')
        width = 16 # 15列,dist-16列
    else:
        print('此数据集包含RSA信息！')
        width = 17 # 15列,加上rsa,dist-17列

    if cutoff_step == 0:
        print('-'*10,'将数据归一化为：%d * %d，此数据集中共 %d 个突变' % (height, width, mutation_num))
        data_array = np.zeros((height, width))  # 存储最终数据点的array

        for i in range(mutation_num):
            if i % 200 == 0:
                print('目前处理到第 %d 个突变,共 %d 个' % (i, mutation_num))

            row = df_mutation.iloc[i, :]
            inode = ' '
            try:
                int(row.POSITION)
            except:
                # print(type(row.POSITION))# str
                inode = row.POSITION[-1:]
                print('pdbid: %s, inode: %s' % (row.PDB[0:4], row.POSITION[-1:]))
                row.POSITION = row.POSITION[0:-1]
                print(row.POSITION)
            [key, pdb, wild_type, chain, position, inode, mutant, ddg] = [row.key, row.PDB[0:4], row.WILD_TYPE, row.CHAIN,
                                                                          int(row.POSITION), inode, row.MUTANT, row.DDG]

            temp_df = df_dataset[
                (df_dataset['pdb'] == pdb) & (df_dataset['wild_type'] == wild_type) & (df_dataset['chain'] == chain) &
                (df_dataset['position'] == position) & (df_dataset['inode'] == inode) &
                (df_dataset['mutant'] == mutant) & (df_dataset['key'] == key)]

            if width == 16:
                temp_df = temp_df.loc[:,
                          ['dist', 'x', 'y', 'z', 'ph', 'temperature', 'C', 'H', 'O', 'N', 'other',
                           'dC', 'dH', 'dO', 'dN', 'dother']]
            else:
                temp_df = temp_df.loc[:, ['dist', 'x', 'y', 'z', 'rsa', 'ph', 'temperature', 'C', 'H', 'O', 'N', 'other',
                                          'dC', 'dH', 'dO', 'dN', 'dother']]

            temp_array = temp_df.values
            # print('temp_array的数据类型为：',temp_array.dtype) #float
            gap = height - temp_array.shape[0]
            if gap == 0:
                data_array = np.vstack((data_array, temp_array))
            else:
                gap_array = np.zeros((gap, width))
                temp_array = np.vstack((temp_array, gap_array))
                assert temp_array.shape[0] == height
                data_array = np.vstack((data_array, temp_array))
                assert data_array.shape[0] % height == 0

            ddg_value.append(ddg)
            if ddg >= 0:
                assert ddg >= 0
                y.append(1)
            else:
                assert ddg < 0
                y.append(0)
        ddg_value = np.array(ddg_value) # 0D张量
        y = np.array(y)
        y = y.reshape(-1, 1)

        data_array = data_array.reshape(-1, height, width)
        data_array = data_array[1:]

        print('数据归一化以及张量表示完成，张量形状如下：')
        print('data_array shape', data_array.shape)
        print('y shape', y.shape)
        print('ddg_value shape',ddg_value.shape)

    return data_array, y, ddg_value

def save_data_array(x,y,ddg_value,dataset_name, radius, k_neighbor, class_num, dist=0):
    """
    :return: 打乱并存储数据到磁盘.
    """
    path_dist = '../datasets_array/' + dataset_name + '/dist/'
    path_k_neighbor = '../datasets_array/' + dataset_name + '/k_neighbor/'
    path_radius = '../datasets_array/' + dataset_name + '/radius/'
    if not os.path.exists(path_dist):
        os.mkdir(path_dist)
    if not os.path.exists(path_k_neighbor):
        os.mkdir(path_k_neighbor)
    if not os.path.exists(path_radius):
        os.mkdir(path_radius)
    ## 打乱数据
    x,y,ddg_value = shuffle_data(x,y,ddg_value)

    print('-'*10,'数据已经打乱,开始写入磁盘...')
    ## 存储dist数据
    if dist == 1:
        np.savez('../datasets_array/%s/dist/%s_dist_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num), x=x,y=y,ddg = ddg_value)
        print('-' * 10, '包含距离矩阵的张量数组已存储到本地磁盘！')
    ## 只考虑邻域
    elif k_neighbor != 0:
        np.savez('../datasets_array/%s/k_neighbor/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num), x=x,y=y,ddg=ddg_value)  # (2648, 437, 15), (2648, 30, 15)
        print('-' * 10, '包含 邻域环境 的坐标数据的张量数组已存储到本地磁盘！')
    ## 只考虑近邻
    else:
        np.savez('../datasets_array/%s/radius/%s_r_%.2f_neighbor_%d_class_%d.npz' % (
            dataset_name, dataset_name, radius, k_neighbor, class_num), x=x,y=y,ddg=ddg_value)  # (2648, 437, 15), (2648, 30, 15)
        print('-' * 10, '包含 k近邻 的坐标数据的张量数组已存储到本地磁盘！')


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
