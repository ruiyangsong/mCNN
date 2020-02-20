#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file_name : Data.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DataPacker(object):
    def x_y_ddg(self,x,y,ddg):
        self.x = x
        self.y = y
        self.ddg = ddg
    def train_test_val(self,train_data,test_data,val_data):
        self.Train = train_data
        self.Test  = test_data
        self.Val   = val_data

class DataLoader(DataPacker):
    '''Load data and sort row'''
    def __init__(self,container,sort_method,permuation_seed):
        '''load as npz data_object'''
        self.sort = sort_method
        self.seed = permuation_seed
        # mCNN
        self.mCNN     = None
        self.mCNN_val = None
        # mCSM
        self.mCSM     = None
        self.mCSM_val = None

        self.load_data(container)

    def sort_row(self,x):
        '''
        :param x: 3D tensor of this dataset, the axis are: data_num, row_num and col_nm.
        :param method: str, row sorting method.
        :return: 3D tensor after sort.
        '''
        if self.sort == 'chain':
            return x
        data_num, row_num, col_num = x.shape
        if self.sort == 'distance':
            for i in range(data_num):
                indices = x[i, :, 0].argsort()
                x[i] = x[i, [indices]]
            return x
        elif self.sort == 'octant':
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
        elif self.sort == 'permutation1':
            indices = np.load('../global/permutation1/indices_%d.npy' % row_num)
        elif self.sort == 'permutation2':
            indices = np.load('../global/permutation2/indices_%d.npy' % row_num)
        elif self.sort == 'permutation':
            indices = [i for i in range(row_num)]
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        for i in range(data_num):
            x[i] = x[i][indices]
        return x


    def load_data(self,container):
        # mCNN
        if container['mCNN_wild_dir'] != '' and container['mCNN_mutant_dir'] != '':
            mCNN_wild = np.load(container['mCNN_wild_dir'])
            mCNN_mutant = np.load(container['mCNN_mutant_dir'])
            x = np.vstack((self.sort_row(mCNN_wild['x']),self.sort_row(mCNN_mutant['x'])))
            y = np.vstack((mCNN_wild['y'],mCNN_mutant['y']))
            ddg = np.vstack((mCNN_wild['ddg'],mCNN_mutant['ddg']))
            self.mCNN = self.x_y_ddg(x,y,ddg)
            # Val data
            if container['val_mCNN_wild_dir'] != '' and container['val_mCNN_mutant_dir'] != '':
                mCNN_wild_val = np.load(container['val_mCNN_wild_dir'])
                mCNN_mutant_val = np.load(container['val_mCNN_mutant_dir'])
                x = np.vstack((self.sort_row(mCNN_wild_val['x']), self.sort_row(mCNN_mutant_val['x'])))
                y = np.vstack((mCNN_wild_val['y'], mCNN_mutant_val['y']))
                ddg = np.vstack((mCNN_wild_val['ddg'], mCNN_mutant_val['ddg']))
                self.mCNN_val = self.x_y_ddg(x, y, ddg)
            elif container['val_mCNN_wild_dir'] != '' or container['val_mCNN_mutant_dir'] != '':
                print('[ERROR] when mCNN_wild and mCNN_mutant are both used, [val_mCNN_wild, val_mCNN_mutant] should assigned None or Two, but one is assigned')
                exit(0)

        elif container['mCNN_wild_dir'] == '' and container['mCNN_mutant_dir'] != '':
            mCNN_mutant = np.load(container['mCNN_mutant_dir'])
            x, y, ddg = self.sort_row(mCNN_mutant['x']), mCNN_mutant['y'], mCNN_mutant['ddg']
            self.mCNN = self.x_y_ddg(x,y,ddg)
            if container['val_mCNN_wild_dir'] == '' and container['val_mCNN_mutant_dir'] != '':
                mCNN_mutant_val = np.load(container['val_mCNN_mutant_dir'])
                x,y,ddg = self.sort_row(mCNN_mutant_val['x']),mCNN_mutant_val['y'],mCNN_mutant_val['ddg']
                self.mCNN_val = self.x_y_ddg(x,y,ddg)

        elif container['mCNN_wild_dir'] != '' and container['mCNN_mutant_dir'] == '':
            mCNN_wild = np.load(container['mCNN_wild_dir'])
            x, y, ddg = self.sort_row(mCNN_wild['x']), mCNN_wild['y'], mCNN_wild['ddg']
            self.mCNN = self.x_y_ddg(x, y, ddg)
            if container['val_mCNN_wild_dir'] != '' and container['val_mCNN_mutant_dir'] == '':
                mCNN_wild_val = np.load(container['val_mCNN_wild_dir'])
                x,y,ddg = self.sort_row(mCNN_wild_val['x']),mCNN_wild_val['y'],mCNN_wild_val['ddg']
                self.mCNN_val = self.x_y_ddg(x,y,ddg)

        ## mCSM
        if container['mCSM_wild_dir'] != '' and container['mCSM_mutant_dir'] != '':
            mCSM_wild = np.load(container['mCSM_wild_dir'])
            mCSM_mutant = np.load(container['mCSM_mutant_dir'])
            x = np.vstack((self.sort_row(mCSM_wild['x']),self.sort_row(mCSM_mutant['x'])))
            y = np.vstack((mCSM_wild['y'],mCSM_mutant['y']))
            ddg = np.vstack((mCSM_wild['ddg'],mCSM_mutant['ddg']))
            self.mCNN = self.x_y_ddg(x,y,ddg)
            if container['val_mCSM_wild_dir'] != '' and container['val_mCSM_mutant_dir'] != '':
                mCSM_wild_val = np.load(container['val_mCSM_wild_dir'])
                mCSM_mutant_val = np.load(container['val_mCSM_mutant_dir'])
                x = np.vstack((self.sort_row(mCSM_wild_val['x']), self.sort_row(mCSM_mutant_val['x'])))
                y = np.vstack((mCSM_wild_val['y'], mCSM_mutant_val['y']))
                ddg = np.vstack((mCSM_wild_val['ddg'], mCSM_mutant_val['ddg']))
                self.mCSM_val = self.x_y_ddg(x, y, ddg)
            elif container['val_mCSM_wild_dir'] or container['val_mCSM_mutant_dir']:
                print('[ERROR] when mCSM_wild and mCSM_mutant are both used, [val_mCSM_wild, val_mCSM_mutant] should assigned None or Two, but one is assigned')
                exit(0)

        elif container['mCSM_wild_dir'] == '' and container['mCSM_mutant_dir'] != '':
            mCSM_mutant = np.load(container['mCSM_mutant_dir'])
            x, y, ddg = self.sort_row(mCSM_mutant['x']), mCSM_mutant['y'], mCSM_mutant['ddg']
            self.mCSM = self.x_y_ddg(x,y,ddg)
            if container['val_mCNN_wild_dir'] == '' and container['val_mCNN_mutant_dir'] != '':
                mCSM_mutant_val = np.load(container['val_mCSM_mutant_dir'])
                x,y,ddg = self.sort_row(mCSM_mutant_val['x']),mCSM_mutant_val['y'],mCSM_mutant_val['ddg']
                self.mCSM_val = self.x_y_ddg(x,y,ddg)

        elif container['mCSM_wild_dir'] != '' and container['mCSM_mutant_dir'] == '':
            mCSM_wild = np.load(container['mCSM_wild_dir'])
            x, y, ddg = self.sort_row(mCSM_wild['x']), mCSM_wild['y'], mCSM_wild['ddg']
            self.mCSM = self.x_y_ddg(x, y, ddg)
            if container['val_mCSM_wild_dir'] != '' and container['val_mCSM_mutant_dir'] == '':
                mCSM_wild_val = np.load(container['val_mCSM_wild_dir'])
                x,y,ddg = self.sort_row(mCSM_wild_val['x']),mCSM_wild_val['y'],mCSM_wild_val['ddg']
                self.mCSM_val = self.x_y_ddg(x,y,ddg)

        assert self.mCNN is not None or self.mCSM is not None

        if self.mCNN is not None and self.mCSM is not None:
            if (self.mCNN_val is not None and self.mCSM_val is None) or (self.mCNN_val is None and self.mCSM_val is not None):
                print('[ERROR] when mCNN and mCSM are both used, [mCNN_val, mCSM_val] should assigned None or Two, but one is assigned')
                exit(0)


class DataExtractor(DataLoader):
    '''
    Return a list, which elements are data for each fold (described by python dictionary).

    * e.g.: [{'feature_type': data_object}]
    
    The dict is organized by the following:
        {
        'mCNN': data_object,
        'mCSM': data_object,
        'other_feature_type': data_object,
        ...
        }

    The data_object is organized by the following:
    * e.g.: data_object:
                * data_object.Train
                * data_object.Test
                * data_object.Val
                    * data_object.Val.x
                    * data_object.Val.y
                    * data_object.Val.ddg
    
    ++ NOTICE: validation data are THE SAME for each fold. ++
    '''

    def __init__(self,container, sort_method, permuation_seed, normalize_method = 'norm', val=True):
        super().__init__(container,sort_method,permuation_seed)

        self.norm_method = normalize_method
        self.val = val
        self.Train = None
        self.Test  = None
        self.Val   = None

        self.data_lst = []


    def given_kfold(self, data_lst):
        '''data_lst = [fold1_data_object, fold2_data_object, ...]'''
        ################################################################################################################
        # 此处需完善 sort_row 和 normalization
        ################################################################################################################
        self.data_lst = data_lst

    def split_kfold(self, fold_num, random_seed=10, train_ratio = 0.7):

        if fold_num >= 3:
            val_num = int(self.mCNN.x.shape[0] / fold_num)  # val_num are same for mCNN and mCSM !
            skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=random_seed)
            if self.mCNN is not None and self.mCSM is not None:
                # mCNN
                if self.mCNN_val is not None:
                    # val_data is all the same for each fold
                    x_val, y_val, ddg_val = self.get_val(self.mCNN_val, random_seed, val_num)
                    x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                for train_index, test_index in skf.split(self.mCNN.x, self.mCNN.y):
                    tmp_dict = {}
                    x_y_ddg_train = self.x_y_ddg(self.mCNN.x[train_index], self.mCNN.y[train_index], self.mCNN.ddg[train_index])
                    x_y_ddg_test = self.x_y_ddg(self.mCNN.x[test_index], self.mCNN.y[test_index], self.mCNN.ddg[test_index])
                    if self.mCNN_val is None and self.val:
                        x_train, y_train, ddg_train, x_val, y_val, ddg_val = self.split_val(x_y_ddg_train, x_y_ddg_test.ddg, random_seed)
                        x_y_ddg_train = self.x_y_ddg(x_train, y_train, ddg_train)
                        x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                    elif self.mCNN_val is None and self.val is False:
                        x_y_ddg_val = None
                    # Nomalization
                    if x_y_ddg_val is None:
                        x_train, x_test= self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=None, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                    else:
                        x_train, x_test, x_val = self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=x_y_ddg_val.x, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                        x_y_ddg_val.x = x_val

                    tmp_dict['mCNN'] = self.train_test_val(x_y_ddg_train,x_y_ddg_test,x_y_ddg_val)
                    self.data_lst.append(tmp_dict)

                # mCSM
                fold_index = 0
                if self.mCSM_val is not None:
                    # val_data is all the same for each fold
                    x_val, y_val, ddg_val = self.get_val(self.mCSM_val, random_seed, val_num)
                    x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                for train_index, test_index in skf.split(self.mCSM.x, self.mCSM.y):
                    x_y_ddg_train = self.x_y_ddg(self.mCSM.x[train_index], self.mCSM.y[train_index], self.mCSM.ddg[train_index])
                    x_y_ddg_test = self.x_y_ddg(self.mCSM.x[test_index], self.mCSM.y[test_index], self.mCSM.ddg[test_index])
                    if self.mCSM_val is None and self.val:
                        x_train, y_train, ddg_train, x_val, y_val, ddg_val = self.split_val(x_y_ddg_train, x_y_ddg_test.ddg, random_seed)
                        x_y_ddg_train = self.x_y_ddg(x_train, y_train, ddg_train)
                        x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                    elif self.mCSM_val is None and self.val is False:
                        x_y_ddg_val = None

                    # Nomalization
                    if x_y_ddg_val is None:
                        x_train, x_test= self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=None, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                    else:
                        x_train, x_test, x_val = self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=x_y_ddg_val.x, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                        x_y_ddg_val.x = x_val

                    self.data_lst[fold_index]['mCSM'] = self.train_test_val(x_y_ddg_train,x_y_ddg_test,x_y_ddg_val)
                    fold_index += 1

            elif self.mCNN is not None and self.mCSM is None:
                # mCNN
                if self.mCNN_val is not None:
                    # val_data is all the same for each fold
                    x_val, y_val, ddg_val = self.get_val(self.mCNN_val, random_seed, val_num)
                    x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                for train_index, test_index in skf.split(self.mCNN.x, self.mCNN.y):
                    tmp_dict = {}
                    x_y_ddg_train = self.x_y_ddg(self.mCNN.x[train_index], self.mCNN.y[train_index], self.mCNN.ddg[train_index])
                    x_y_ddg_test = self.x_y_ddg(self.mCNN.x[test_index], self.mCNN.y[test_index], self.mCNN.ddg[test_index])
                    if self.mCNN_val is None and self.val:
                        x_train, y_train, ddg_train, x_val, y_val, ddg_val = self.split_val(x_y_ddg_train, x_y_ddg_test.ddg, random_seed)
                        x_y_ddg_train = self.x_y_ddg(x_train, y_train, ddg_train)
                        x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                    elif self.mCNN_val is None and self.val is False:
                        x_y_ddg_val = None
                    # Nomalization
                    if x_y_ddg_val is None:
                        x_train, x_test= self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=None, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                    else:
                        x_train, x_test, x_val = self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=x_y_ddg_val.x, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                        x_y_ddg_val.x = x_val

                    tmp_dict['mCNN'] = self.train_test_val(x_y_ddg_train, x_y_ddg_test, x_y_ddg_val)
                    self.data_lst.append(tmp_dict)

            elif self.mCSM is not None and self.mCNN is None:
                # mCSM
                if self.mCSM_val is not None:
                    # val_data is all the same for each fold
                    x_val, y_val, ddg_val = self.get_val(self.mCSM_val, random_seed, val_num)
                    x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                for train_index, test_index in skf.split(self.mCSM.x, self.mCSM.y):
                    tmp_dict = {}
                    x_y_ddg_train = self.x_y_ddg(self.mCSM.x[train_index], self.mCSM.y[train_index], self.mCSM.ddg[train_index])
                    x_y_ddg_test = self.x_y_ddg(self.mCSM.x[test_index], self.mCSM.y[test_index], self.mCSM.ddg[test_index])
                    if self.mCSM_val is None and self.val:
                        x_train, y_train, ddg_train, x_val, y_val, ddg_val = self.split_val(x_y_ddg_train, x_y_ddg_test.ddg,random_seed)
                        x_y_ddg_train = self.x_y_ddg(x_train, y_train, ddg_train)
                        x_y_ddg_val = self.x_y_ddg(x_val, y_val, ddg_val)
                    elif self.mCSM_val is None and self.val is False:
                        x_y_ddg_val = None
                    # Nomalization
                    if x_y_ddg_val is None:
                        x_train, x_test= self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=None, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                    else:
                        x_train, x_test, x_val = self.normalize(x_y_ddg_train.x, x_y_ddg_test.x, x_val=x_y_ddg_val.x, method=self.norm_method)
                        x_y_ddg_train.x = x_train
                        x_y_ddg_test.x = x_test
                        x_y_ddg_val.x = x_val

                    tmp_dict['mCSM'] = self.train_test_val(x_y_ddg_train, x_y_ddg_test, x_y_ddg_val)
                    self.data_lst.append(tmp_dict)

        elif fold_num == 2:
            pass
            # for label in set(y.reshape(-1)):
            #     index = np.argwhere(y.reshape(-1) == label)
            #     train_num = int(index.shape[0] * train_ratio)
            #     train_index = index[:train_num]
            #     test_index  = index[train_num:]
            #     x_train.append(x[train_index])
            #     y_train.append(y[train_index])
            #     ddg_train.append(ddg[train_index])
            #     x_test.append(x[test_index])
            #     y_test.append(y[test_index])
            #     ddg_test.append(ddg[test_index])
            #     reshape_lst = list(x.shape[1:])
            #     reshape_lst.insert(0,-1)
            #     ## transform python list to numpy array
            #     x_train   = np.array(x_train).reshape(reshape_lst)
            #     y_train   = np.array(y_train).reshape(-1,1)
            #     ddg_train = np.array(ddg_train).reshape(-1,1)
            #     x_test    = np.array(x_test).reshape(reshape_lst)
            #     y_test    = np.array(y_test).reshape(-1, 1)
            #     ddg_test  = np.array(ddg_test).reshape(-1, 1)
            #     ## shuffle data
            #     x_train, y_train, ddg_train = self.shuffle_data(x_train, y_train, ddg_train, random_seed)
            #     x_test, y_test, ddg_test    = self.shuffle_data(x_test, y_test, ddg_test, random_seed)

        else:
            print('[ERROR] The fold number should not smaller than 2!')
            exit(0)

    def normalize(x_train, x_test, x_val = None, method='norm'):
        train_shape, test_shape = x_train.shape, x_test.shape
        col_train = x_train.shape[-1]
        col_test = x_test.shape[-1]
        x_train = x_train.reshape((-1, col_train))
        x_test = x_test.reshape((-1, col_test))

        if x_val is not None:
            val_shape = x_val.shape
            col_val = x_val.shape[-1]
            x_val = x_val.reshape((-1, col_val))

        if method == 'norm':
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[np.argwhere(std == 0)] = 0.01
            x_train -= mean
            x_train /= std
            x_test -= mean
            x_test /= std
            if x_val is not None:
                x_val -= mean
                x_val /= std
        elif method == 'max':
            max_ = x_train.max(axis=0)
            max_[np.argwhere(max_ == 0)] = 0.01
            x_train /= max_
            x_test /= max_
            if x_val is not None:
                x_val /= max_

        x_train = x_train.reshape(train_shape)
        x_test = x_test.reshape(test_shape)

        if x_val is not None:
            x_val = x_val.reshape(val_shape)
            return x_train, x_test, x_val
        else:
            return x_train, x_test


    def shuffle_data(self,x, y, ddg, random_seed):
        indices = [i for i in range(x.shape[0])]
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        ddg = ddg[indices]
        return x, y, ddg

    def split_val(self, train_x_y_ddg_obj, ddg_test, seed):
        '''split val_data from the same dataset'''
        x_train, y_train, ddg_train = train_x_y_ddg_obj.x, train_x_y_ddg_obj.y, train_x_y_ddg_obj.ddg
        ddg_train, ddg_test = ddg_train.reshape(-1), ddg_test.reshape(-1)
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
        x_train_new, y_train_new, ddg_train_new = np.vstack((x_p_train,x_n_train)), np.vstack((y_p_train,y_n_train)), np.hstack((ddg_p_train,ddg_n_train))
        ## shuffe data.
        x_train_new, y_train_new, ddg_train_new = self.shuffle_data(x_train_new, y_train_new, ddg_train_new, random_seed=seed)

        assert x_train_new.shape[0] + x_val.shape[0] == x_train.shape[0]
        assert x_val.shape[0] == ddg_test.shape[0]

        return x_train_new, y_train_new, ddg_train_new, x_val, y_val, ddg_val

    def get_val(self, x_y_ddg_obj, seed, val_num):
        '''get val_data from another dataset'''
        indices = [i for i in range(x_y_ddg_obj.x.size[0])]
        np.random.seed(seed)
        np.random.shuffle(indices)
        val_indices = indices[:val_num]
        x_val = x_y_ddg_obj.x[val_indices]
        y_val = x_y_ddg_obj.y[val_indices]
        ddg_val = x_y_ddg_obj.ddg[val_indices]
        return x_val, y_val, ddg_val

if __name__ == '__main__':
    homedir = '/public/home/sry'
    dataset_name = 'S2648'
    val_dataset_name = 'S1925'
    center = 'CA'
    str_pca = 'False'
    str_k_neighbor = '50'
    container = {}

    container['mCNN_wild_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' % (homedir, dataset_name, center, str_pca, str_k_neighbor)
    container['mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' % (homedir, dataset_name, center, str_pca, str_k_neighbor)
    container['val_mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
    container['val_mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)

    container['mCSM_wild_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' % (homedir, dataset_name, center, str_pca, str_k_neighbor)
    container['mCSM_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' % (homedir, dataset_name, center, str_pca, str_k_neighbor)
    container['val_mCSM_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
    container['val_mCSM_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)


    sort_method = 'chain'
    permuation_seed = 1

    fold_num = 5
    random_seed = 1

    DE = DataExtractor(container,sort_method,permuation_seed)
    DE.split_kfold(fold_num,random_seed)
    print(DE.data_lst)