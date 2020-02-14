#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file_name : Data.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com

class DataPacker(object):
    def x_y_ddg(x,y,ddg):
        self.x = x
        self.y = y
        self.ddg = ddg
    def train_test_val(train_data,test_data,val_data):
        self.train = train_data
        self.test  = test_data
        self.val   = val_data

class DataLoader(object):
    def __init__(self,container):
        # mCNN
        self.mCNN_wild   = None
        self.mCNN_mutant   = None
        self.val_mCNN_wild   = None
        self.val_mCNN_mutant = None
        # mCSM
        self.mCSM_wild   = None
        self.mCSM_mutant   = None
        self.val_mCSM_wild   = None
        self.val_mCSM_mutant = None

        self.load_data(container)

    def load_data(self,container):
        # mCNN
        if container['mCNN_wild_dir'] != '':
            self.mCNN_wild = np.load(container['mCNN_wild_dir'])

        if container['mCNN_mutant_dir'] != '':
            self.mCNN_mutant = np.load(container['mCNN_mutant_dir'])

        if container['val_mCNN_wild_dir'] != '':
            self.val_mCNN_wild = np.load(container['val_mCNN_wild_dir'])

        if container['val_mCNN_mutant_dir'] != '':
            self.val_mCNN_mutant = np.load(container['val_mCNN_mutant_dir'])
        # mCSM
        if container['mCSM_wild_dir'] != '':
            mCSM_wild = np.load(container['mCSM_wild_dir'])

        if container['mCSM_mutant_dir'] != '':
            mCSM_mutant = np.load(container['mCSM_mutant_dir'])

        if container['val_mCSM_wild_dir'] != '':
            val_mCSM_wild = np.load(container['val_mCSM_wild_dir'])

        if container['val_mCSM_mutant_dir'] != '':
            val_mCSM_mutant = np.load(container['val_mCSM_wild_dir'])


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
                * data_object.train
                * data_object.test
                * data_object.val           
                    * data_object.val.x
                    * data_object.val.y
                    * data_object.val.ddg
    
    ++ NOTICE: validation data are THE SAME for each fold. ++
    '''

    def __init__(self,container, val=True):
        super().__init__(container)
        self.data_lst = []
        self.

    def given_kfold(self, data_lst):
        self.data_lst = data_lst

    def processing(self):
        # only cosider mCNN features
        if self.mCSM_wild is None and self.mCSM_mutant is None:
            assert self.mCNN_wild is not None or self.mCNN_mutant is not None
            # only cosider mCNN_wild features
            if self.mCNN_wild is not None and self.mCNN_mutant is None:
                self.mCNN = self.mCNN_wild
                self.val_mCNN = self.val_mCNN_wild
            # only cosider mCNN_mutant features
            elif self.mCNN_wild is None and self.mCNN_mutant is not None:
                self.mCNN = self.mCNN_mutant
                self.val_mCNN = self.val_mCNN_mutant
            # cosider mCNN_wild and mCNN_mutant features
            elif self.mCNN_wild is not None and self.mCNN_mutant is not None:
                self.mCNN = np.vstack((self.mCNN_wild,self.mCNN_mutant))
                try:
                    self.val_mCNN = np.vstack((self.mCNN_wild,self.mCNN_mutant))
            else:
                print('[ERROR] All features were failed to load (only cosider mCNN features)')
                exit(0)
    
        # only cosider mCSM features
        elif self.mCNN_wild is None and self.mCNN_mutant is None:
            assert self.mCSM_wild is not None or self.mCSM_mutant is not None
            # only cosider mCSM_wild features
            if self.mCSM_wild is not None and self.mCSM_mutant is None:
                pass
            # only cosider mCSM_wild features
            elif self.mCSM_wild is None and self.mCSM_mutant is not None:
                pass
            # cosider mCSM_wild and mCSM_mutant features
            elif self.mCSM_wild is not None and self.mCSM_mutant is not None:
                pass

            else:
                print('[ERROR] All features were failed to load (only cosider mCSM features)')
                exit(0)

        # consider mCNN_wild and mCSM features
        else:
            if not (self.mCNN_wild is None or self.mCNN_mutant is None or self.mCSM_wild is None or self.mCSM_mutant is None):
                pass
            # consider mCNN_wild and mCSM_wild features
            elif self.mCNN_wild is not None and self.mCSM_wild is not None:
                pass
            # consider mCNN_mutant and mCSM_mutant features
            elif self.mCNN_mutant is not None and self.mCSM_mutant is not None:
                pass
            # consider mCNN_wild and mCSM_mutant features
            elif self.mCNN_wild is not None and self.mCSM_mutant is not None:
                pass
            elif self.mCNN_mutant is not None and self.mCSM_wild is not None:
                pass
            else:
                print('[WARNING] Bad mixed features.')

    def split_kfold(self, fold_num, random_seed=10, train_ratio = 0.7):
        if self.
        if fold_num >= 3:
            skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=random_seed)
            for key in x_dict.keys():
                self.x_train_dict[key] = []
                self.y_traiin_dict[key] = []
                self.ddg_train_dict[key] = []
                self.x_test_dict[key] = []
                self.y_test_dict[key] = []
                self.ddg_test_dict[key] = []
                x,y,ddg = x_dict[key],y_dict[key],ddg_dict[key]
                for train_index, test_index in skf.split(x, y):
                    self.x_train_dict[key].append(x[train_index])
                    self.y_train_dict[key].append(y[train_index])
                    self.ddg_train_dict[key].append(ddg[train_index])
                    self.x_test_dict[key].append(x[test_index])
                    self.y_test_dict[key].append(y[test_index])
                    self.ddg_test_dict[key].append(ddg[test_index])
        elif fold_num == 2:
            for key in x_dict.keys():
                self.x_train_dict[key] = []
                self.y_traiin_dict[key] = []
                self.ddg_train_dict[key] = []
                self.x_test_dict[key] = []
                self.y_test_dict[key] = []
                self.ddg_test_dict[key] = []
                x,y,ddg = x_dict[key],y_dict[key],ddg_dict[key]
                x_train   = []
                y_train   = []
                ddg_train = []
                x_test    = []
                y_test    = []
                ddg_test  = []
                for label in set(y.reshape(-1)):
                    index = np.argwhere(y.reshape(-1) == label)
                    train_num = int(index.shape[0] * train_ratio)
                    train_index = index[:train_num]
                    test_index  = index[train_num:]
                    x_train.append(x[train_index])
                    y_train.append(y[train_index])
                    ddg_train.append(ddg[train_index])

                    x_test.append(x[test_index])
                    y_test.append(y[test_index])
                    ddg_test.append(ddg[test_index])
                reshape_lst = list(x.shape[1:])
                reshape_lst.insert(0,-1)
                ## transform python list to numpy array
                x_train   = np.array(x_train).reshape(reshape_lst)
                y_train   = np.array(y_train).reshape(-1,1)
                ddg_train = np.array(ddg_train).reshape(-1,1)
                x_test    = np.array(x_test).reshape(reshape_lst)
                y_test    = np.array(y_test).reshape(-1, 1)
                ddg_test  = np.array(ddg_test).reshape(-1, 1)
                ## shuffle data
                x_train, y_train, ddg_train = shuffle_data(x_train, y_train, ddg_train, random_seed)
                x_test, y_test, ddg_test    = shuffle_data(x_test, y_test, ddg_test, random_seed)

            self.x_train_dict[key].append(x_train)
            self.y_train_dict[key].append(y_train)
            self.ddg_train_dict[key].append(ddg_train)
            self.x_test_dict[key].append(x_test)
            self.y_test_dict[key].append(y_test)
            self.ddg_test_dict[key].append(ddg_test)
        else:
            print('[ERROR] The fold number should not smaller than 2!')
            exit(0)

    def given_val_data(self, x_val_dict, y_val_dict, ddg_val_dict):
        self.x_val_dict   = x_val_dict
        self.y_val_dict   = y_val_dict
        self.ddg_val_dict = ddg_val_dict
    
    def split_val_data(self):
        '''split val_data from the same dataset'''
        pass

    def get_val_data(self, seed, val_num):
        '''get val_data from another dataset'''
        key_lst = list(x_val_dict.keys())
        indices = [i for i in range(x_val_dict[key_lst[0]].size[0])]
        np.random.seed(seed)
        np.random.shuffle(indices)
        val_indices = indices[:val_num]
        for key in key_lst:
            x_val_dict[key] = x_val_dict[key][val_indices]
            y_val_dict[key] = y_val_dict[key][val_indices]
            ddg_val_dict[key] = ddg_val_dict[key][val_indices]
        self.x_val_dict   = x_val_dict
        self.y_val_dict   = y_val_dict
        self.ddg_val_dict = ddg_val_dict

    

if __name__ == '__main__':
    fold_num = 5


    if fold_num == 1:
        self.given_kfold()

    if fold_num == 2:
        self.split_kfold

    if fold_num >= 3:
        pass        