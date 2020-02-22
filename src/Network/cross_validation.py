#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
from keras.models import load_model
from mCNN.Network.BuildNet import Conv2DMultiTaskIn1
from mCNN.Network.Data import DataExtractor
from mCNN.processing import shell, str2bool


def main():
    PP = ParamsParser()
    data_lst = getData(PP_obj=PP)
    cross_validation(PP_obj=PP, data_lst=data_lst)


def getData(PP_obj):
    DE = DataExtractor(container=PP_obj.container, sort_method = PP_obj.sort_method, permutation_seed=PP_obj.seed_tuple[0], normalize_method = PP_obj.normalize_method, val=PP_obj.split_val)
    DE.split_kfold(fold_num=PP_obj.k, k_fold_seed = PP_obj.seed_tuple[1], val_seed = PP_obj.seed_tuple[2], train_ratio = 0.7)
    return DE.data_lst


def cross_validation(PP_obj,data_lst):
    print('\n----------Cross validation, fold number is %s'%PP_obj.k)
    fold_flag = 0
    for data_dict in data_lst:
        fold_flag += 1
        print('\n----------Fold %s is running'%fold_flag)
        if PP_obj.nn_model == 'Conv2DMultiTaskIn1':
            network = Conv2DMultiTaskIn1(CUDA=PP_obj.CUDA, data_dict=data_dict, summary=PP_obj.summary, batch_size=PP_obj.batch_size, epoch=PP_obj.epoch, verbose=PP_obj.verbose)
        elif PP_obj.nn_model == 'Conv2DClassifierIn1':
            network = Conv2DClassifierIn1(CUDA=PP_obj.CUDA, data_dict=data_dict, summary=PP_obj.summary, batch_size=PP_obj.batch_size, epoch=PP_obj.epoch, verbose=PP_obj.verbose)
        elif PP_obj.nn_model == 'Conv2DRegressorIn1':
            network = Conv2DRegressorIn1(CUDA=PP_obj.CUDA, data_dict=data_dict, summary=PP_obj.summary, batch_size=PP_obj.batch_size, epoch=PP_obj.epoch, verbose=PP_obj.verbose)
        network.train()
        network.evaluate()


class ParamsParser(object):
    def __init__(self):
        self.parse_params()
        self.CHK()

    def parse_params(self):
        # Init
        homedir = shell('echo $HOME')
        self.container = {'mCNN_wild_dir': '', 'mCNN_mutant_dir': '', 'val_mCNN_wild_dir': '', 'val_mCNN_mutant_dir': '',
                          'mCSM_wild_dir': '', 'mCSM_mutant_dir': '', 'val_mCSM_wild_dir': '', 'val_mCSM_mutant_dir': '', }
        # Input parameters----------------------------------------------------------------------------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset_name',        type=str,   help='dataset_name.')
        parser.add_argument('wild_or_mutant',      type=str,   default='wild',  choices=['wild','mutant','stack'],     help='wild, mutant or stack array, default is wild.')
        parser.add_argument('--val_dataset_name',  type=str,   help='validation dataset_name.')
        parser.add_argument('-C', '--center',      type=str,   default='CA',    choices=['CA', 'geometric'],   help='The MT site center, default is CA.')
        parser.add_argument('--split_val',         type=str,   default='True', help='whether split val_data from train_data when val_data is not given,default is True')
        ## parameters for reading mCNN array, array locates at: '~/mCNN/dataset/S2648/feature/mCNN/mutant/npz/center_CA_PCA_False_neighbor_50.npz'
        parser.add_argument('--mCNN',              type=str,   nargs=2,         help='PCA, k_neighbor')
        ## parameters for reading mCSM array, array locates at: '~/mCNN/dataset/S2648/feature/mCSM/mutant/npz/min_0.1_max_7.0_step_2.0_center_CA_class_2.npz'
        parser.add_argument('--append',            type=str,   default='False', help='whether append mCSM features to mCNN features, default is False.')
        parser.add_argument('--mCSM',              type=str,   nargs=4,         help='min, max, step, atom_class_num.')
        ## Config data processing params
        parser.add_argument('-n', '--normalize',   type=str,   choices=['norm', 'max'], default='norm', help='normalize_method to choose, default = norm.')
        parser.add_argument('-s', '--sort',        type=str,
                            choices=['chain', 'distance', 'octant', 'permutation', 'permutation1', 'permutation2'],
                            default='chain',       help='row sorting methods to choose, default = "chain".')
        parser.add_argument('-d', '--random_seed', type=int, nargs=3, default=(1,1,1), help='permutation-seed, k-fold-seed, split-val-seed, default sets to (1,1,1).')

        ## Config training
        parser.add_argument('-K', '--Kfold',       type=int, help='Fold numbers to cross validation.')
        parser.add_argument('-D', '--nn_model',    type=str, help='Network model to chose.', required=True)
        parser.add_argument('-S', '--summary',     type=str, help='Print model.summary()', default=False)
        parser.add_argument('-V', '--verbose',     type=int, choices=[0, 1], default=1, help='the verbose flag, default is 1.')
        parser.add_argument('-E', '--epoch',       type=int, default=100, help='training epoch, default is 100.')
        parser.add_argument('-B', '--batch_size',  type=int, default=128,  help='training batch size, default is 128.')
        ## config hardware
        parser.add_argument('--CUDA', type=str, default='0', choices=['0', '1', '2', '3'], help='Which gpu to use, default = "0"')
        # parser--------------------------------------------------------------------------------------------------------
        args = parser.parse_args()
        self.dataset_name   = args.dataset_name
        self.wild_or_mutant = args.wild_or_mutant
        self.center = args.center
        self.split_val = str2bool(args.split_val)
        if args.mCNN:
            self.str_pca, self.str_k_neighbor = args.mCNN
            if self.wild_or_mutant == 'stack':
                self.container['mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                    %(homedir,self.dataset_name,self.center,self.str_pca,self.str_k_neighbor)
                self.container['mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                    %(homedir,self.dataset_name,self.center,self.str_pca,self.str_k_neighbor)
                if args.val_dataset_name:
                    self.val_dataset_name = args.val_dataset_name
                    self.container['val_mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                            %(homedir,self.val_dataset_name,self.center,self.str_pca,self.str_k_neighbor)
                    self.container['val_mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                            %(homedir,self.val_dataset_name,self.center,self.str_pca,self.str_k_neighbor)
            else:
                self.container['mCNN_%s_dir'%self.wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                                    %(homedir,self.dataset_name,self.wild_or_mutant,self.center,self.str_pca,self.str_k_neighbor)
                if args.val_dataset_name:
                    self.val_dataset_name = args.val_dataset_name
                    self.container['val_mCNN_%s_dir'%self.wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz'\
                                                                            %(homedir,self.val_dataset_name,self.wild_or_mutant,self.center,self.str_pca,self.str_k_neighbor)
        if args.mCSM:
            self.min_, self.max_, self.step, self.atom_class_num = args.mCSM
            if self.wild_or_mutant == 'stack':
                self.container['mCSM_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                    %(homedir, self.dataset_name, self.min_, self.max_, self.step, self.center, self.atom_class_num)
                self.container['mCSM_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                    %(homedir, self.dataset_name, self.min_, self.max_, self.step, self.center, self.atom_class_num)
                if args.val_dataset_name:
                    self.val_dataset_name = args.val_dataset_name
                    self.container['val_mCSM_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                            %(homedir, self.val_dataset_name, self.min_, self.max_, self.step, self.center, self.atom_class_num)
                    self.container['val_mCSM_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                            %(homedir, self.val_dataset_name, self.min_, self.max_, self.step, self.center, self.atom_class_num)
            else:
                self.container['mCSM_%s_dir'%self.wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                                    %(homedir, self.dataset_name, self.wild_or_mutant, self.min_, self.max_, self.step, self.center, self.atom_class_num)
                if args.val_dataset_name:
                    self.val_dataset_name = args.val_dataset_name
                    self.container['val_mCSM_%s_dir'%self.wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                                                            %(homedir, self.val_dataset_name, self.wild_or_mutant, self.min_, self.max_, self.step, self.center, self.atom_class_num)
        self.append = args.append
        ## parser for data processing
        self.normalize_method = args.normalize
        self.sort_method = args.sort
        self.seed_tuple = tuple(args.random_seed)
        ## parser for training
        self.k = args.Kfold
        self.nn_model = args.nn_model
        self.summary = str2bool(args.summary)
        self.verbose = args.verbose
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.CUDA = args.CUDA

    def CHK(self):
        # print input info. --------------------------------------------------------------------------------------------
        print('\n----------Input params.'
              '\ncontainer: %r,'
              '\nkfold: %s, append: %s, split_val: %s,'
              '\nnormalize_method: %s, sort_method: %s, (permutation-seed, k-fold-seed, split-val-seed): %r,'
              '\nnn_model: %s, summary: %s, verbose_flag: %s, epoch: %s, batch_size: %s, CUDA: %r.'
              % (self.container, self.k, self.append, self.split_val, self.normalize_method, self.sort_method, self.seed_tuple,
                 self.nn_model, self.summary, self.verbose, self.epoch, self.batch_size, self.CUDA))


class NetworkEvaluator(object):
    ''''''
    def __init__(self, test_data, model=None, model_dir=None):
        '''use the model from param or load model from hard drive.'''
        self.model = model
        self.model_dir = model_dir
        self.test_data = test_data

    def load_model(self):
        if self.model is None:
            assert os.path.exists(self.model_dir)
            self.model = load_model(self.model_dir)

    def test_model(self):
        '''evaluate model with test_data which have sort row and normalization done)'''
        pass

    def predict(self):
        '''predict labels of brand new data'''
        pass


if __name__ == '__main__':
    main()