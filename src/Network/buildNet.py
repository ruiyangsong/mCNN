#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : build_model.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import os, sys, argparse
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from mCNN.processing import shuffle_data, load_sort_data, shell, append_mCSM
from keras.backend.tensorflow_backend import set_session

from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from keras import Input
from keras.utils import plot_model


import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense , Dropout , Activation , Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution1D , MaxPooling1D , AveragePooling1D
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    # get command line params ------------------------------------------------------------------------------------------
    homedir = shell('echo $HOME')
    ## Init container
    container = {'mCNN_arrdir': '', 'mCSM_arrdir': '', 'val_mCNN_arrdir': '', 'val_mCSM_arrdir': ''}
    ## Input parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name',        type=str,   help='dataset_name.')
    parser.add_argument('val_dataset_name',    type=str,   help='validation dataset_name.')
    parser.add_argument('wild_or_mutant',      type=str,   default='wild',  choices=['wild','mutant','stack','split'],     help='wild, mutant, stack or split array, default is wild.')
    parser.add_argument('-C', '--center',      type=str,   default='CA',    choices=['CA', 'geometric'],   help='The MT site center, default is CA.')
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
    parser.add_argument('-D', '--model',       type=str, help='Network model to chose.', required=True)
    parser.add_argument('-K', '--Kfold',       type=int, help='Fold numbers to cross validation.')
    parser.add_argument('-V', '--verbose',     type=int, choices=[0, 1], default=1, help='the verbose flag, default is 1.')
    parser.add_argument('-E', '--epoch',       type=int, default=100, help='training epoch, default is 100.')
    parser.add_argument('-B', '--batch_size',  type=int, default=64,  help='training batch size, default is 64.')
    ## config hardware
    parser.add_argument('--CUDA', type=str, default='0', choices=['0', '1', '2', '3'], help='Which gpu to use, default = "0"')
    ## parser
    args = parser.parse_args()
    dataset_name   = args.dataset_name
    val_dataset_name   = args.val_dataset_name
    wild_or_mutant = args.wild_or_mutant
    center = args.center
    if args.mCNN:
        str_pca, str_k_neighbor = args.mCNN
        container['mCNN_arrdir'] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz'%(homedir,dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
        container['val_mCNN_arrdir'] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz'%(homedir,val_dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
    if args.mCSM:
        min_, max_, step, atom_class_num = args.mCSM
        container['mCSM_arrdir'] = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'%(homedir, dataset_name, wild_or_mutant, min_, max_, step, center, atom_class_num)
        container['val_mCSM_arrdir'] = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz'%(homedir, val_dataset_name, wild_or_mutant, min_, max_, step, center, atom_class_num)
    if container['mCNN_arrdir'] == '' and container['mCSM_arrdir'] == '':
        print('[ERROR] parsing feature_type param error, check the argparser code!')
        exit(0)
    append = args.append
    ## parser for data processing
    normalize_method = args.normalize
    sort_method = args.sort
    seed_tuple = tuple(args.random_seed)
    ## parser for training
    nn_model = args.model
    k = args.Kfold
    verbose = args.verbose
    epoch = args.epoch
    batch_size = args.batch_size
    CUDA = args.CUDA
    ## print input info.
    print('dataset_name: %s, val_dateset_name: %s, wild_or_mutant: %s, center: %s,'
          '\nmCNN_arrdir: %s,'
          '\nval_mCNN_arrdir: %s,'
          '\nmCSM_arrdir: %s,'
          '\nval_mCSM_arrdir: %s,'
          '\nappend: %s,'
          '\nnormalize_method: %s,'
          '\nsort_method: %s,'
          '\n(permutation-seed, k-fold-seed, split-val-seed): %r,'
          '\nmodel: %s,'
          '\nkfold: %s,'
          '\nverbose_flag: %s,'
          '\nepoch: %s,'
          '\nbatch_size: %s,'
          '\nCUDA: %r.'
          % (dataset_name, val_dataset_name, wild_or_mutant, center, container['mCNN_arrdir'], container['val_mCNN_arrdir'],container['mCSM_arrdir'],container['val_mCSM_arrdir'],append,
             normalize_method,sort_method,seed_tuple,nn_model,k,verbose,epoch,batch_size,CUDA))

    # load data and sort row (return python dictionary)-----------------------------------------------------------------
    if container['mCNN_arrdir'] != '':
        x_mCNN, y_mCNN, ddg_mCNN = load_sort_data(container['mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_mCNN_val, y_mCNN_val, ddg_mCNN_val = load_sort_data(container['val_mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
    if container['mCSM_arrdir'] != '':
        x_mCSM, y_mCSM, ddg_mCSM = load_sort_data(container['mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_mCSM_val, y_mCSM_val, ddg_mCSM_val = load_sort_data(container['val_mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
    if container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] != '' and append == 'True':
        x_append = append_mCSM(x_mCNN_dict=x_mCNN, x_mCSM_dict=x_mCSM)
        x_append_val = append_mCSM(x_mCNN_dict=x_mCNN_val, x_mCSM_dict=x_mCSM_val)
        del x_mCNN, x_mCSM, x_mCNN_val, x_mCSM_val

    # Cross validation. ------------------------------------------------------------------------------------------------
    print('%d-fold cross validation begin.' % (k))
    DE = DataExtractor
    DE.split_kfold(x_dict, y_dict, ddg_dict, fold_num=k, random_seed=seed_tuple[1], train_ratio = 0.7)
    key_lst = list(DE.x_test_dict.keys())
    val_num = int(DE.x_test_dict[key_lst[0]][0].shape[0])
    DE.split_val_data(x_val_dict, y_val_dict, ddg_val_dict, seed_tuple[2], val_num)



def cross_validation():
    pass


    print_result(nn_model, kfold_score)
    ## plot.
    # plotfigure(history_dict)

def save_model(model, model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        model.save('%s/%s'%(model_path, model_name))
        print('---model saved at %s.'%model_path)

class DataExtractor(object):
    '''每一个数据是一个python字典，字典中每一个值是一个列表，列表存储了每一折数据，eg.{'wild':[fold1,fold2,...],...}'''
    def __init__(self):
        self.x_val_dict   = None
        self.y_val_dict   = None
        self.ddg_val_dict = None

    def split_val_data(self, x_val_dict, y_val_dict, ddg_val_dict,val_num):
        key_lst = list(x_val_dict.keys())
        indices = [i for i in range(x_val_dict[key_lst[0]].size[0])]
        np.random.seed(p_seed)
        np.random.shuffle(indices)
        val_indices = indices[:val_num]
        for key in key_lst:
            x_val_dict[key] = x_val_dict[key][val_indices]
            y_val_dict[key] = y_val_dict[key][val_indices]
            ddg_val_dict[key] = ddg_val_dict[key][val_indices]
        self.x_val_dict   = x_val_dict
        self.y_val_dict   = y_val_dict
        self.ddg_val_dict = ddg_val_dict

    def given_val_data(self, x_val_dict, y_val_dict, ddg_val_dict):
        self.x_val_dict   = x_val_dict
        self.y_val_dict   = y_val_dict
        self.ddg_val_dict = ddg_val_dict

    def given_kfold(self, x_train_dict, y_train_dict, ddg_train_dict, x_test_dict, y_test_dict, ddg_test_dict):
        self.x_train_dict   = x_train_dict
        self.y_train_dict   = y_train_dict
        self.ddg_train_dict = ddg_train_dict

        self.x_test_dict    = x_test_dict
        self.y_test_dict    = y_test_dict
        self.ddg_test_dict  = ddg_test_dict

    def split_kfold(self, x_dict, y_dict, ddg_dict, fold_num, random_seed=10, train_ratio = 0.7):
        '''参数中的dict的值所存储的数据没有分折（split fold）'''
        self.x_train_dict   = {}
        self.y_traiin_dict  = {}
        self.ddg_train_dict = {}

        self.x_test_dict    = {}
        self.y_test_dict    = {}
        self.ddg_test_dict  = {}
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

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.pearson_r = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_pearson_r = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.pearson_r['batch'].append(logs.get('pearson_r'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_pearson_r['batch'].append(logs.get('val_pearson_r'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.pearson_r['epoch'].append(logs.get('pearson_r'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_pearson_r['epoch'].append(logs.get('val_pearson_r'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.pearson_r[loss_type], 'r', label='train_pearson_r')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train_loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_pearson_r[loss_type], 'b', label='val_pearson_r')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('pearson_r-loss')
        plt.legend(loc="upper right")
        plt.savefig("/public/home/yels/project/Data_pre/QA_global.png")
        # plt.show()


class NetworkTrainer(object):
    def __init__(self, DE_object=None, nn_model=None, verbose=1, CUDA='0', epoch=100, batch_size=128):

        self.DE_obiect  = DE_object
        self.nn_model   = nn_model
        self.verbose    = verbose
        self.CUDA       = CUDA
        self.epoch      = epoch
        self.batch_size = batch_size

        self.tag        = None
        self.model_path = model_path
    
    def run(self):
        self.config_tf()
        self.load_data()
        self.train_model()

    def config_tf(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def load_data(self, data):
        self.x_train   = data.x_train
        self.y_train   = data.y_train
        self.ddg_train = data.ddg_train
        self.x_val     = data.x_val
        self.y_val     = data.y_val
        self.ddg_val   = data.ddg_val
        self.x_test    = data.x_test
        self.y_test    = data.y_test
        self.ddg_test  = data.ddg_test

    def train_model(self):
        def train_In1_Out2(self, _optimizer, class_loss, ddg_loss, class_weight, ddg_weight, epochs, batch_size, verbose = 1, validation = True):
        self.model.compile(optimizer=_optimizer,
                           loss={'class': class_loss,
                                 'ddg'  : ddg_loss},
                           loss_weights={'class': class_weight,
                                         'ddg'  : ddg_weight},
                           metrics={'class': ['accuracy',自定义的损失],
                                    'ddg'  : ['mse']})
        val_data = None
        if validation:
            val_data = (self.x_val, [self.y_val,self.ddg_val])

        
        # if validation:
        #     self.model.fit(x=self.x_train,
        #                    y={'class': self.y_train,
        #                       'ddg'  : self.ddg_train},
        #                    validation_data=(self.x_val, [self.y_val,self.ddg_val]),
        #                    epochs=epochs,
        #                    batch_size=batch_size,
        #                    verbose=verbose,
        #                    shuffle=True)
        # else:
        #     self.model.fit(x=self.x_train,
        #                    y={'class': self.y_train,
        #                       'ddg': self.ddg_train},
        #                    epochs=epochs,
        #                    batch_size=batch_size,
        #                    verbose=verbose,
        #                    shuffle=True)

    def save_model(self, metric, threshold, model_path):
        pass

class NetworkEvaluator(object):
    def __init__(self,model_dir,test_data,model):
        '''use the model from param or load model from hard drive.'''
        self.model_dir = model_dir
        self.test_data = test_data
        self.model = model

    def load_model(self):
        self.model = load_model(self.model_dir)

    def test_model(self):
        pass

class TrainCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.pearson_r = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_pearson_r = {'batch':[], 'epoch':[]}

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.pearson_r['epoch'].append(logs.get('pearson_r'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_pearson_r['epoch'].append(logs.get('val_pearson_r'))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.pearson_r['batch'].append(logs.get('pearson_r'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_pearson_r['batch'].append(logs.get('val_pearson_r'))


    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.pearson_r[loss_type], 'r', label='train_pearson_r')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train_loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_pearson_r[loss_type], 'b', label='val_pearson_r')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('pearson_r-loss')
        plt.legend(loc="upper right")
        plt.savefig("/public/home/yels/project/Data_pre/QA_global.png")
        # plt.show()


class Metrics_Generator(object):
    def pearson_r(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x, axis=0)
        my = K.mean(y, axis=0)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / r_den
        return K.mean(r)

    def recall_p():
        pass

    def recall_n():
        pass

    def mcc():
        pass

    def acc():
        pass

class LossFunction(object):
    # def binary_crossentropy_focal_loss(y_true, y_pred):
    #     alpha = 0.25; gamma = 2
    #     label  = y_true[0]
    #     output = K.clip(y_pred[0], K.epsilon(), 1 - K.epsilon())
    #     pt = label*output + (1-label)*(1-output)
    #     bc = - alpha * K.pow(1-pt, gamma) * K.log(pt)
    #     positive = K.sum(label)
    #     return K.sum(bc)/positive
    def binary_focal_loss(gamma=2, alpha=0.25):
        """
        Binary form of focal loss.
        适用于二分类问题的focal loss
        focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)
        def binary_focal_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
            p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
            return K.mean(focal_loss)
        return binary_focal_loss_fixed

class ConvNet(object):
    def __init__(self, input_num, output_num, input_shape, channel_last = True, kernel_size=(3,3),initializer='random_uniform',activator='relu',pool_size=(2,2),padding_style='same',regular_rate=[0.001,0.001],dropout_rate=0.3), optimizer='adam', summary=True:
        if channel_last:
            self.batch_norm_axis = -1
        else:
            self.batch_norm_axis = 1
        self.input_num   = input_num
        self.output_num  = output_num
        self.input_shape = input_shape[1:]# Attention HERE!
        
        self.kernel_size = kernel_size
        self.initializer = initializer
        self.activator   = activator
        
        self.pool_size   = pool_size
        self.padding_style = padding_style
        
        self.regular_rate  = regular_rate
        self.dropout_rate  = dropout_rate

        self.optimizer = optimizer
        self.summary   = summary

        self.model      = None


    def run(self,):
        self.build_model(self.task_type)


    def multi_task(self, class_num=2,loss_type_lst=['mse','binary_crossentropy'], loss_weights_lst=[0.5,10.], metrics_lst=['mse','accuracy']):
        if self.input_num == 1 and self.output == 2:
            input_layer = Input(shape=self.input_shape)
            conv1 = layers.Conv2D(16,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(input_layer)
            conv2 = layers.Conv2D(32,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(conv1)
            pool1 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv2)
            conv3 = layers.Conv2D(64,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv4)
            flat = layers.Flatten()(pool3)
            
            dense = layers.Dense(128, activation=self.activator)(flat)
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(self.dropout_rate)(dense_BatchNorm)
            ddg_prediction = layers.Dense(1, name='ddg')(drop)
            class_prediction = layers.Dense(class_num,activation='softmax',name='class')

            model = Model(inputs=input_layer, outputs=[ddg_prediction,class_prediction])
            
            if self.summary:
                model.summary()

            model.compile(optimizer=self.optimizer,
                          loss={'ddg':loss_type_lst[0],'class':loss_type_lst[1]},
                          loss_weights={'ddg':loss_weights_lst[0],'class':loss_weights_lst[1]},
                          metrics={'ddg':metrics_lst[0],'class':metrics_lst[1]}
                          )
            self.model = model
            self.model.fit(x=self.x_train,
                       y={'class': self.y_train,
                          'ddg': self.ddg_train},
                       validation_data=val_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose,
                       shuffle=True,
                       class_weight = [],
                       callbacks = [history])

        if self.input_num == 2 and self.output == 2:
            pass

    def classifiter(self, class_num=2, loss_type='binary_crossentropy',metrics=['accuracy']):
        if self.input_num == 1 and self.output == 1:
            input_layer = Input(shape=self.input_shape)
            conv1 = layers.Conv2D(16,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(input_layer)
            conv2 = layers.Conv2D(32,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(conv1)
            pool1 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv2)
            conv3 = layers.Conv2D(64,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv4)
            flat = layers.Flatten()(pool3)
            
            dense = layers.Dense(128, activation=self.activator)(flat)
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(self.dropout_rate)(dense_BatchNorm)

            output_layer = layers.Dense(class_num,activation='softmax')(drop)
            model = models.Model(inputs=input_layer, outputs=output_layer)

            if self.summary:
                model.summary()
            # rmsp = optimizers.RMSprop(lr=0.0008)
            model.compile(optimizer=self.optimizer,
                          loss=loss_type,
                          metrics=metrics) # accuracy
            self.model = model

        if self.input_num == 2 and self.output == 1:
            pass

    def regressor(self, loss_type='mse', metrics=['mae']):
        if self.input_num == 1 and self.output_num == 1:
            input_layer = Input(shape=self.input_shape)
            conv1 = layers.Conv2D(16,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(input_layer)
            conv2 = layers.Conv2D(32,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(conv1)
            pool1 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv2)
            conv3 = layers.Conv2D(64,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv4)
            flat = layers.Flatten()(pool3)
            
            dense = layers.Dense(128, activation=self.activator)(flat)
            
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(self.dropout_rate)(dense_BatchNorm)

            output_layer = layers.Dense(1)(drop)
            model = models.Model(inputs=input_layer, outputs=output_layer)

            if self.summary:
                model.summary()

            model.compile(optimizer=self.optimizer,
                          loss=loss_type,
                          metrics=metrics)
            self.model = model

        if self.input_num == 2 and self.output_num == 1:
            Input_1 = Input(shape=input_shape[0], name='input_1')
            conv1_input_1 = layers.Conv2D(16,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator)(input_layer)
            pool1_input_1 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv1_input_1)
            
            residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
            
            conv2_input_1 = layers.Conv2D(32,self.kernel_size,kernel_initializer=self.initializer,activation=self.activator,kernel_regularizer=regularizers.l1_l2(l1=self.regular_rate[0],l2=self.regular_rate[1]))(pool1_input_1)
            conv2_input_1_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv2_input_1)
            pool2_input_1 = layers.MaxPooling2D(self.pool_size,padding=self.padding_style)(conv2_input_1_BatchNorm)
            flat_input_1 = layers.Flatten()(pool2_input_1)
            dense_input_1 = layers.Dense(128, activation=self.activator)(flat_input_1)
            dense_input_1_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense_input_1)
            drop_input_1  = layers.Dropout(self.dropout_rate)(dense_input_1_BatchNorm)



            x = layers.Conv1D(128, 5, activation=activator)(Input_1)
            y = layers.MaxPooling1D(5)(x)

            y = layers.BatchNormalization(axis=self.batch_norm_axis)(y)
            residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
            y = layers.add([y, residual])

            Output_class = layers.Dense(class_num,
                                        activation=activator,
                                        name='class')(y)
            Output_ddg   = layers.Dense(1,
                                        name='ddg')(y)

            model = Model(Input_merge, [Output_class, Output_ddg])
            func_name  = sys._getframe().f_code.co_name
            self.tag   = func_name
            self.model = model
            if summary:
                self.model.summary()
  
    def train_parameter(self, data):
        pass

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(x=self.x_test,
                                             y={'class': self.y_test,
                                                'ddg'  : self.ddg_test})
        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def predict(self, data):
        result = self.model.predict_proba(data)  # 测算一下该img属于某个label的概率
        max_index = np.argmax(result)  # 找出概率最高的

        return max_index, result[0][max_index]  # 第一个参数为概率最高的label的index,第二个参数为对应概率

    def save(self, model_dir):
        self.model.save(model_dir)
        print('Model Saved.')

    def load(self, model_dir):
        self.model = load_model(model_dir)
        print('Model Loaded.')















def build_model(nn_model,sample_size):
    print('sample_size: ',sample_size)
    try:
        row_num,col_num = sample_size
    except:
        col_num = sample_size[0]
    ## 序贯模型使用network, API模型使用model.
    network = models.Sequential()

    if nn_model == 1.01:
        #print('SingleNet classification')
        network.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(row_num, col_num, 1)))
        # network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # network.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        network.add(layers.Conv2D(32, (3, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        network.add(layers.Conv2D(64, (3, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        network.add(layers.Conv2D(32, (5, 5), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        network.add(layers.Flatten())
        network.add(layers.Dense(32, activation='relu'))
        network.add(layers.Dropout(0.1))
        network.add(layers.Dense(2, activation='softmax'))
        # print(network.summary())
        adam = optimizers.adam(lr=1e-4,decay=1e-5)
        # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
        network.compile(optimizer=adam,  # SGD,adam,rmsprop
                        loss=[binary_focal_loss(alpha=.25, gamma=2)],#'binary_crossentropy',
                        metrics=['accuracy'])  # accuracy
        return network

    if nn_model == 1.02:
        #print('SingleNet-M classification')
        input1 = layers.Input(shape=(row_num,99,1),name='mCNN')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        drop  = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu')(drop)

        input2 = layers.Input(shape=(col_num-99,),name='mCSM')
        dense2 = layers.Dense(128, activation='relu')(input2)

        added = layers.concatenate([dense1, dense2],axis=-1)
        out = layers.Dense(2,activation='softmax')(added)
        model = models.Model(inputs=[input1, input2], outputs=out)

        #model.summary()
        # rmsp = optimizers.RMSprop(lr=0.0008)
        model.compile(optimizer='rmsprop',  # 'rmsprop'
                      loss='binary_crossentropy',
                      metrics=['accuracy']) # accuracy
        return model

    if nn_model == 1.03:
        #conv1D model for mCSM.
        network.add(layers.Conv1D(filters=16,kernel_size=3,activation='relu',padding='same',input_shape=(col_num,1)))
        network.add(layers.Conv1D(filters=32,kernel_size=3,activation='relu',padding='same'))
        network.add(layers.MaxPool1D(pool_size=2))
        network.add(layers.Conv1D(filters=64,kernel_size=3,activation='relu'))
        network.add(layers.MaxPool1D(pool_size=2))
        network.add(layers.Conv1D(filters=128,kernel_size=3,activation='relu'))
        network.add(layers.MaxPool1D(pool_size=2))
        network.add(layers.Flatten())
        network.add(layers.Dense(128,activation='relu'))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(2, activation='softmax'))
        # print(network.summary())
        adam = optimizers.adam(lr=1e-4, decay=1e-5)
        # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
        network.compile(optimizer=adam,  # SGD,adam,rmsprop
                        loss=[binary_focal_loss(alpha=.25, gamma=2)],  # 'binary_crossentropy',
                        metrics=['accuracy'])  # accuracy
        return network

    if nn_model == 2.01:
        #print('SingleNet regression')
        ## single-net for regression.
        # print('using single-net for regression...')
        network.add(layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu', input_shape=(row_num, col_num, 1)))
        # network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(layers.Conv2D(32, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(layers.Conv2D(64, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(layers.Flatten())
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.5))
        network.add(layers.Dense(16, activation='relu'))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(1))
        # print(network.summary())
        # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
        network.compile(optimizer='rmsprop',  # SGD,adam,rmsprop
                        loss='mse',
                        metrics=['mae'])  #mae平均绝对误差（mean absolute error） accuracy
        return network

    if nn_model == 2.02:
        #print('SingleNet-M regression')
        input1 = layers.Input(shape=(row_num,99,1),name='mCNN')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        dense1 = layers.Dense(128, activation='relu')(flat1)
        drop1 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(16, activation='relu')(drop1)
        drop_2 = layers.Dropout(0.3)(dense2)

        input2 = layers.Input(shape=(col_num-99,),name='mCSM')
        conv1  = layers.Conv1D(16,3,activation='relu')(input2)
        conv2  = layers.Conv1D(32,3,activation='relu')(conv1)
        pool1  = layers.MaxPooling1D(pool_size=2)(conv2)
        flat   = layers.Flatten()(pool1)
        dense1 = layers.Dense(32,activation='relu')(flat)
        drop_1  = layers.Dropout(0.1)(dense1)

        added = layers.concatenate([drop_2, drop_1],axis=-1)
        out = layers.Dense(1)(added)
        model = models.Model(inputs=[input1, input2], outputs=out)

        #model.summary()
        # rmsp = optimizers.RMSprop(lr=0.0008)
        adam = optimizers.adam(lr=1e-4, decay=1e-5)
        model.compile(optimizer=adam,  # 'rmsprop'
                      loss='mse',
                      metrics=['mae']) # accuracy
        return model

    if nn_model == 2.03:
        #conv1D model for mCSM.
        network.add(layers.Conv1D(filters=16,kernel_size=3,activation='relu',input_shape=(col_num,1)))
        network.add(layers.Conv1D(filters=32,kernel_size=3,activation='relu'))
        network.add(layers.MaxPool1D(pool_size=2))
        network.add(layers.Flatten())
        network.add(layers.Dense(32,activation='relu'))
        network.add(layers.Dropout(0.1))

        network.add(layers.Dense(1))
        adam = optimizers.adam(lr=1e-4, decay=1e-5)
        # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
        network.compile(optimizer=adam,  # SGD,adam,rmsprop
                        loss='mse',
                        metrics=['mae'])  # accuracy
        return network


if __name__ == '__main__':
    nn_model, sample_size = 3, (50,59)
    model = build_model(nn_model, sample_size)
    from IPython.display import SVG, display
    from keras.utils.vis_utils import model_to_dot
    # plot_model(model,show_shapes=True,to_file='model1.png')
    # plot_model(model, show_shapes=True, to_file='model.png')
