#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file_name : build_model.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import os, argparse
import numpy as np
from sklearn.utils import class_weight
from mCNN.Network.Data import DataExtractor
from mCNN.Network.metrics import pearson_r, rmse, mcc, recall, recall_p, recall_n, precision, precision_p, precision_n
from mCNN.Network.CallBack import TrainCallback
from mCNN.processing import shell, append_mCSM, str2bool

import tensorflow as tf

from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.models import Model, load_model
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, regularizers, optimizers
from keras.callbacks import ModelCheckpoint



def main():
    # adam = optimizers.adam(lr=1e-4, decay=1e-5)
    pass


def cross_validation():
    print('\n----------Cross validation, fold number is %s'%k)
    fold_flag = 0
    '''all the parameters are got from args'''
    DE = DataExtractor(container=container, sort_method = sort_method, permutation_seed=seed_tuple[0], normalize_method = normalize_method, val=split_val)
    DE.split_kfold(fold_num=k, k_fold_seed = seed_tuple[1], val_seed = seed_tuple[2], train_ratio = 0.7)
    for data_dict in DE.data_lst:
        fold_flag += 1
        print('\n----------Fold %s is running'%fold_flag)
        network = ConvNet(data_dict,input_num=1,model_base_dir=None,CUDA=CUDA)

        # network.multi_task_conv2D(kernel_size=(3,3),pool_size=(2,2),initializer='random_uniform', padding_style='same',
        #                           activator='relu', regular_rate=(0.001,0.001), dropout_rate = 0.1, optimizer='adam',
        #                           summary=True, loss_type_lst=['mse','binary_crossentropy'],loss_weights_lst=[0.5,10.],
        #                           metrics_lst=(['mae', pearson_r, rmse],
        #                                        ['accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n]),
        #                           batch_size=batch_size, epoch=epoch, verbose=verbose, callbacks=None)

        # network.multi_task_conv2D(kernel_size=(3,3),pool_size=(2,2),initializer='random_uniform', padding_style='same',
        #                           activator='relu', regular_rate=(0.01,0.01), dropout_rate = 0.3, optimizer='adam',
        #                           summary=False, loss_type_lst=['mse','binary_crossentropy'],loss_weights_lst=[0.1,0.5],
        #                           metrics_lst=(['mae'],['accuracy']),
        #                           batch_size=batch_size, epoch=epoch, verbose=verbose, callbacks=None)

        network.classifier_conv2D(kernel_size=(3, 3), pool_size=(2, 2), initializer='random_uniform',padding_style='same',
                                  activator='relu', regular_rate=(0.001, 0.1), dropout_rate=0.3, optimizer='adam',
                                  summary=True, batch_size=batch_size, epoch=epoch, verbose=verbose, callbacks=None,
                                  loss_type='binary_crossentropy',
                                  metrics=('accuracy', mcc))

class ConvNet(object):
    def __init__(self, data_dict,input_num, model_base_dir=None,CUDA='0'):
        self.model       = None
        self.data_dict   = data_dict
        self.input_num   = input_num
        self.model_base_dir = model_base_dir
        self.CUDA = CUDA

        self.config_tf()

    def config_tf(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        if K.image_data_format() == 'channels_last':
            self.batch_norm_axis = -1
        elif K.image_data_format() == 'channels_first':
            self.batch_norm_axis = 1

    def save_model(model, model_path, model_name):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            model.save('%s/%s' % (model_path, model_name))
            print('---model saved at %s.' % model_path)

    def multi_task_conv2D(self, kernel_size=(3,3),pool_size=(2,2),initializer='random_uniform', padding_style='same',
                          activator='relu', regular_rate=(0.001,0.001), dropout_rate = 0.1, optimizer='adam', summary=True,
                          loss_type_lst=('mse','binary_crossentropy'),loss_weights_lst=(0.5,10.),
                          metrics_lst=(['mae', pearson_r, rmse],
                                       ['accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n]),
                          batch_size=128, epoch=100, verbose=1, callbacks=None):
        '''conv2D are only used for mCNN features, but mCNN feature can be divided into multi inputs'''
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.data_dict['mCNN'].Train.y), self.data_dict['mCNN'].Train.y.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        class_num = len(np.unique(self.data_dict['mCNN'].Train.y))
        if self.input_num == 1:
            # mCNN feature inputs as a whole 2D array
            input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
            conv1 = layers.Conv2D(16,kernel_size,kernel_initializer=initializer,activation=activator)(input_layer)
            conv2 = layers.Conv2D(32,kernel_size,kernel_initializer=initializer,activation=activator)(conv1)
            pool1 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv2)
            conv3 = layers.Conv2D(64,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv4)
            flat = layers.Flatten()(pool3)

            dense = layers.Dense(128, activation=activator)(flat)
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(dropout_rate)(dense_BatchNorm)

            ddg_prediction = layers.Dense(1, name='ddg')(drop)
            class_prediction = layers.Dense(class_num,activation='softmax',name='class')(drop)

            model = Model(inputs=input_layer, outputs=[ddg_prediction,class_prediction])

            if summary:
                model.summary()
            ############################################################################################################
            # Train Model
            ############################################################################################################
            model.compile(optimizer=optimizer,
                          loss={'ddg':loss_type_lst[0],
                                'class':loss_type_lst[1]
                                },
                          loss_weights={'ddg':loss_weights_lst[0],
                                        'class':loss_weights_lst[1]
                                        },
                          metrics={'ddg':metrics_lst[0],
                                   'class':metrics_lst[1]
                                   }
                          )
            x_train = self.data_dict['mCNN'].Train.x[:, :, :, np.newaxis]
            y_train = to_categorical(self.data_dict['mCNN'].Train.y)
            x_test = self.data_dict['mCNN'].Test.x[:, :, :, np.newaxis]
            y_test = to_categorical(self.data_dict['mCNN'].Test.y)
            if self.data_dict['mCNN'].Val is not None:
                x_val = self.data_dict['mCNN'].Val.x[:, :, :, np.newaxis]
                y_val = to_categorical(self.data_dict['mCNN'].Val.y)

                model.fit(x=x_train,
                          y={'ddg':self.data_dict['mCNN'].Train.ddg,
                             'class':y_train
                             },
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          validation_data=(x_val,
                                           {'ddg':self.data_dict['mCNN'].Val.ddg,
                                            'class':y_val}
                                           ),
                          shuffle=True,
                          class_weight={'ddg':None,
                                        'class':class_weights_dict}
                          )
            else:
                model.fit(x=x_train,
                          y={'ddg':self.data_dict['mCNN'].Train.ddg,
                             'class':y_train
                             },
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=True,
                          class_weight={'ddg':None,
                                        'class':class_weights_dict}
                          )
            print('\n----------Test model')
            ## evaluate calc params in metrics
            metrics_score_lst = model.evaluate(x=x_test,
                                               y={'ddg':self.data_dict['mCNN'].Test.ddg,
                                                  'class':y_test},
                                               verbose=verbose
                                               )
            print('\nmetrics_score_lst', metrics_score_lst)

        elif self.input_num == 2:
            # mCNN feature can be divided into multi inputs
            pass

    def multi_task_conv1D(self):
        # conv1D are only used for mCSM feature
        pass

    def classifier_conv2D(self, kernel_size=(3,3),pool_size=(2,2),initializer='random_uniform', padding_style='same',
                          activator='relu', regular_rate=(0.001,0.001), dropout_rate = 0.1, optimizer='adam',
                          summary=True,batch_size=128, epoch=100, verbose=1, callbacks=None,
                          loss_type='binary_crossentropy',
                          metrics=('accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n)):
        # conv2D are only used for mCNN features, but mCNN feature can be divided into multi inputs
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.data_dict['mCNN'].Train.y), self.data_dict['mCNN'].Train.y.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        class_num = len(np.unique(self.data_dict['mCNN'].Train.y))
        if self.input_num == 1:
            # mCNN feature inputs as a whole 2D array
            input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
            conv1 = layers.Conv2D(16,kernel_size,kernel_initializer=initializer,activation=activator)(input_layer)
            conv2 = layers.Conv2D(32,kernel_size,kernel_initializer=initializer,activation=activator)(conv1)
            pool1 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv2)
            conv3 = layers.Conv2D(64,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv4)
            flat = layers.Flatten()(pool3)

            dense = layers.Dense(128, activation=activator)(flat)
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(dropout_rate)(dense_BatchNorm)

            output_layer = layers.Dense(class_num,activation='softmax')(drop)
            model = models.Model(inputs=input_layer, outputs=output_layer)

            if summary:
                model.summary()
            ############################################################################################################
            # Train Model
            ############################################################################################################
            # rmsp = optimizers.RMSprop(lr=0.0008)
            model.compile(optimizer=optimizer,
                          loss=loss_type,
                          metrics=list(metrics) # accuracy
                          )
            x_train = self.data_dict['mCNN'].Train.x[:,:,:,np.newaxis]
            y_train = to_categorical(self.data_dict['mCNN'].Train.y)
            x_test = self.data_dict['mCNN'].Test.x[:, :, :, np.newaxis]
            y_test = to_categorical(self.data_dict['mCNN'].Test.y)
            if self.data_dict['mCNN'].Val is not None:
                x_val = self.data_dict['mCNN'].Val.x[:, :, :, np.newaxis]
                y_val = to_categorical(self.data_dict['mCNN'].Val.y)
                model.fit(x=x_train,
                          y=y_train,
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          validation_data=(x_val, y_val),
                          shuffle=True,
                          class_weight=class_weights_dict
                          )
            else:
                model.fit(x=x_train,
                          y=y_train,
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=True,
                          class_weight=class_weights_dict
                          )
            print('\n----------Test model')
            metrics_score_lst = model.evaluate(x=x_test,
                                               y=y_test,
                                               verbose=verbose
                                               )
            print('\nmetrics_score_lst:', metrics_score_lst)

        elif self.input_num == 2:
            pass

    def classifier_conv1D(self):
        pass

    def regressor_conv2D(self, kernel_size=(3,3),pool_size=(2,2),initializer='random_uniform', padding_style='same',
                          activator='relu', regular_rate=(0.001,0.001), dropout_rate = 0.1, optimizer='adam',
                          summary=True,batch_size=128, epoch=100, verbose=1, callbacks=None,
                          loss_type='mse', metrics=('mae', 'mse', rmse, pearson_r)):
        # conv2D are only used for mCNN features, but mCNN feature can be divided into multi inputs
        class_num = 1
        if self.input_num == 1:
            # mCNN feature inputs as a whole 2D array
            input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
            conv1 = layers.Conv2D(16,kernel_size,kernel_initializer=initializer,activation=activator)(input_layer)
            conv2 = layers.Conv2D(32,kernel_size,kernel_initializer=initializer,activation=activator)(conv1)
            pool1 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv2)
            conv3 = layers.Conv2D(64,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool1)
            conv3_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(conv3)
            pool2 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv3_BatchNorm)
            conv4 = layers.Conv2D(128,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool2)
            pool3 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv4)
            flat = layers.Flatten()(pool3)

            dense = layers.Dense(128, activation=activator)(flat)
            dense_BatchNorm = layers.BatchNormalization(axis=self.batch_norm_axis)(dense)
            drop  = layers.Dropout(dropout_rate)(dense_BatchNorm)

            output_layer = layers.Dense(class_num)(drop)
            model = models.Model(inputs=input_layer, outputs=output_layer)

            if summary:
                model.summary()
            ############################################################################################################
            # Train Model
            ############################################################################################################
            # rmsp = optimizers.RMSprop(lr=0.0008)
            model.compile(optimizer=optimizer,
                          loss=loss_type,
                          metrics=list(metrics)
                          )
            x_train = self.data_dict['mCNN'].Train.x[:, :, :, np.newaxis]
            x_test = self.data_dict['mCNN'].Test.x[:, :, :, np.newaxis]
            if self.data_dict['mCNN'].Val is not None:
                x_val = self.data_dict['mCNN'].Val.x[:, :, :, np.newaxis]
                model.fit(x=x_train,
                          y=self.data_dict['mCNN'].Train.ddg,
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          validation_data=(x_val, self.data_dict['mCNN'].Val.ddg),
                          shuffle=True,
                          )
            else:
                model.fit(x=x_train,
                          y=self.data_dict['mCNN'].Train.ddg,
                          batch_size=batch_size,
                          epochs=epoch,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=True,
                          )
            print('\n----------Test model')
            metrics_score_lst = model.evaluate(x=x_test,
                                               y=self.data_dict['mCNN'].Test.ddg,
                                               verbose=verbose
                                               )
            print('\nmetrics_score_lst:', metrics_score_lst)

        elif self.input_num == 2:
            pass


    def regressor_conv1D(self):
        pass


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
    # get command line params ------------------------------------------------------------------------------------------
    homedir = shell('echo $HOME')
    ## Init container
    container = {'mCNN_wild_dir': '', 'mCNN_mutant_dir': '', 'val_mCNN_wild_dir': '', 'val_mCNN_mutant_dir': '',
                 'mCSM_wild_dir': '', 'mCSM_mutant_dir': '', 'val_mCSM_wild_dir': '', 'val_mCSM_mutant_dir': '', }
    ## Input parameters.
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
    wild_or_mutant = args.wild_or_mutant
    center = args.center
    split_val = str2bool(args.split_val)
    if args.mCNN:
        str_pca, str_k_neighbor = args.mCNN
        if wild_or_mutant == 'stack':
            container['mCNN_wild_dir']       = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,center,str_pca,str_k_neighbor)
            container['mCNN_mutant_dir']     = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,center,str_pca,str_k_neighbor)
            if args.val_dataset_name:
                val_dataset_name = args.val_dataset_name
                container['val_mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
                container['val_mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
        else:
            container['mCNN_%s_dir'%wild_or_mutant]     = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
            if args.val_dataset_name:
                val_dataset_name = args.val_dataset_name
                container['val_mCNN_%s_dir'%wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
    elif args.mCSM:
        min_, max_, step, atom_class_num = args.mCSM
        if wild_or_mutant == 'stack':
            container['mCSM_wild_dir']       = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, min_, max_, step, center, atom_class_num)
            container['mCSM_mutant_dir']     = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, min_, max_, step, center, atom_class_num)
            if args.val_dataset_name:
                val_dataset_name = args.val_dataset_name
                container['val_mCSM_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, val_dataset_name, min_, max_, step, center, atom_class_num)
                container['val_mCSM_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, val_dataset_name, min_, max_, step, center, atom_class_num)
        else:
            container['mCSM_%s_dir'%wild_or_mutant]     = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, wild_or_mutant, min_, max_, step, center, atom_class_num)
            if args.val_dataset_name:
                val_dataset_name = args.val_dataset_name
                container['val_mCSM_%s_dir'%wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, val_dataset_name, wild_or_mutant, min_, max_, step, center, atom_class_num)
    
    else:
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
    # print input info. ------------------------------------------------------------------------------------------------
    print('\n----------Input params.'
          '\ndataset_name: %s, wild_or_mutant: %s, center: %s, split_val: %s,'
          '\nmCNN_wild_dir: %s,'
          '\nmCNN_mutant_dir: %s,'
          '\nval_mCNN_wild_dir: %s,'
          '\nval_mCNN_mutant_dir: %s,'
          '\nmCSM_wild_dir: %s,'
          '\nmCSM_mutant_dir: %s,'
          '\nval_mCSM_wild_dir: %s,'
          '\nval_mCSM_mutant_dir: %s,'
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
          % (dataset_name, wild_or_mutant, center, split_val, container['mCNN_wild_dir'], container['mCNN_mutant_dir'],
             container['val_mCNN_wild_dir'], container['val_mCNN_mutant_dir'], container['mCSM_wild_dir'], container['mCSM_mutant_dir'], 
             container['val_mCSM_wild_dir'], container['val_mCSM_mutant_dir'],append, normalize_method, sort_method, seed_tuple,
             nn_model, k, verbose, epoch, batch_size, CUDA))

    cross_validation()
