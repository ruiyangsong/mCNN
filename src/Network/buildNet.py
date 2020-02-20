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
from sklearn.utils import class_weight
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
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    pass



    # load data and sort row (return python dictionary)-----------------------------------------------------------------
    
    if container['mCNN_wild_dir'] != '' and container['val_mCNN_wild_dir']:
        mCNN_wild


    if container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] == '':
        x_dict, y_dict, ddg_dict = load_sort_data(container['mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_val_dict, y_val_dict, ddg_val_dict = load_sort_data(container['val_mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])

    elif container['mCSM_arrdir'] != '' and container['mCNN_arrdir'] == '':
        x_dict, y_dict, ddg_dict = load_sort_data(container['mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_val_dict, y_val_dict, ddg_val_dict = load_sort_data(container['val_mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])

    if container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] != '' and append == 'True':
        x_mCNN_dict, y_mCNN_dict, ddg_mCNN_dict = load_sort_data(container['mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_mCNN_val_dict, y_mCNN_val_dict, ddg_mCNN_val_dict = load_sort_data(container['val_mCNN_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_mCSM_dict, y_mCSM_dict, ddg_mCSM_dict = load_sort_data(container['mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_mCSM_val_dict, y_mCSM_val_dict, ddg_mCSM_val_dict = load_sort_data(container['val_mCSM_arrdir'],wild_or_mutant,sort_method,seed_tuple[0])
        x_append_dict = append_mCSM(x_mCNN_dict=x_mCNN_dict, x_mCSM_dict=x_mCSM_dict)
        x_append_val_dict = append_mCSM(x_mCNN_dict=x_mCNN_val_dict, x_mCSM_dict=x_mCSM_val_dict)
        del x_mCNN_dict, x_mCSM_dict, x_mCNN_val_dict, x_mCSM_val_dict
        ################################################################################################################
        # 关于 append 的网络结构没有进行设计
        ################################################################################################################
    elif container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] != '' and append == 'True':
        pass

    # Split K-fold data. -----------------------------------------------------------------------------------------------
    print('%d-fold cross validation begin.' % (k))
    DE = DataExtractor
    DE.split_kfold(x_dict, y_dict, ddg_dict, fold_num=k, random_seed=seed_tuple[1], train_ratio = 0.7)
    key_lst = list(DE.x_test_dict.keys())
    val_num = int(DE.x_test_dict[key_lst[0]][0].shape[0])
    DE.split_val_data(x_val_dict, y_val_dict, ddg_val_dict, seed_tuple[2], val_num)


def cross_validation():
    pass


def save_model(model, model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        model.save('%s/%s'%(model_path, model_name))
        print('---model saved at %s.'%model_path)


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







class ConvNet(LossFunction, Metrics_Generator):
    def __init__(self, data, nn_model, output_num, input_shape, kernel_size=(3,3),initializer='random_uniform',
                 activator='relu', pool_size=(2,2), padding_style='same', regular_rate=(0.001,0.001), dropout_rate=0.3,
                 optimizer='adam', summary=True,
                 verbose=1, CUDA='0', epoch=100, batch_size=128, model_path=None):
        self.model       = None
        self.nn_model    = nn_model
        self.input_num   = len(data.x_train.keys()) # number of input layers
        self.output_num  = output_num # number of output layers, 2 output layers means training ddg and classes concurrently.
        self.input_shape = input_shape[1:]+(1,)# Attention HERE!

        self.kernel_size = kernel_size
        self.initializer = initializer
        self.activator   = activator

        self.pool_size     = pool_size
        self.padding_style = padding_style

        self.regular_rate  = regular_rate
        self.dropout_rate  = dropout_rate

        self.optimizer = optimizer
        self.summary   = summary
        if K.image_data_format() == 'channels_last':
            self.batch_norm_axis = -1
        elif K.image_data_format() == 'channels_first':
            self.batch_norm_axis = 1
        # init training params -----------------------------------------------------------------------------------------
        self.tag        = '%s_in%s_out%s'%(self.nn_model,self.input_num,self.output_num)
        self.data       = data
        self.verbose    = verbose
        self.CUDA       = CUDA
        self.epoch      = epoch
        self.batch_size = batch_size
        self.model_path = model_path

    def config_tf(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def run(self):
        self.config_tf()
        self.run_model()

    def run_model(self):
        if self.nn_model == 'muiti_task_conv2D':
            self.multi_task_conv2D()
        elif self.nn_model == 'muiti_task_conv1D':
            self.multi_task_conv1D()

        if self.nn_model == 'classifier_conv2D':
            self.classifier_conv2D()
        elif self.nn_model == 'classifier_conv1D':
            self.classifier_conv1D()

        if self.nn_model == 'regressor_conv2D':
            self.regressor_conv2D()
        elif self.nn_model == 'regressor_conv1D':
            self.regressor_conv1D()

    def calc_classes(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.data.y_train), self.data.y_train.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        class_num = len(np.unique(self.data.y_train))
        y_train = to_categorical(self.data.y_train)
        y_test = to_categorical(self.data.y_test)
        y_val = None
        if self.data.x_val:
            y_val = to_categorical(self.data.x_val)
        return class_weights_dict, class_num, y_train, y_test, y_val

    def multi_task_conv2D(self, loss_type_lst=['mse','binary_crossentropy'], loss_weights_lst=[0.5,10.],
                          metrics_lst=(['mae', Metrics_Generator.pearson_r, Metrics_Generator.rmse],
                                       ['accuracy', Metrics_Generator.mcc, Metrics_Generator.recall,
                                        Metrics_Generator.recall_p, Metrics_Generator.recall_n, Metrics_Generator.precision,
                                        Metrics_Generator.precision_p, Metrics_Generator.precision_n])):
        class_weights_dict, class_num, y_train, y_test, y_val = self.calc_classes()
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

            model.fit(x=self.data.x_train,
                      y={'ddg':self.data.ddg_train,
                         'class':y_train
                         },
                      batch_size=self.batch_size,
                      epochs=self.epoch,
                      verbose=self.verbose,
                      callbacks=None,
                      validation_data=(self.data.x_val,
                                       {'ddg':self.data.ddg_val,
                                        'class':y_val}
                                       ),
                      shuffle=True,
                      class_weight={'ddg':None,
                                    'class':class_weights_dict}
                      )

            ## evaluate calc params in metrics
            metrics_score_lst = model.evaluate(x=self.data.x_test,
                                               y={'ddg':self.data.ddg_test,
                                                  'class':y_test},
                                               verbose=self.verbose
                                               )

        elif self.input_num == 2 and self.output == 2:
            pass

    def multi_task_conv1D(self):
        pass

    def classifier_conv2D(self, loss_type='binary_crossentropy',
                          metrics=('accuracy', Metrics_Generator.mcc, Metrics_Generator.recall, Metrics_Generator.recall_p,
                                   Metrics_Generator.recall_n, Metrics_Generator.precision, Metrics_Generator.precision_p,
                                   Metrics_Generator.precision_n)):
        class_weights_dict, class_num, y_train, y_test, y_val = self.calc_classes()
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
                          metrics=list(metrics) # accuracy
                          )
            model.fit(x=self.data.x_train,
                      y=y_train,
                      batch_size=self.batch_size,
                      epochs=self.epoch,
                      verbose=self.verbose,
                      callbacks=None,
                      validation_data=(self.data.x_val, y_val),
                      shuffle=True,
                      class_weight=class_weights_dict
                      )
            metrics_score_lst = model.evaluate(x=self.data.x_test,
                                               y=y_test,
                                               verbose=self.verbose
                                               )

        elif self.input_num == 2 and self.output == 1:
            pass

    def classifier_conv1D(self):
        pass

    def regressor_conv2D(self, loss_type='mse', metrics=('mae', 'mse', Metrics_Generator.rmse, Metrics_Generator.pearson_r)):
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
                          metrics=list(metrics)
                          )
            model.fit(x=self.data.x_train,
                      y=self.data.ddg_train,
                      batch_size=self.batch_size,
                      epochs=self.epoch,
                      verbose=self.verbose,
                      callbacks=None,
                      validation_data=(self.x_val, self.data.ddg_val),
                      shuffle=True
                      )
            metrics_score_lst = model.evaluate(x=self.data.x_test,
                           y=self.data.ddg_test,
                           verbose=self.verbose
                           )

        elif self.input_num == 2 and self.output_num == 1:
            pass

    def regressor_conv1D(self):
        pass


    def save(self, model_dir):
        self.model.save(model_dir)
        print('Model Saved.')

    def load(self, model_dir):
        self.model = load_model(model_dir)
        print('Model Loaded.')









def build_model(nn_model,sample_size,):
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
                        loss=[LossFunction.binary_focal_loss(alpha=.25, gamma=2)],#'binary_crossentropy',
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
    if args.mCNN:
        str_pca, str_k_neighbor = args.mCNN
        if wild_or_mutant == 'stack':
            container['mCNN_wild_dir']       = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,center,str_pca,str_k_neighbor)
            container['mCNN_mutant_dir']     = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,center,str_pca,str_k_neighbor)
            if args.val_dataset_name:
                container['val_mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
                container['val_mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,center,str_pca,str_k_neighbor)
        else:
            container['mCNN_%s_dir'%wild_or_mutant]     = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
            if args.val_dataset_name:
                container['val_mCNN_%s_dir'%wild_or_mutant] = '%s/mCNN/dataset/%s/feature/mCNN/%s/npz/center_%s_PCA_%s_neighbor_%s.npz' %(homedir,val_dataset_name,wild_or_mutant,center,str_pca,str_k_neighbor)
    elif args.mCSM:
        min_, max_, step, atom_class_num = args.mCSM
        if wild_or_mutant == 'stack':
            container['mCNN_wild_dir']       = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, min_, max_, step, center, atom_class_num)
            container['mCNN_mutant_dir']     = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, min_, max_, step, center, atom_class_num)
            if args.val_dataset_name:
                container['val_mCNN_wild_dir']   = '%s/mCNN/dataset/%s/feature/mCSM/wild/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, val_dataset_name, min_, max_, step, center, atom_class_num)
                container['val_mCNN_mutant_dir'] = '%s/mCNN/dataset/%s/feature/mCSM/mutant/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, val_dataset_name, min_, max_, step, center, atom_class_num)
        else:
            container['mCSM_%s_dir'%wild_or_mutant]     = '%s/mCNN/dataset/%s/feature/mCSM/%s/npz/min_%s_max_%s_step_%s_center_%s_class_%s.npz' %(homedir, dataset_name, wild_or_mutant, min_, max_, step, center, atom_class_num)
            if args.val_dataset_name:
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
    print('dataset_name: %s, wild_or_mutant: %s, center: %s,'
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
          % (dataset_name, wild_or_mutant, center, container['mCNN_wild_dir'], container['mCNN_mutant_dir'],
             container['val_mCNN_wild_dir'], container['val_mCNN_mutant_dir'], container['mCSM_wild_dir'], container['mCSM_mutant_dir'], 
             container['val_mCSM_wild_dir'], container['val_mCSM_mutant_dir'],append, normalize_method, sort_method, seed_tuple,
             nn_model, k, verbose, epoch, batch_size, CUDA))
