#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, regularizers, optimizers
from mCNN.Network.metrics import pearson_r, rmse, mcc, recall, recall_p, recall_n, precision, precision_p, precision_n
from mCNN.Network.CallBack import TrainCallback


class ConfigTF(object):
    def __init__(self, CUDA):
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        if K.image_data_format() == 'channels_last':
            self.batch_norm_axis = -1
        elif K.image_data_format() == 'channels_first':
            self.batch_norm_axis = 1

########################################################################################################################
# Conv2D Network (adam = optimizers.adam(lr=1e-4, decay=1e-5)
########################################################################################################################
class Conv2DMultiTaskIn1(ConfigTF):
    def __init__(self, CUDA, data_dict, summary, batch_size, epoch, verbose):
        super().__init__(CUDA)
        self.data_dict = data_dict
        self.summary = summary
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

        self.model = None

        self.set_HyperParams()
        self.build()


    def set_HyperParams(self):
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.initializer = 'random_uniform'
        self.padding_style = 'same'
        self.activator = 'relu'
        self.regular_rate = (0.001, 0.001)
        self.dropout_rate = 0.1
        self.optimizer = 'adam'
        self.loss_type_lst = ('mse', 'binary_crossentropy')
        self.loss_weights_lst = (0.5, 10.)
        self.metrics_lst = (['mae', pearson_r, rmse],
                            ['accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n]),
        self.callbacks = None


    def build(self):
        '''mCNN feature inputs as a whole 2D array'''
        input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
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
        class_prediction = layers.Dense(len(np.unique(self.data_dict['mCNN'].Train.y)),activation='softmax',name='class')(drop)

        self.model = Model(inputs=input_layer, outputs=[ddg_prediction,class_prediction])

        if self.summary:
            self.model.summary()


    def train(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.data_dict['mCNN'].Train.y), self.data_dict['mCNN'].Train.y.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        self.model.compile(optimizer=self.optimizer,
                           loss={'ddg':self.loss_type_lst[0],
                                 'class':self.loss_type_lst[1]
                                 },
                           loss_weights={'ddg':self.loss_weights_lst[0],
                                         'class':self.loss_weights_lst[1]
                                         },
                           metrics={'ddg':self.metrics_lst[0],
                                    'class':self.metrics_lst[1]
                                    }
                           )

        if self.data_dict['mCNN'].Val is not None:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y={'ddg':self.data_dict['mCNN'].Train.ddg,
                              'class':self.data_dict['mCNN'].Train.y
                              },
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           validation_data=(self.data_dict['mCNN'].Val.x,
                                            {'ddg':self.data_dict['mCNN'].Val.ddg,
                                             'class':self.data_dict['mCNN'].Val.y}
                                            ),
                           shuffle=True,
                           class_weight={'ddg':None,
                                         'class':class_weights_dict}
                           )
        else:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y={'ddg':self.data_dict['mCNN'].Train.ddg,
                              'class':self.data_dict['mCNN'].Train.y
                              },
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           shuffle=True,
                           class_weight={'ddg':None,
                                         'class':class_weights_dict}
                           )


    def evaluate(self):
            print('\n----------Test model')
            ## evaluate calc params in metrics
            metrics_score_lst = self.model.evaluate(x=self.data_dict['mCNN'].Test.x,
                                                    y={'ddg':self.data_dict['mCNN'].Test.ddg,
                                                       'class':self.data_dict['mCNN'].Test.y},
                                                    verbose=self.verbose
                                                    )
            print('\nmetrics_score_lst', metrics_score_lst)




class Conv2DClassifierIn1(ConfigTF):
    def __init__(self, CUDA, data_dict, summary, batch_size, epoch, verbose):
        super().__init__(CUDA)
        self.data_dict = data_dict
        self.summary = summary
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

        self.model = None

        self.set_HyperParams()
        self.build()


    def setHyperParams(self):
        self.kernel_size=(3,3)
        self.pool_size=(2,2)
        self.initializer='random_uniform'
        self.padding_style='same',
        self.activator='relu'
        self.regular_rate=(0.001,0.001)
        self.dropout_rate = 0.1
        self.optimizer='adam'
        self.loss_type='binary_crossentropy'
        self.metrics=('accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n)
        self.callbacks = None


    def build(self):
        # mCNN feature inputs as a whole 2D array
        input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
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

        output_layer = layers.Dense(len(np.unique(self.data_dict['mCNN'].Train.y)),activation='softmax')(drop)
        self.model = models.Model(inputs=input_layer, outputs=output_layer)

        if self.summary:
            self.model.summary()


    def train(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.data_dict['mCNN'].Train.y), self.data_dict['mCNN'].Train.y.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_type,
                           metrics=list(self.metrics) # accuracy
                           )

        if self.data_dict['mCNN'].Val is not None:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y=self.data_dict['mCNN'].Train.y,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           validation_data=(self.data_dict['mCNN'].Val.x, self.data_dict['mCNN'].Val.y),
                           shuffle=True,
                           class_weight=class_weights_dict
                           )
        else:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y=self.data_dict['mCNN'].Train.y,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           shuffle=True,
                           class_weight=class_weights_dict
                           )


    def evaluate(self):
        print('\n----------Test model')
        metrics_score_lst = self.model.evaluate(x=self.data_dict['mCNN'].Test.x,
                                                y=self.data_dict['mCNN'].Test.y,
                                                verbose=self.verbose
                                                )
        print('\nmetrics_score_lst:', metrics_score_lst)


class Conv2DRegressorIn1(ConfigTF):
    def __init__(self, CUDA, data_dict, summary, batch_size, epoch, verbose):
        super().__init__(CUDA)
        self.data_dict = data_dict
        self.summary = summary
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

        self.model = None

        self.set_HyperParams()
        self.build()


    def set_HyperParams(self):
        self.kernel_size=(3,3)
        self.pool_size=(2,2)
        self.initializer='random_uniform'
        self.padding_style='same'
        self.activator='relu'
        self.regular_rate=(0.001,0.001)
        self.dropout_rate = 0.1
        self.optimizer='adam'
        self.loss_type='mse'
        self.metrics=('mae', 'mse', rmse, pearson_r)
        self.callbacks = None

    def build(self):
        # mCNN feature inputs as a whole 2D array
        input_layer = Input(shape=self.data_dict['mCNN'].Train.x.shape[1:] + (1,))
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

        output_layer = layers.Dense(self.class_num)(drop)
        self.model = models.Model(inputs=input_layer, outputs=output_layer)

        if self.summary:
            self.model.summary()


    def train(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_type,
                           metrics=list(self.metrics)
                           )
        if self.data_dict['mCNN'].Val is not None:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y=self.data_dict['mCNN'].Train.ddg,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           validation_data=(self.data_dict['mCNN'].Val.x, self.data_dict['mCNN'].Val.ddg),
                           shuffle=True,
                           )
        else:
            self.model.fit(x=self.data_dict['mCNN'].Train.x,
                           y=self.data_dict['mCNN'].Train.ddg,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           shuffle=True,
                           )


    def evaluate(self):
        print('\n----------Test model')
        metrics_score_lst = self.model.evaluate(x=self.data_dict['mCNN'].Test.x,
                                                y=self.data_dict['mCNN'].Test.ddg,
                                                verbose=self.verbose
                                                )
        print('\nmetrics_score_lst:', metrics_score_lst)


class Conv2DMultiTaskIn2(ConfigTF):
    '''mCNN feature can be divided into multi inputs'''
    pass


class Conv2DClassifierIn2(ConfigTF):
    pass


class Conv2DRegressorIn2(ConfigTF):
    pass

########################################################################################################################
# Conv1D Network
########################################################################################################################
class Conv1DMultiTask(ConfigTF):
    '''conv1D are only used for mCSM feature'''
    pass


class Conv1DClassifier(ConfigTF):
    '''conv1D are only used for mCSM feature'''
    pass


class Conv1DRegressor(ConfigTF):
    '''conv1D are only used for mCSM feature'''
    pass

if __name__ == '__main__':
    pass