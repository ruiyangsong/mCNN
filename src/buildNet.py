#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : build_model.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------
import sys
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from keras import Input
from keras.utils import plot_model


import numpy as np
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


class MyCallback(keras.callbacks.Callback):
    callbacks_list = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss'
        factor = 0.1,
                 patience = 10,
    )
    ]
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

class DataSet(object):
   def __init__(self, data_dir):
       self.num_classes = None
       self.X_train = None
       self.X_test = None
       self.Y_train = None
       self.Y_test = None
       self.img_size = 128
       self.extract_data(data_dir)

   def extract_data(self,path):
        #根据指定路径读取出图片、标签和类别数
        imgs,labels,counter = read_file(data_array_dir)

        #将数据集打乱随机分组
        X_train,X_test,y_train,y_test = train_test_split(imgs,labels,test_size=0.2,random_state=random.randint(0, 100))

        #重新格式化和标准化
        # 本案例是基于thano的，如果基于tensorflow的backend需要进行修改
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size)/255.0
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size) / 255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        #将labels转成 binary class matrices
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)

        #将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

   def check(self):
       print('num of dim:', self.X_test.ndim)
       print('shape:', self.X_test.shape)
       print('size:', self.X_test.size)
       print('num of dim:', self.X_train.ndim)
       print('shape:', self.X_train.shape)
       print('size:', self.X_train.size)

class ConvNet(object):
    def __init__(self, model_path):
        self.tag        = None
        self.model      = None
        self.model_path = model_path

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

    def In1_Out2(self, input_shape, class_num, kernel_size, activator, dropout, kernel_init, regular, dformat, summary=True):
        Input_merge = Input(shape=input_shape,
                            dtype='float32',
                            name='merge')

        x = layers.Conv1D(128, 5, activation=activator)(Input_merge)
        y = layers.MaxPooling1D(5)(x)

        y = layers.BatchNormalization(axis=-1)(y)
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

    def In2_Out2(self, input_shape, class_num, kernel_size, activator, dropout, kernel_init, regular, dformat, summary):
        Input_merge = Input(shape=input_shape,
                            dtype='float32',
                            name='merge')

        x = layers.Conv1D(128, 5, activation=activator)(Input_merge)
        x = layers.MaxPooling1D(5)(x)

        Output_class = layers.Dense(class_num,
                                    activation=activator,
                                    name='class')(x)
        Output_ddg   = layers.Dense(1,
                                    name='ddg')(x)
        model = Model(Input_merge, [Output_class, Output_ddg])
        self.model = model
        if summary:
            self.model.summary()

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
