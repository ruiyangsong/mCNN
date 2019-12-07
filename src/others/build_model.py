#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : build_model.py
# time      : 3/29/2019 15:18
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from keras import Input
from keras.utils import plot_model

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
