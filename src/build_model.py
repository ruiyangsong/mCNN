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
from keras import Input
from keras.utils import plot_model


def build_model(nn_model,sample_size):
    row_num,col_num = sample_size
    ## 序贯模型使用network, API模型使用model.
    network = models.Sequential()
    if nn_model == 1.01:
        #print('SingleNet classification')
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
        network.add(layers.Dense(2, activation='softmax'))
        # print(network.summary())
        # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
        network.compile(optimizer='rmsprop',  # SGD,adam,rmsprop
                        loss='binary_crossentropy',
                        metrics=['accuracy'])  # accuracy
        return network

    if nn_model == 1.02:
        #print('SingleNet-M classification')
        input1 = layers.Input(shape=(row_num,col_num-5,1),name='structure')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu')(drop)
        input2 = layers.Input(shape=(5,),name='delta_r')
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
        input1 = layers.Input(shape=(row_num,col_num-5,1),name='structure')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu')(drop)
        input2 = layers.Input(shape=(5,),name='delta_r')
        dense2 = layers.Dense(128, activation='relu')(input2)

        added = layers.concatenate([dense1, dense2],axis=-1)
        out = layers.Dense(1)(added)
        model = models.Model(inputs=[input1, input2], outputs=out)

        #model.summary()
        # rmsp = optimizers.RMSprop(lr=0.0008)
        model.compile(optimizer='rmsprop',  # 'rmsprop'
                      loss='mse',
                      metrics=['mae']) # accuracy
        return model


if __name__ == '__main__':
    nn_model, sample_size = 1, (50,17)
    model = build_model(nn_model, sample_size)
    from IPython.display import SVG, display
    from keras.utils.vis_utils import model_to_dot
    plot_model(model,show_shapes=True,to_file='model1.png')
    # plot_model(model, show_shapes=True, to_file='model.png')