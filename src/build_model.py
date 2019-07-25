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
    """
    :function:
    :param nn_model: network model to choose.
    :sample_size: size of one sample.
    """

    row_num,col_num = sample_size
    channel_num = 12 if col_num == 17 else 11
    ## 序贯模型使用network, API模型使用model.
    network = models.Sequential()
    if nn_model == 0:
        # print('using dense network...')

        network.add(layers.Dense(800, activation='relu', input_shape=(row_num * col_num,))) #437*15
        network.add(layers.Dense(100, activation='relu'))
        network.add(layers.Dense(20, activation='relu'))
        network.add(layers.Dense(2, activation='sigmoid'))  # softmax, sigmoid
        network.compile(optimizer='SGD', #rmsprop
                        loss='binary_crossentropy',
                        metrics=['accuracy']) # loss = [categorical_crossentropy, binary_crossentropy]
        return network
    if nn_model == 1:
        #print('SingleNet classification')
        ##分类模型
        # print('using CNN network to classification...')

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

    if nn_model == 1.03:
        #print('SingleNet without Delta_res, classification')
        network.add(layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu', input_shape=(row_num, col_num-5, 1)))
        network.add(layers.Conv2D(32, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        network.add(layers.Conv2D(64, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(layers.Flatten())
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.5))
        network.add(layers.Dense(16, activation='relu'))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(2, activation='softmax'))
        network.compile(optimizer='rmsprop', loss='binary_crossentropy',
                        metrics=['accuracy'])  # accuracy
        return network

    if nn_model == 1.01:
        #print('MultiNet-M classification')
        ## 分类模型，多通道表示，[[dist], [x,y,z], [ph, temperature], [rsa], [one-hot]] # 5! = 120.
        ## 对于S1932数据集，列属性排列为120
        ## 对于S2648数据集，列属性也为120，去除了rsa, 拆开ph和temperature.
        # print('using multi_channel CNN network...')
        input1 = layers.Input(shape=(120,120,channel_num,),name='structure')

        conv1 = layers.Conv2D(16,(5,5),activation='relu')(input1)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (5, 5),activation='relu')(pool)
        pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = layers.Conv2D(32, (5, 5),activation='relu')(pool1)
        pool = layers.MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = layers.Conv2D(32, (5, 5),activation='relu')(pool)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = layers.Conv2D(256, (3, 3))(pool2)
        # conv6 = layers.Conv2D(256, (3, 3))(conv5)
        # conv7 = layers.Conv2D(256, (3, 3))(conv6)
        #
        # pool3 = layers.AveragePooling2D(pool_size=(2, 2))(conv7)

        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(drop)

        input2 = layers.Input(shape=(5,),name='pchange')
        dense3 = layers.Dense(128, activation='relu')(input2)

        added = layers.concatenate([dense1, dense3],axis=-1)
        out = layers.Dense(2,activation='softmax')(added)

        model = models.Model(inputs=[input1, input2], outputs=out)

        #model.summary()
        rmsp = optimizers.RMSprop(lr=0.0008)
        model.compile(optimizer=rmsp,  # 'rmsprop'
                      loss='binary_crossentropy',
                      metrics=['accuracy']) # accuracy
        return model

    if nn_model == 1.02:
        #print('SingleNet-M classification')
        input1 = layers.Input(shape=(row_num,channel_num,1),name='structure')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu')(drop)
        input2 = layers.Input(shape=(5,),name='pchange')
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

    if nn_model == 2:
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

    if nn_model == 2.03:
        #print('SingleNet without Delta_res, regression')
        network.add(layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu', input_shape=(row_num, col_num-5, 1)))
        network.add(layers.Conv2D(32, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        network.add(layers.Conv2D(64, (5, 3), activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(layers.Flatten())
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.5))
        network.add(layers.Dense(16, activation='relu'))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(1))
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return network

    if nn_model == 2.01:
        #print('MultiNet-M regression')
        ## multi-net for regression.
        input1 = layers.Input(shape=(120, 120, channel_num,), name='structure')

        conv1 = layers.Conv2D(16, (5, 5))(input1)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (5, 5))(pool)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(32, (5, 5))(pool1)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(32, (5, 5), )(pool)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = layers.Conv2D(256, (3, 3))(pool2)
        # conv6 = layers.Conv2D(256, (3, 3))(conv5)
        # conv7 = layers.Conv2D(256, (3, 3))(conv6)
        #
        # pool3 = layers.AveragePooling2D(pool_size=(2, 2))(conv7)

        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(drop)

        input2 = layers.Input(shape=(5,), name='pchange')
        dense3 = layers.Dense(128, activation='relu')(input2)

        added = layers.concatenate([dense1, dense3], axis=-1)
        out = layers.Dense(1)(added)

        model = models.Model(inputs=[input1, input2], outputs=out)

        # model.summary()
        rmsp = optimizers.RMSprop(lr=0.0008)
        model.compile(optimizer=rmsp,  # 'rmsprop'
                      loss='mse',
                      metrics=['mae'])
        return model

    if nn_model == 2.02:
        #print('SingleNet-M regression')
        input1 = layers.Input(shape=(row_num,channel_num,1),name='structure')
        conv1 = layers.Conv2D(16,(5,3),activation='relu')(input1)
        conv2 = layers.Conv2D(32, (5, 3),activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv2)
        conv3 = layers.Conv2D(64, (5, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
        flat1 = layers.Flatten()(pool2)
        drop = layers.Dropout(0.3)(flat1)
        dense1 = layers.Dense(128, activation='relu')(drop)
        input2 = layers.Input(shape=(5,),name='pchange')
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


    #
    # plot_model(model, show_shapes=True, to_file='model.png')
