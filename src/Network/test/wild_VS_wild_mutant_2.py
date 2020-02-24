from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras import Input, models, layers, regularizers, callbacks, optimizers
from keras.utils import to_categorical


def data():
    random_seed = 10
    ####################################################
    # 处理data1
    ####################################################
    # data_1 = np.load('/public/home/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz')
    data_1 = np.load(r'E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\wild\center_CA_PCA_False_neighbor_30.npz')
    x = data_1['x']
    y = data_1['y']
    ddg = data_1['ddg'].reshape(-1)
    train_num = x.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    positive_indices, negative_indices = ddg >= 0, ddg < 0
    x_positive, x_negative = x[positive_indices], x[negative_indices]
    y_positive, y_negative = y[positive_indices], y[negative_indices]
    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train, x_test = np.vstack((x_positive[:left_positive], x_negative[:left_negative])), np.vstack(
        (x_positive[left_positive:], x_negative[left_negative:]))
    y_train, y_test = np.vstack((y_positive[:left_positive], y_negative[:left_negative])), np.vstack(
        (y_positive[left_positive:], y_negative[left_negative:]))
    # sort row default is chain
    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # normalization
    train_shape = x_train.shape
    test_shape = x_test.shape
    col_train = train_shape[-1]
    col_test = test_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_test = x_test.reshape((-1, col_test))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_train = x_train.reshape(train_shape)
    x_test = x_test.reshape(test_shape)

    # reshape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    # print(x_train.shape, y_train.shape)
    ####################################################
    # 处理data2
    ####################################################
    data_2 = np.load('E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\mutant\center_CA_PCA_False_neighbor_30.npz')
    x = data_2['x']
    y = data_2['y']
    train_num = x.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # sort row default is chain
    # reshape and one-hot
    y = to_categorical(y)
    # normalization
    x_shape = x.shape
    col_x = train_shape[-1]
    x = x.reshape((-1, col_x))
    x -= mean
    x /= std
    x = x.reshape(x_shape)
    # reshape
    x = x.reshape(x.shape + (1,))

    x_train = np.vstack((x_train,x))
    y_train = np.vstack((y_train,y))
    # print(x_train.shape,y_train.shape)

    return x_train, y_train, x_test, y_test



def Conv2DClassifierIn1(x_train,y_train,x_test,y_test):
        summary = True
        verbose = 1
        CUDA = '0'
        # setHyperParams------------------------------------------------------------------------------------------------
        batch_size = 128
        epoch = {{choice([25,50,75,100,150,200])}}

        kernel_size=(3,3)
        pool_size=(2,2)
        initializer='random_uniform'
        padding_style='same'
        activator='relu'
        l1_regular_rate=0.01
        l2_regular_rate=0.01
        optimizer='adam'
        loss_type='binary_crossentropy'
        metrics=['accuracy']
        # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        # checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
        #                                verbose=1,
        #                                save_best_only=True)
        # my_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                                           patience=5, min_lr=0.0001)
        my_callback = None
        # config TF-----------------------------------------------------------------------------------------------------
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # build --------------------------------------------------------------------------------------------------------
        input_layer = Input(shape=x_train.shape[1:])
        conv1 = layers.Conv2D(16,kernel_size,kernel_initializer=initializer,activation=activator)(input_layer)
        conv2 = layers.Conv2D(32,kernel_size,kernel_initializer=initializer,activation=activator)(conv1)
        pool1 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv2)
        conv3 = layers.Conv2D(64,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=l1_regular_rate,l2=l2_regular_rate))(pool1)
        conv3_BatchNorm = layers.BatchNormalization(axis=-1)(conv3)
        pool2 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv3_BatchNorm)
        conv4 = layers.Conv2D(128,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=l1_regular_rate,l2=l2_regular_rate))(pool2)
        pool3 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv4)
        flat = layers.Flatten()(pool3)

        dense = layers.Dense({{choice([128,256,512,1024])}}, activation=activator)(flat)
        drop1 = layers.Dropout({{uniform(0, 1)}})(dense)
        dense_BatchNorm = layers.BatchNormalization(axis=-1)(drop1)
        drop  = layers.Dropout({{uniform(0, 1)}})(dense_BatchNorm)

        output_layer = layers.Dense(len(np.unique(y_train)),activation='softmax')(drop)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        if summary:
            model.summary()

 # train(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        model.compile(optimizer=optimizer,
                      loss=loss_type,
                      metrics=metrics # accuracy
                      )

        result = model.fit(x=x_train,
                           y=y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=verbose,
                  callbacks=my_callback,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  class_weight=class_weights_dict
                  )
        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = data()
    # Conv2DClassifierIn1(x_train, y_train, x_test, y_test)
    # data()


    best_run, best_model = optim.minimize(model=Conv2DClassifierIn1,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    for trial in Trials():
        print(trial)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)