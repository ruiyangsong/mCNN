from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform

import os, sys
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras import Input, models, layers, regularizers, callbacks, optimizers
from keras.utils import to_categorical


def data():
    random_seed = 10
    # data = np.load('/public/home/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_30.npz')
    data = np.load('/dl/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_30.npz')
    # data = np.load('E:/projects/mCNN/yanglab/mCNN-master/dataset/S2648/mCNN/wild/center_CA_PCA_False_neighbor_30.npz')
    x = data['x']
    y = data['y']
    ddg = data['ddg'].reshape(-1)
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
    return x_train, y_train, x_test, y_test



def Conv2DClassifierIn1(x_train,y_train,x_test,y_test):
        summary = True
        verbose = 1

        # setHyperParams------------------------------------------------------------------------------------------------
        batch_size = {{choice([32,64,128,256,512])}}
        epoch = {{choice([25,50,75,100,125,150,175,200])}}

        conv_block={{choice(['two', 'three'])}}
        res_block={{choice(['two','one','zero'])}}

        conv1_num={{choice([8, 16, 32, 64])}}
        conv2_num={{choice([16,32,64,128])}}
        conv3_num={{choice([32,64,128])}}

        dense1_num={{choice([128, 256, 512])}}
        dense2_num={{choice([64, 128, 256])}}

        l1_regular_rate = {{uniform(0.00001, 1)}}
        l2_regular_rate = {{uniform(0.000001, 1)}}
        drop1_num={{uniform(0.1, 1)}}
        drop2_num={{uniform(0.0001, 1)}}

        activator={{choice(['elu','relu','tanh'])}}
        optimizer={{choice(['adam','rmsprop','SGD'])}}

        #---------------------------------------------------------------------------------------------------------------
        kernel_size = (3, 3)
        pool_size = (2, 2)
        initializer = 'random_uniform'
        padding_style = 'same'
        loss_type='binary_crossentropy'
        metrics=['accuracy']
        my_callback = None
        # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        # checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
        #                                verbose=1,
        #                                save_best_only=True)
        # my_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                                           patience=5, min_lr=0.0001)



        # build --------------------------------------------------------------------------------------------------------
        x = Input(shape=x_train.shape[1:])
        y = layers.Conv2D(conv1_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(x)
        y = layers.Conv2D(conv1_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.MaxPooling2D(pool_size,padding=padding_style)(y)
        if res_block == 'one':
            residual = layers.Conv2D(conv1_num, 1, strides=2, padding='same')(x)
            y = layers.add([y, residual])
        if conv_block == 'two':
            y = layers.Conv2D(conv2_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.Conv2D(conv2_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.MaxPooling2D(pool_size,padding=padding_style)(y)
            if res_block == 'two':
                residual = layers.Conv2D(conv2_num, 1, strides=4, padding='same')(x)
                y = layers.add([y, residual])
        elif conv_block == 'three':
            y = layers.Conv2D(conv2_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.Conv2D(conv2_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.MaxPooling2D(pool_size,padding=padding_style)(y)
            if res_block == 'two':
                residual = layers.Conv2D(conv2_num, 1, strides=4, padding='same')(x)
                y = layers.add([y, residual])

            y = layers.Conv2D(conv3_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.MaxPooling2D(pool_size,padding=padding_style)(y)


        flat = layers.Flatten()(y)
        drop = layers.Dropout(drop1_num)(flat)

        dense = layers.Dense(dense1_num, activation=activator, kernel_regularizer=regularizers.l1_l2(l1=l1_regular_rate,l2=l2_regular_rate))(drop)
        BatchNorm = layers.BatchNormalization(axis=-1)(dense)
        drop  = layers.Dropout(drop2_num)(BatchNorm)

        dense = layers.Dense(dense2_num, activation=activator, kernel_regularizer=regularizers.l1_l2(l1=l1_regular_rate,l2=l2_regular_rate))(drop)

        output_layer = layers.Dense(len(np.unique(y_train)),activation='softmax')(dense)

        model = models.Model(inputs=x, outputs=output_layer)

        if summary:
            model.summary()

 # train(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        model.compile(optimizer=optimizer,
                      loss=loss_type,
                      metrics=metrics # accuracy
                      )

        K.set_session(tf.Session(graph=model.output.graph))
        init = K.tf.global_variables_initializer()
        K.get_session().run(init)

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
    # config TF-----------------------------------------------------------------------------------------------------
    CUDA, max_eval, algo_flag = sys.argv[1:]
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5),)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    # x_train, y_train, x_test, y_test = data()
    # Conv2DClassifierIn1(x_train, y_train, x_test, y_test)

    if algo_flag == 'tpe':
        algo=tpe.suggest
    elif algo_flag == 'rand':
        algo=rand.suggest

    best_run, best_model = optim.minimize(model=Conv2DClassifierIn1,
                                          data=data,
                                          eval_space=True,
                                          algo=algo,
                                          max_evals=int(max_eval),
                                          keep_temp=False,
                                          verbose=False,
                                          trials=Trials())
    for trial in Trials():
        print(trial)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
