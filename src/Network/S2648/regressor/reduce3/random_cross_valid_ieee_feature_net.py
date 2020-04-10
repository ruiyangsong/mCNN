import os, json
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks
from mCNN.Network.metrics import test_report_reg,pearson_r
from keras.utils import to_categorical

'This network is optimized by S2648 [0.2 vaildation|0.8 train]. suppose that we have neighbor 120'

def data(x,y,ddg,train_index,test_index):
    ## train data
    x_train = x[train_index]
    y_train = y[train_index]
    ddg_train = ddg[train_index].reshape(-1)

    # erase rosetta energy[0:41 \union 58:end]
    idx_list = [0,1,2,3,17,21,22,23,24,25,26,100,101,102,103,104]

    # print(idx_list)
    x_train = x_train[:,:,idx_list]
    print(x_train.shape)

    ## valid data
    random_seed = 10
    train_num = x_train.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    ddg_train = ddg_train[indices]
    positive_indices, negative_indices = ddg_train >= 0, ddg_train < 0
    x_positive, x_negative = x_train[positive_indices], x_train[negative_indices]
    y_positive, y_negative = y_train[positive_indices], y_train[negative_indices]
    ddg_positive, ddg_negative = ddg_train[positive_indices], ddg_train[negative_indices]

    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train = np.vstack((x_positive[:left_positive], x_negative[:left_negative]))
    x_val = np.vstack((x_positive[left_positive:], x_negative[left_negative:]))
    y_train = np.vstack((y_positive[:left_positive], y_negative[:left_negative]))
    y_val = np.vstack((y_positive[left_positive:], y_negative[left_negative:]))
    ddg_train = np.hstack((ddg_positive[:left_positive], ddg_negative[:left_negative]))
    ddg_val = np.hstack((ddg_positive[left_positive:], ddg_negative[left_negative:]))

    ## test data
    x_test = x[test_index]
    y_test = y[test_index]
    ddg_test = ddg[test_index].reshape(-1)

    # erase rosetta energy[0:41 \union 58:end]

    x_test = x_test[:, :, idx_list]
    print(x_test.shape)

    # sort row default is chain, pass
    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    # normalization
    train_shape = x_train.shape
    test_shape = x_test.shape
    val_shape = x_val.shape
    col_train = train_shape[-1]
    col_test = test_shape[-1]
    col_val = val_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_test = x_test.reshape((-1, col_test))
    x_val = x_val.reshape((-1,col_val))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_val -= mean
    x_val /= std
    x_train = x_train.reshape(train_shape)
    x_test = x_test.reshape(test_shape)
    x_val = x_val.reshape(val_shape)

    # reshape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val

def ieee_net(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val):
    row_num, col_num = x_train.shape[1:3]
    network = models.Sequential()
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
                    metrics=['mae'])  # mae平均绝对误差（mean absolute error） accuracy
    history = network.fit(
        x_train, ddg_train, validation_data=(x_val, ddg_val),
        epochs=100, batch_size=64, verbose=0)
    history_dict = history.history
    return network, history_dict



if __name__ == '__main__':
    CUDA,CUDA_rate = '3','0.01'
    ## config TF
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate)<0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))
    score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}
    data_basedir = '/dl/sry/mCNN/dataset/S2648/npz/wild/cross_valid'

    # data_pth = '/dl/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_50.npz'
    data_pth = '/dl/sry/mCNN/dataset/S2648/npz/wild/all/all_neighbor_120.npz'
    Data = np.load(data_pth)
    x, y, ddg = Data['x'], Data['y'], Data['ddg']
    k_count = 1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(x, y):
        print('%d-th fold is in progress.' % (k_count))
        k_count+=1
        x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val = data(x, y, ddg, train_index, test_index)
        model, history_dict = ieee_net(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val)

        pearson_coeff, std, mae = test_report_reg(model, x_test, ddg_test)
        print('\n----------Predict:'
              '\npearson_coeff: %s, std: %s, mae: %s'
              % (pearson_coeff, std, mae))
        score_dict['pearson_coeff'].append(pearson_coeff)
        score_dict['std'].append(std)
        score_dict['mae'].append(mae)
    #print average
    with open('/dl/sry/mCNN/dataset/S2648/saved_model/wild/cross_valid/regressor_avg_scores.json', 'w') as file:
        file.write(json.dumps(score_dict))
    print('\nAVG results','-'*10)
    for key in score_dict.keys():
        print('*avg(%s): %s'%(key,np.mean(score_dict[key])))


