import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import models
import numpy as np
import scipy.stats as stats
from mCNN.queueGPU import queueGPU
'''
@current_hyper_tag	140_4_32_32_32_64_0.5	
@current	optmized_loss	1.096142613
'''
def main():
    config_tf()
    train_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold1_train_center_CA_PCA_False_neighbor_140.npz'
    val_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold1_valid_center_CA_PCA_False_neighbor_140.npz'
    test_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_foldall_test_center_CA_PCA_False_neighbor_140.npz'
    ## select by large pcc
    model_pth_lst =  [
        [
            '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/opt_all_simpleNet_v4/fine_tuning/140_4_32_32_32_64_0.5/fold_%s_model.json'%x,
            '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/opt_all_simpleNet_v4/fine_tuning/140_4_32_32_32_64_0.5/fold_%s_weightsFinal.h5'%x
        ] for x in [1,2,3,4,5,6,7,8,9,10]
    ]

    print(len(model_pth_lst))
    model_performance_lst = [1] * len(model_pth_lst)

    x_train, ddg_train, x_test, ddg_test = data(train_data_pth, test_data_pth, val_data_pth)
    model_lst = model_loader(model_pth_lst=model_pth_lst)
    model_weight_lst = [x / sum(model_performance_lst) for x in model_performance_lst]

    train_pcc, train_rmse, train_mae = test_ensemble_model(
        model_bin_lst=model_lst,
        model_weight_lst=model_weight_lst,
        x=x_train,
        label=ddg_train)
    print('\n--Predict train_data:'
          '\npcc: %s'
          '\nrmse: %s'
          '\nmae: %s'
          % (train_pcc, train_rmse, train_mae))

    test_pcc, test_rmse, test_mae = test_ensemble_model(
        model_bin_lst=model_lst,
        model_weight_lst=model_weight_lst,
        x=x_test,
        label=ddg_test)
    print('\n--Predict test_data:'
          '\npcc: %s'
          '\nrmse: %s'
          '\nmae: %s'
          % (test_pcc, test_rmse, test_mae))

def config_tf():
    CUDA_rate = '0.2'
    ## config TF
    queueGPU(USER_MEM=3000, INTERVAL=60, Verbose=1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate) < 0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

def data(train_data_pth,test_data_pth, val_data_pth):
    neighbor = 140
    idx = [x for x in range(158) if x not in [24, 26] + [x for x in range(27, 37)] + [x for x in range(40, 46)]]# 去除无用特征+二级结构信息
    ## train data
    train_data = np.load(train_data_pth)
    x_train = train_data['x']
    y_train = train_data['y']
    ddg_train = train_data['ddg'].reshape(-1)

    # select kneighbor atoms
    x_train_kneighbor_lst = []
    for sample in x_train:
        dist_arr = sample[:, 0]
        indices = sorted(dist_arr.argsort()[:neighbor])
        x_train_kneighbor_lst.append(sample[indices, :])
    x_train = np.array(x_train_kneighbor_lst)
    # rechose feature
    x_train = x_train[:, :, idx]


    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    # class_weights_dict = dict(enumerate(class_weights))
    ## valid data
    val_data = np.load(val_data_pth)
    x_val = val_data['x']
    y_val = val_data['y']
    ddg_val = val_data['ddg'].reshape(-1)
    # select kneighbor atoms
    x_val_kneighbor_lst = []
    for sample in x_val:
        dist_arr = sample[:, 0]
        indices = sorted(dist_arr.argsort()[:neighbor])
        x_val_kneighbor_lst.append(sample[indices, :])
    x_val = np.array(x_val_kneighbor_lst)
    # rechose feature
    x_val = x_val[:, :, idx]

    x_train = np.vstack((x_train, x_val))
    ddg_train = np.hstack((ddg_train, ddg_val))

    ## test data
    test_data = np.load(test_data_pth)
    x_test = test_data['x']
    ddg_test = test_data['ddg'].reshape(-1)
    # select kneighbor atoms
    x_test_kneighbor_lst = []
    for sample in x_test:
        dist_arr = sample[:, 0]
        indices = sorted(dist_arr.argsort()[:neighbor])
        x_test_kneighbor_lst.append(sample[indices, :])
    x_test = np.array(x_test_kneighbor_lst)
    # rechose feature
    x_test = x_test[:, :, idx]


    # sort row default is chain, pass
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
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, ddg_train, x_test, ddg_test

def test_ensemble_model(model_bin_lst, model_weight_lst, x, label):
    pred_lst = []
    for model in model_bin_lst:
        pred_lst.append(model.predict(x, batch_size=32, verbose=0).reshape(-1))
    pred = np.sum(np.array(pred_lst).T * model_weight_lst, axis=1)
    pcc, rmse, mae = test_score(real=label, pred=pred)
    return pcc, rmse, mae

def model_loader(model_pth_lst):
    model_lst = []
    for sublst in model_pth_lst:
        model_structure_pth = sublst[0]
        model_weights_pth   = sublst[1]
        with open(model_structure_pth, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
        loaded_model.load_weights(filepath=model_weights_pth)
        model_lst.append(loaded_model)
    return model_lst

def test_score(real, pred):
    try:
        pcc, p_value = stats.pearsonr(real, pred)
    except:
        print('\nCalc pcc err'
              'ddg_test: %s\nddg_pred: %s\n' % (real, pred))
        exit()
    # rmse = np.sqrt(np.sum((ddg_test - ddg_pred) ** 2) / (len(ddg_test) - 2))
    rmse = np.sqrt(np.sum((real - pred) ** 2) / len(real))
    mae =np.sum(np.abs(real - pred))/len(real)
    return pcc, rmse, mae

if __name__ == '__main__':
    main()