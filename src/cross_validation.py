#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : cross_validation.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import argparse
from sklearn.model_selection import StratifiedKFold
from processing import *
from train_model import train_model
from test_model import test_model

def cross_validation(x, y, ddg, k, nn_model, normalize_method, random_seed, flag,oversample, CUDA, epoch, batch_size, train_ratio=0.7):
    '''
    :param x: 2D or 3D numpy array, stored numerical representation of this dataset.
    :param y: 1D numpy array, labels of x.
    :param ddg: 1D numpy array, ddg array.
    :param k: int, fold number.
           k == 0 --> when giving blind data manually.
           k == 1 --> blind test.
           k == 2 --> when giving k_fold data manually.
           k >= 3 --> kfold cross validation.
    :param nn_model: float, which network structure to choose.
           nn_model == 1.xx --> classification task.
           nn_model == 2.xx --> regression task.
    :param normalize_method: str, 'max' or 'norm'.
    :param random_seed: tuple, random seed for shuffle k_fold and split_val.
    :param flag: tuple, flag for val_data and verbose.
    :param train_ratio: float, split ratio for blind test. 0.7 is the default option.
    :return: label balanced train data and test data.
    '''
    k_seed, v_seed = random_seed
    kfold_score = np.zeros((k, 6))
    history_list = []
    ## giving blind data manually.
    if k == 0:
        pass
    ## blind test.
    if k == 1:
        positive_indices, negative_indices = ddg >= 0, ddg < 0
        x_positive, x_negative = x[positive_indices], x[negative_indices]
        y_positive, y_negative = y[positive_indices], y[negative_indices]
        ddg_positive, ddg_negative = ddg[positive_indices], ddg[negative_indices]
        left_positive, left_negative = round(0.7 * x_positive.shape[0]), round(0.7 * x_negative.shape[0])
        x_train, x_test = np.vstack((x_positive[:left_positive], x_negative[:left_negative])), np.vstack(
            (x_positive[left_positive:], x_negative[left_negative:]))
        y_train, y_test = np.vstack((y_positive[:left_positive], y_negative[:left_negative])), np.vstack(
            (y_positive[left_positive:], y_negative[left_negative:]))
        ddg_train, ddg_test = np.vstack((ddg_positive[:left_positive], ddg_negative[:left_negative])), np.vstack(
            (ddg_positive[left_positive:], ddg_negative[left_negative:]))
        x_train, y_train, ddg_train = shuffle_data(x_train, y_train, ddg_train, v_seed)
        x_test, y_test, ddg_test = shuffle_data(x_test, y_test, ddg_test, v_seed)
        ## train model.
    ## giving k_fold data manually.
    if k == 2:
        pass
    ## k_fold cross validation.
    if k >= 3:
        k_count = 1
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = k_seed)
        for train_index, test_index in skf.split(x, y):
            print('%d-th fold is in progress.' % (k_count))
            x_train, y_train, ddg_train, x_test, y_test, ddg_test = x[train_index], y[train_index], ddg[train_index], x[test_index], y[test_index], ddg[test_index]
            ## train model on each fold.
            network, history_dict, x_test, y_test, ddg_test = train_model(x_train, y_train, ddg_train, x_test, y_test, ddg_test, nn_model, normalize_method, v_seed, flag, oversample, CUDA, epoch, batch_size)
            history_list.append(history_dict)
            ## test model on each fold.
            if nn_model < 2:
                # classification model
                acc, recall_p, recall_n, precision_p, precision_n, mcc = test_model(network, x_test, y_test, ddg_test, nn_model)
                kfold_score[k_count - 1, :] = [acc, recall_p, recall_n, precision_p, precision_n, mcc]
            elif nn_model > 2:
                # regression model
                pearson_coeff, rmse = test_model(network, x_test, y_test, ddg_test, nn_model)
                kfold_score[k_count - 1, :2] = [pearson_coeff, rmse]
            k_count += 1

    return kfold_score, history_list

if __name__ == '__main__':
    ## Init container
    container = {'mCNN_arrdir':'', 'mCSM_arrdir':''}
    ### Input parameters.
    parser = argparse.ArgumentParser()
    ## parameters for reading feature array
    parser.add_argument('dataset_name',        type=str,                                              help='dataset_name.')
    parser.add_argument('-C', '--center',      type=str, default='CA', choices=['CA', 'geometric'],   help='The MT site center, default is CA.')
    parser.add_argument('-T', '--PCA',         type=str, default='False',                             help='If PCA, default is False')
    parser.add_argument('-k', '--k_neighbor',  type=int,                                              help='First k neighbors around Alpha-C atom at the mutant site.')
    parser.add_argument('-r', '--radius',      type=float,                                            help='The neighborhood radius.')
    ## If consider feature mCSM
    parser.add_argument('--mCSM',              type=float,   nargs=4,                                 help='min, max, step, atom_class_num.')
    ## Config data processing
    parser.add_argument('-n', '--normalize',   type=str,     choices=['norm','max'], default='norm',  help='normalize_method to choose, default = norm.')
    parser.add_argument('-s', '--sort',        type=str,     choices=['chain','distance','octant','permutation','permutation1','permutation2'], default='chain',help='row sorting methods to choose, default = chain.')
    parser.add_argument('-d', '--random_seed', type=int,     nargs=3,                default=(1,1,1), help='permutation-seed, k-fold-seed, split-val-seed, default sets to (1,1,1).')
    parser.add_argument('-O', '--oversample',  type=str,                             default='True',  help='if consider oversampling, default = True.')
    ## Config training
    parser.add_argument('-D', '--model',       type=float,                                            help='Network model to chose.', required=True)
    parser.add_argument('-K', '--Kfold',       type=int,                                              help='Fold numbers to cross validation.', required=True)
    parser.add_argument('-V', '--val_ver',     type=int, nargs=2,    choices=[0, 1], default=(1, 1),  help='if split val and the verbose flag, default sets to (1,1).')
    parser.add_argument('-E', '--epoch',       type=int, default=10,                                  help='training epoch, default is 10.')
    parser.add_argument('-B', '--batch_size',  type=int, default=64,                                  help='training batch size, default is 64.')
    ## config hardware
    parser.add_argument('--CUDA',              type=str, default='0', choices=['0','1','2','3'],      help='Which gpu to use, default = "0"')
    ## parser
    args = parser.parse_args()
    dataset_name = args.dataset_name
    if args.center:
        center = args.center
    if args.PCA:
        pca = str2bool(args.PCA)
    if args.k_neighbor:
        k_neighbor = args.k_neighbor
        container['mCNN_arrdir'] = '../datasets_array/%s/k_neighbor/%s_center_%s_PCA_%s_neighbor_%s.npz' %(dataset_name,dataset_name,center,pca,k_neighbor)
    else:
        k_neighbor = 0
    if args.radius:
        radius = args.radius
        container['mCNN_arrdir'] = '../datasets_array/%s/radius/%s_center_%s_PCA_%s_radius_%s.npz' %(dataset_name, dataset_name, center, pca, radius)
    else:
        radius = 0
    ## parser mCSM
    if args.mCSM:
        min_, max_, step, atom_class_num = args.mCSM
        if step in [1.0,2.0]:
            step = int(step)
        container['mCSM_arrdir'] = '../datasets_array/%s/mCSM/%s/%s_min_%s_max_%s_step_%s_center_%s_class_%s/%s_min_%s_max_%s_step_%s_center_%s_class_%s.npz'\
                                   %(dataset_name, center,dataset_name,min_,int(max_),step,center,int(atom_class_num),dataset_name,min_,max_,float(step),center,int(atom_class_num))
    else:
        atom_class_num = 5
    ## parser data processing
    if args.normalize:
        normalize_method = args.normalize
    if args.sort:
        sort_method = args.sort
    if args.random_seed:
        seed_tuple = tuple(args.random_seed)
    if args.oversample:
        oversample = str2bool(args.oversample)
    ## parser training
    if args.model:
        nn_model = args.model
    if args.Kfold:
        k = args.Kfold
    if args.val_ver:
        flag_tuple = tuple(args.val_ver)
    if args.epoch:
        epoch = args.epoch
    if args.batch_size:
        batch_size = args.batch_size
    if args.CUDA:
        CUDA = args.CUDA
    ## print input info.
    print('dataset_name: %s, radius: %.2f, k_neighbor: %d, class_num: %d, k-fold: %d, nn_model: %.2f,'
          '\nnormalize_method: %s,'
          '\nsort_method: %s,'
          '\n(permutation-seed, k-fold-seed, split-val-seed): %r,'
          '\n(val_flag, verbose_flag): %r.'
          '\noversample: %r'
          '\nCUDA: %r'
          '\nepoch: %r'
          '\nbatch_size: %r'
          %(dataset_name, radius, k_neighbor, atom_class_num, k, nn_model, normalize_method, sort_method, seed_tuple, flag_tuple, oversample, CUDA, epoch, batch_size))

    ############################################################
    ##                main program begins here                ##
    ############################################################

    ## load data
    if container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] == '':
        x, y, ddg = load_data(container['mCNN_arrdir'])
    if container['mCSM_arrdir'] != '' and container['mCNN_arrdir'] == '':
        x, y, ddg = load_data(container['mCSM_arrdir'])
    if container['mCNN_arrdir'] != '' and container['mCSM_arrdir'] != '':
        x, y, ddg = load_data(container['mCNN_arrdir'])
        x_mCSM, y_mCSM, ddg_mCSM = load_data(container['mCSM_arrdir'])        
        x = append_mCSM(x_mCNN=x,x_mCSM=x_mCSM)
    print(x.shape)
    print('Loading data from hard drive is done.')

    ## sort row of each mutation matrix.
    x = sort_row(x, sort_method, seed_tuple[0]) # chain sorting return self!
    print('Sort row is done, sorting method is %s.' % sort_method)

    ## Cross validation.
    print('%d-fold cross validation begin.' % (k))
    kfold_score, history_list = cross_validation(x, y, ddg, k, nn_model, normalize_method, seed_tuple[1:], flag_tuple,oversample, CUDA, epoch, batch_size, train_ratio=0.7)

    print_result(nn_model,kfold_score)
    ## plot.
    #plotfigure(history_dict)
