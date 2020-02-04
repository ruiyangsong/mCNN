#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : processing.py
# time      : 4/6/2019 14:20
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import os, time, functools
import numpy as np
import pandas as pd

aa_321dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
              'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
              'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
              'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}  # from wiki

aa_123dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
              'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
              'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
              'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('\n@call %s()' % func.__name__)
        start = time.time()
        res = func(*args, **kw)
        print('runtime: %f seconds.' % (time.time() - start))
        return res
    return wrapper

def check_qsub(tag,sleep_time,verbose=1):
    jobs = int(shell('qzy | grep %s | wc -l' % tag))
    while jobs > 0:
        time.sleep(sleep_time)
        jobs = int(shell('qzy | grep %s | wc -l' % tag))
    if verbose:
        print('---qsub %s done!'%tag)

def split_tag(dir):
    tag = dir.split('/')[-1]
    if tag == '':
        tag = dir.split('/')[-1]
    return tag

def shell(cmd):
    res=os.popen(cmd).readlines()[0].strip()
    return res

def PDBparser(pdbdir,MDL=0,write=0,outpath=None):
    import warnings
    from Bio.PDB import PDBIO, Select
    from Bio import BiopythonWarning
    from Bio.PDB.PDBParser import PDBParser
    warnings.simplefilter('ignore', BiopythonWarning)
    pdbid = pdbdir.split('/')[-1][0:4]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pdbdir)
    model = structure[MDL]
    if write == 1:
        if outpath == None:
            raise RuntimeError('out path is None!')
        class ModelSelect(Select):
            def accept_model(self, model):
                if model.get_id() == 0:
                    return True
                else:
                    return False

            def accept_chain(self, chain):
                """Overload this to reject chains for output."""
                return 1

            def accept_residue(self, residue):
                if residue.get_resname() in aa_123dict.values():
                    return True
                else:
                    return False

            def accept_atom(self, atom):
                """Overload this to reject atoms for output."""
                return 1
        io = PDBIO()
        io.set_structure(structure)
        io.save('%s/%s.pdb' % (outpath,pdbid), ModelSelect(), preserve_atom_numbering=True)
        # structure_new = parser.get_structure('mdl0', '%s/%s.pdb' % (outpath,pdbid))
        # model = structure_new[MDL]
    return model
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

def read_csv(csvdir):
    f = open(csvdir, 'r')
    df = pd.read_csv(f)
    f.close
    return df

def save_data_array(x,y,ddg_value,filename,outdir):
    if not os.path.exists(outdir):
        os.system('mkdir -p %s'%outdir)
    np.savez('%s/%s.npz' % (outdir,filename), x=x,y=y,ddg=ddg_value)
    print('The 3D array which stored numerical representation has stored at %s.'%outdir)

## function for appending mCSM array
def append_mCSM(x_mCNN, x_mCSM):
    xlst = []
    for i in range(len(x_mCNN)):
        x = x_mCNN[i]
        x_m = x_mCSM[i]
        arr = np.hstack((x, np.dot(np.ones((x.shape[0], 1)), x_m.reshape(1, -1))))
        xlst.append(arr)
    return np.array(xlst)

def load_data(dir):
    data = np.load(dir)
    x = data['x']
    y = data['y']
    ddg = data['ddg']
    return x,y,ddg

def calc_coor_pValue(feature_a_list, feature_b_list):
    import scipy.stats as stats
    pearson_coeff, p_value = stats.pearsonr(np.array(feature_a_list).reshape(-1), np.array(feature_b_list).reshape(-1))
    return pearson_coeff, p_value

def transform(coord_array_before, center_coord):
    from sklearn.decomposition import PCA
    assert len(coord_array_before) >= 3  # row number.
    pca_model = PCA(n_components=3)
    pca_model.fit(coord_array_before)
    coord_array_after = pca_model.transform(coord_array_before)
    center_coord_after = pca_model.transform(center_coord.reshape(-1, 3))
    coord_array_after = coord_array_after - center_coord_after
    return coord_array_after

def sort_row(x, method = 'chain', p_seed = 1):
    '''
    :param x: 3D tensor of this dataset, the axis are: data_num, row_num and col_nm.
    :param method: str, row sorting method.
    :return: 3D tensor after sort.
    '''
    if method == 'chain':
        return x
    data_num, row_num, col_num = x.shape
    if method == 'distance':
        for i in range(data_num):
            indices = x[i,:,0].argsort()
            x[i] = x[i,[indices]]
        return x
    elif method == 'octant':
        x_new = np.zeros(x.shape)
        for i in range(x.shape[0]):
            data = pd.DataFrame(x[i])
            octant1 = data[(data[1] >= 0) & (data[2] >= 0) & (data[3] >= 0)]
            octant2 = data[(data[1] < 0) & (data[2] > 0) & (data[3] > 0)]
            octant3 = data[(data[1] < 0) & (data[2] < 0) & (data[3] > 0)]
            octant4 = data[(data[1] > 0) & (data[2] < 0) & (data[3] > 0)]
            octant5 = data[(data[1] > 0) & (data[2] > 0) & (data[3] < 0)]
            octant6 = data[(data[1] < 0) & (data[2] > 0) & (data[3] < 0)]
            octant7 = data[(data[1] < 0) & (data[2] < 0) & (data[3] < 0)]
            octant8 = data[(data[1] > 0) & (data[2] < 0) & (data[3] < 0)]
            temp_array = np.vstack((octant1, octant2, octant3, octant4, octant5, octant6, octant7, octant8))
            x_new[i] = temp_array
        return x_new
    elif method == 'permutation1':
        indices = np.load('../global/permutation1/indices_%d.npy' % row_num)
    elif method == 'permutation2':
        indices = np.load('../global/permutation2/indices_%d.npy' % row_num)
    elif method == 'permutation':
        indices = [i for i in range(row_num)]
        np.random.seed(p_seed)
        np.random.shuffle(indices)
    for i in range(data_num):
        x[i] = x[i][indices]
    return x

def shuffle_data(x, y, ddg, random_seed):
    indices = [i for i in range(x.shape[0])]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    ddg = ddg[indices]
    return x,y,ddg

def split_val(x_train, y_train, ddg_train, ddg_test, random_seed):
    # print(ddg_train.shape)
    ddg_train, ddg_test = ddg_train.reshape(-1), ddg_test.reshape(-1)
    p_train_indices, n_train_indices = ddg_train >= 0, ddg_train < 0
    x_p_train, x_n_train = x_train[p_train_indices], x_train[n_train_indices]
    y_p_train, y_n_train = y_train[p_train_indices], y_train[n_train_indices]
    ddg_p_train, ddg_n_train = ddg_train[p_train_indices], ddg_train[n_train_indices]

    num_p_test, num_n_test = sum(ddg_test >= 0), sum(ddg_test < 0)

    x_p_val, x_n_val = x_p_train[:num_p_test], x_n_train[:num_n_test]
    y_p_val, y_n_val = y_p_train[:num_p_test], y_n_train[:num_n_test]
    ddg_p_val, ddg_n_val = ddg_p_train[:num_p_test], ddg_n_train[:num_n_test]

    x_p_train, x_n_train = x_p_train[num_p_test:], x_n_train[num_n_test:]
    y_p_train, y_n_train = y_p_train[num_p_test:], y_n_train[num_n_test:]
    ddg_p_train, ddg_n_train = ddg_p_train[num_p_test:], ddg_n_train[num_n_test:]

    x_val, y_val, ddg_val = np.vstack((x_p_val, x_n_val)), np.vstack((y_p_val, y_n_val)), np.hstack((ddg_p_val, ddg_n_val))
    x_train_new, y_train_new, ddg_train_new = np.vstack((x_p_train,x_n_train)), np.vstack((y_p_train,y_n_train)),\
                                              np.hstack((ddg_p_train,ddg_n_train))
    ## shuffe data.
    x_train_new, y_train_new, ddg_train_new = shuffle_data(x_train_new, y_train_new, ddg_train_new, random_seed=random_seed)

    assert x_train_new.shape[0] + x_val.shape[0] == x_train.shape[0]
    assert x_val.shape[0] == ddg_test.shape[0]

    return x_train_new, y_train_new, ddg_train_new, x_val, y_val, ddg_val

def oversampling(x_train, y_train):
    from imblearn.over_sampling import RandomOverSampler
    train_shape = x_train.shape
    train_num,train_col = train_shape[0], train_shape[-1]
    x_train = x_train.reshape(train_num, -1)
    y_train = y_train.reshape(train_num)

    ros = RandomOverSampler(random_state=10)
    x_train_new, y_train_new = ros.fit_sample(x_train, y_train)
    if len(train_shape) == 3:
        x_train = x_train_new.reshape(-1,train_shape[1],train_col)
    else:
        x_train = x_train_new
    y_train = y_train_new.reshape(-1,1)
    positive_indices, negative_indices = y_train.reshape(-1, ) == 1, y_train.reshape(-1, ) == 0
    assert x_train[positive_indices].shape[0] == x_train[negative_indices].shape[0]
    assert x_train[positive_indices].shape[0] == x_train[negative_indices].shape[0]
    return x_train, y_train

def normalize(x_train, x_test, x_val, val_flag = 1, method = 'norm'):
    train_shape, test_shape = x_train.shape, x_test.shape
    col_train = x_train.shape[-1]
    col_test  = x_test.shape[-1]
    x_train   = x_train.reshape((-1, col_train))
    x_test    = x_test.reshape((-1, col_test))

    if val_flag == 1:
        val_shape = x_val.shape
        col_val   = x_val.shape[-1]
        x_val     = x_val.reshape((-1, col_val))

    if method == 'norm':
        mean = x_train.mean(axis=0)
        std  = x_train.std(axis=0)
        std[np.argwhere(std==0)] = 0.01
        x_train -= mean
        x_train /= std
        x_test  -= mean
        x_test  /= std
        if val_flag == 1:
            x_val -= mean
            x_val /= std
    elif method == 'max':
        max_ = x_train.max(axis=0)
        max_[np.argwhere(max_ == 0)] = 0.01
        x_train /= max_
        x_test  /= max_
        if val_flag == 1:
            x_val /= max_

    x_train = x_train.reshape(train_shape)
    x_test  = x_test.reshape(test_shape)

    if val_flag == 1:
        x_val = x_val.reshape(val_shape)
        return x_train, x_test, x_val
    elif val_flag == 0:
        return x_train, x_test

def reshape_tensor(x_):
    ## reshape array to Input shape
    # data_num, row_num, col_num = x_.shape
    # x_ = x_.reshape(data_num, row_num, col_num, 1)
    x_ = x_[...,np.newaxis]
    return x_

def split_delta_r(x_train):
    x_train, delta_r_train = x_train[:, :, :99], x_train[:, 0, 99:]
    x_train = x_train[:, :, :, np.newaxis]
    return x_train, delta_r_train

def save_model(dataset_name, radius, k_neighbor, class_num, dist,network,test_acc,k_count,acc_threshold=0.86):
    ## Create model dir.
    path_k_neighbor = '../models/' + dataset_name + '/k_neighbor/'
    path_radius = '../models/' + dataset_name + '/radius/'
    if not os.path.exists(path_k_neighbor):
        os.mkdir(path_k_neighbor)
    if not os.path.exists(path_radius):
        os.mkdir(path_radius)
    ##保存模型
    if test_acc >= acc_threshold:
        if dist == 1:
            #将模型存入dist文件夹
            network.save('../models/%s/dist/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))
        elif k_neighbor != 0:
            #将模型存入k_neighbor文件夹
            network.save('../models/%s/k_neighbor/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))
        else:
            #将模型存入radius文件夹
            network.save('../models/%s/radius/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))

def print_result(nn_model, kfold_score):
    print('+'*5, 'The average test results are showed below:')
    if nn_model < 2:
        print('--acc:', np.mean(kfold_score[:, 0]))
        print('--recall_p:', np.mean(kfold_score[:, 1]))
        print('--recall_n:', np.mean(kfold_score[:, 2]))
        print('--precision_p:', np.mean(kfold_score[:, 3]))
        print('--precision_n:', np.mean(kfold_score[:, 4]))
        print('--mcc:', np.mean(kfold_score[:, 5]))

    elif nn_model > 2:
        print('--rho:', np.mean(kfold_score[:, 0]))
        print('--rmse:', np.mean(kfold_score[:, 1]))

def plotfigure(history_dict):
    import matplotlib.pyplot as plt
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = np.load('../datasets_array/S1925/k_neighbor/S1925_r_50.00_neighbor_50_class_5.npz')
    x = data['x']
    print('x_shape:',x.shape)
    # print(x[0,0:5,:])
