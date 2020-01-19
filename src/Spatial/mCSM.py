#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, itertools
import numpy as np
from mCNN.processing import read_csv, save_data_array
from scipy.spatial.distance import pdist, squareform

def main():
    parser = argparse.ArgumentParser()
    parser.description = 'A script to calculate mCSM features'
    parser.add_argument('dataset_name')
    parser.add_argument('csv_feature_dir', type=str,                help='eg:~/mCNN/dataset/S1925/feature/mCNN/wild/csv')
    parser.add_argument('-o', '--outdir',  type=str, required=True, help='output dir')
    parser.add_argument('--min',           type=str, required=True, help='The minimum distance from mutant center')
    parser.add_argument('--max',           type=str, required=True, help='The maximum distance from mutant center')
    parser.add_argument('--step',          type=str, required=True, help='The cutoff step')
    parser.add_argument('--center',        type=str, choices=['CA', 'geometric'], default='geometric', help='The MT center type, default is "geometric"')
    parser.add_argument('--class_num',     type=int, choices=[2,8],               default=2,           help='atom classification number, default is 2')
    args = parser.parse_args()

    csv_feature_dir = args.csv_feature_dir
    outdir = args.outdir
    minimum = float(args.min)
    maximum = float(args.max)
    step = float(args.step)
    center = args.center
    print(center)
    class_num = args.class_num

    filename = 'min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s' % (minimum, maximum, step, center, class_num)

    feature_dirlst = [csv_feature_dir + '/' + x + '/center_%s.csv' % center for x in os.listdir(csv_feature_dir)]

    for feature_dir in feature_dirlst:
        assert os.path.exists(feature_dir)

    feature_all = []
    ylst = []
    ddglst = []

    for feature_dir in feature_dirlst:
        df = read_csv(feature_dir)
        ddg = df.loc[:, 'ddg'].values[0]
        # print(type(ddg),ddg)# @@++
        ddglst.append(ddg)
        if ddg >= 0:
            ylst.append(1)
        else:
            ylst.append(0)

        feature_arr = cal_mCSM(df, maximum=maximum, minimum=minimum, step=step, class_num=class_num)

        feature_all.append(feature_arr)

    x = np.array(feature_all)
    ddg = np.array(ddglst).reshape(-1, 1)
    y = np.array(ylst).reshape(-1, 1)
    save_data_array(x, y, ddg, filename, outdir)


def cal_mCSM(df, maximum, minimum, step, class_num=2):
    '''

    :param df: feature df calculated by coord.py
        column name:
            ['chain', 'res', 'het', 'posid', 'inode', 'full_name', 'atom_name', 'secondary', 'dist', 'x', 'y', 'z',
            'occupancy', 'b_factor', 's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C', 's_Helix', 's_Strand', 's_Coil',
            'sa', 'rsa', 'asa', 'phi', 'psi', 'ph', 'temperature', 'C', 'O', 'N', 'Other', 'hydrophobic', 'positive',
            'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur', 'hydrophobic_bak', 'polar', 'C_mass',
            'O_mass', 'N_mass', 'S_mass', 'dC', 'dH', 'dO', 'dN', 'dOther', 'dhydrophobic', 'dpositive', 'dnegative',
            'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur', 'dhydrophobic_bak', 'dpolar', 'dEntropy',
            'entWT', 'entMT', 'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I','WT_L',
            'WT_K', 'WT_M', 'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-', 'MT_A', 'MT_R', 'MT_N',
            'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S',
            'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-', 'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4',
            'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13',
            'atom_pair_constraint', 'angle_constraint', 'dihedral_constraint', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity',
            'ref', 'rama_prepro', 'total', 'ddg']
    :param maximum: maximum length considered by mCSM environment.
    :param minimum: minimum length considered by mCSM environment.
    :param step: cutoff setp.
    :param class_num: atom classification class number, the default is 2 (HP).
    '''
    if class_num == 8:
        atom_class = ['hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur']
        delta_r    = ['dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur']
    if class_num == 2:
        atom_class = ['hydrophobic_bak', 'polar']
        delta_r    = ['dhydrophobic_bak', 'dpolar']
        #########################
        # pay attention here!!! #
        #########################
        df = df.loc[(df.hydrophobic_bak != 0) | (df.polar != 0), :]

    class_num   = len(atom_class)
    combilst    = list(itertools.combinations(list(range(class_num)), 2)) + [(x, x) for x in range(class_num)] #[(0, 1),(),(),...]
    featurelst  = []

    class_arr   = df.loc[df.dist<=maximum,atom_class].values
    coords      = df.loc[df.dist<=maximum,['x','y','z']].values
    delta_r_arr = df.loc[:, delta_r].values[0]

    dist_matrix =  squareform(pdist(coords, metric='euclidean'))
    cutofflist  = list(np.arange(minimum, maximum, step))
    cutofflist.append(maximum)

    ## Cutoff scanning here.
    for cutoff in cutofflist:
        initlst = [0 for _ in range(len(combilst))]
        indices = [list(x) for x in np.argwhere(dist_matrix <= cutoff)] #[[],[],[],...]
        indices = list(filter(lambda x: x[0] > x[1], indices))
        if indices == []:
            featurelst.append(initlst)
            continue

        subfeature_arr = np.array(initlst)
        for indice in indices:
            arr_1, arr_2 = class_arr[indice[0], :], class_arr[indice[1], :] # one-hot coding feature vector(0D array)
            tmplst = []
            for combi in combilst:
                if arr_1[combi[0]] + arr_2[combi[1]] == 2 or arr_1[combi[1]] + arr_2[combi[0]] == 2:
                    tmplst.append(1)
                else:
                    tmplst.append(0)
            subfeature_arr = subfeature_arr+tmplst
        featurelst.append(list(subfeature_arr))


    # print('delta_r_arr shape:', delta_r_arr.shape)
    # print('len featurelst:', len(featurelst))
    # print(np.array(featurelst).shape)
    feature_arr = np.hstack((np.array(featurelst).reshape(-1),delta_r_arr))
    # print(feature_arr.shape)
    return feature_arr


if __name__ == '__main__':
    main()