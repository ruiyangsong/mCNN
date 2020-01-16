#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Convert mCNN features(csv) to numpy array (at dataset level).
All the features we have were listed below, from left to right:
------------------------------------------------------------------------------------------------------------------------
dist,x,y,z,occupancy,b_factor,s_H,s_G,s_I,s_E,s_B,s_T,s_C,s_Helix,s_Strand,s_Coil,sa,rsa,asa,phi,psi,ph,temperature,
C,O,N,Other,hydrophobic,positive,negative,neutral,acceptor,donor,aromatic,sulphur,hydrophobic_bak,polar,C_mass,O_mass,
N_mass,S_mass,dC,dH,dO,dN,dOther,dhydrophobic,dpositive,dnegative,dneutral,dacceptor,ddonor,daromatic,dsulphur,
dhydrophobic_bak,dpolar,dEntropy,entWT,entMT,WT_A,WT_R,WT_N,WT_D,WT_C,WT_Q,WT_E,WT_G,WT_H,WT_I,WT_L,WT_K,WT_M,WT_F,WT_P,
WT_S,WT_T,WT_W,WT_Y,WT_V,WT_-,MT_A,MT_R,MT_N,MT_D,MT_C,MT_Q,MT_E,MT_G,MT_H,MT_I,MT_L,MT_K,MT_M,MT_F,MT_P,MT_S,MT_T,MT_W,
MT_Y,MT_V,MT_-,fa_atr,fa_rep,fa_sol,fa_intra_rep,fa_intra_sol_xover4,lk_ball_wtd,fa_elec,pro_close,hbond_sr_bb,
hbond_lr_bb,hbond_bb_sc,hbond_sc,dslf_fa13,atom_pair_constraint,angle_constraint,dihedral_constraint,omega,fa_dun,
p_aa_pp,yhh_planarity,ref,rama_prepro,total,ddg
------------------------------------------------------------------------------------------------------------------------
'''
import os, argparse
import numpy as np
from processing import read_csv, str2bool, save_data_array, transform

def main():
    ## The concerned features
    keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor',

            's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
            's_Helix', 's_Strand', 's_Coil',

            'sa', 'rsa', 'asa', 'phi', 'psi',

            'ph', 'temperature',

            'C', 'O', 'N', 'Other',

            'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',

            'hydrophobic_bak', 'polar',

            'C_mass', 'O_mass', 'N_mass', 'S_mass',

            'dC', 'dH', 'dO', 'dN', 'dOther',

            'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',

            'dhydrophobic_bak', 'dpolar',

            'dEntropy', 'entWT', 'entMT',

            'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M',
            'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
            'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
            'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',

            'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
            'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'atom_pair_constraint',
            'angle_constraint', 'dihedral_constraint', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref',
            'rama_prepro', 'total']

    ## parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name',       type=str, help='dataset_name')
    # parser.add_argument('-r', '--radius',     type=float  nargs='+',      help='All the radius, separated with space')
    parser.add_argument('-k', '--k_neighbor', type=int, nargs='+',       required=True, help='All the k_neighbors, separated with space')
    parser.add_argument('-C', '--center',     type=str, nargs='+',       required=True, choices=['CA','geometric'], help='The MT site center type list, separated with space')
    parser.add_argument('-T', '--pca',        type=str, default='False', choices=['False','True'], help='If consider pca transform, the default is False')
    # parser.add_argument('--class_num',        type=int,  choices=[2,8],   help='atom classification scheme')
    args = parser.parse_args()
    dataset_name  = args.dataset_name
    k_neighborlst = args.k_neighbor
    centerlst     = args.center
    pca           = str2bool(args.pca)

    print('dataset_name: %s'
          '\nk_neighborlst: %r'
          '\ncenter: %s'%(dataset_name,k_neighborlst,centerlst))

    # set output dir for feature array
    wild_outdir_k = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/wild/npz'%dataset_name
    mutant_outdir_k = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/mutant/npz/'%dataset_name
    if not os.path.exists(wild_outdir_k):
        os.makedirs(wild_outdir_k)
    if not os.path.exists(mutant_outdir_k):
        os.makedirs(mutant_outdir_k)

    wild_csv_path   = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/wild/csv'%dataset_name
    mutant_csv_path = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/mutant/csv'%dataset_name

    wild_runner(keys,k_neighborlst,centerlst,wild_csv_path,pca,wild_outdir_k)
    mutant_runner(keys,k_neighborlst,centerlst, mutant_csv_path, pca, mutant_outdir_k)
# ----------------------------------------------------------------------------------------------------------------------

def wild_runner(keys,k_neighborlst,centerlst,wild_csv_path,pca,wild_outdir_k):
    for k_neighbor in k_neighborlst:
        for center in centerlst:
            wild_csvdirlst = [wild_csv_path+'/'+x+'/'+'center_%s_neighbor_%s.csv'%(center,k_neighbor) for x in os.listdir(wild_csv_path)]
            x,y,ddg = array_generator(keys,k_neighbor,center,wild_csvdirlst,pca)

            filename = 'center_%s_PCA_%s_neighbor_%s' % (center, pca, k_neighbor)
            save_data_array(x, y, ddg, filename, wild_outdir_k)

# ----------------------------------------------------------------------------------------------------------------------

def mutant_runner(keys,k_neighborlst,centerlst,mutant_csv_path,pca,mutant_outdir_k):
    for k_neighbor in k_neighborlst:
        for center in centerlst:
            mutant_csvdirlst = [mutant_csv_path+'/'+x+'/'+'center_%s_neighbor_%s.csv'%(center,k_neighbor) for x in os.listdir(mutant_csv_path)]
            x,y,ddg = array_generator(keys,k_neighbor,center,mutant_csvdirlst,pca)

            filename = 'center_%s_PCA_%s_neighbor_%s' % (center, pca, k_neighbor)
            save_data_array(x, y, ddg, filename, mutant_outdir_k)

# ----------------------------------------------------------------------------------------------------------------------

def array_generator(keys,k_neighbor, center, csvdirlst, pca=False):
    ddglst = []
    ylst   = []
    arrlst = []
    for csvdir in csvdirlst:
        df = read_csv(csvdir)
        ddg = df.loc[:, 'ddg'].values[0]
        ddglst.append(ddg)
        if ddg >= 0:
            ylst.append(1)
        else:
            ylst.append(0)
        tmp_arr = df.loc[:, keys].values
        if pca:
            try:
                prefix = '/'.join(csvdir.split('/')[:-1])
                center_coord_dir = '%s/center_%s_neighbor_%s_center_coord.npy'%(prefix,center,k_neighbor)
            except:
                center_coord_dir = '%s/center_%s_neighbor_all_center_coord.npy'%(prefix,center)
            center_coord = np.load(center_coord_dir)
            tmp_arr[:, 1:4] = transform(tmp_arr[:, 1:4], center_coord)

        arrlst.append(tmp_arr)

    x = np.array(arrlst).reshape(-1, k_neighbor, len(keys))
    ddg = np.array(ddglst).reshape(-1, 1)
    y = np.array(ylst).reshape(-1, 1)
    assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]

    return x,y,ddg

# ----------------------------------------------------------------------------------------------------------------------

# for radii in radiuslst:
#     r_arrlst = []
#     for i in range(len(arrlst)):
#         center_coord = center_coordlst[i]
#         arr = arrlst[i]
#         indices = arr[:, 0] <= radii
#         r_arr = arr[indices]
#         if pca:
#             r_arr[:, 1:4] = transform(r_arr[:, 1:4], center_coord)
#         r_arrlst.append(r_arr)
#     max_atom_num = max(list(map(lambda x:x.shape[0], r_arrlst)))
#     for i in range(len(r_arrlst)):
#         r_arr = r_arrlst[i]
#         gap = max_atom_num - r_arr.shape[0]
#         assert gap >= 0
#         if gap > 0:
#             gap_array = np.zeros((gap, col_num))
#             r_arrlst[i] = np.vstack((r_arr, gap_array))
#     x = np.array(r_arrlst).reshape(-1, max_atom_num, col_num)
#     ddg = np.array(ddglst).reshape(-1, 1)
#     y = np.array(ylst).reshape(-1, 1)
#     assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]
#     filename = '%s_center_%s_PCA_%s_radius_%s' % (dataset_name, center, pca, radii)
#     save_data_array(x, y, ddg, filename, outdir_r)

if __name__ == '__main__':
    main()