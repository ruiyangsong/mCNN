#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, argparse
import numpy as np
from processing import read_csv, str2bool, save_data_array, transform

print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))
radiuslst     = []
k_neighborlst = []
## parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',       type=str,  help='dataset_name')
parser.add_argument('-r', '--radius',     nargs='*', type=float,      help='All the radius, separated with space')
parser.add_argument('-k', '--k_neighbor', nargs='*', type=int,        help='All the k_neighbors, separated with space')
parser.add_argument('-C', '--center',     type=str,  default='CA',    choices=['CA','geometric'], required=True, help='The MT site center type')
parser.add_argument('-T', '--PCA',        type=str,  default='False', help='If consider PCA transform, default = False.')
parser.add_argument('--mCNN', type=str,   choices=['only','append'],  default='only',help='only is mCNN; append is append mCSM to mCNN, i.e. mCNN and mCSM, default is only')
parser.add_argument('--max',  type=float, help='max range of wildtype environment')
parser.add_argument('--step', type=float, help='cut off step')
parser.add_argument('--class_num',        type=int,  choices=[2,8],   help='atom classification scheme')
args = parser.parse_args()
dataset_name = args.dataset_name
if args.radius:
    radiuslst = args.radius
if args.k_neighbor:
    k_neighborlst = args.k_neighbor
if args.center:
    center = args.center
if args.PCA:
    pca = str2bool(args.PCA)
if args.mCNN:
    mCNN = args.mCNN
if args.max:
    maximum = args.max
if args.step:
    step = args.step
if args.class_num:
    class_num = args.class_num
# print('dataset_name: %s'
#       '\nradiuslst: %r'
#       '\nk_neighborlst: %r'
#       '\npca: %s'
#       '\nmCNN: %s'
#       '\nmax: %s'
#       '\nstep: %s'
#       '\natom_class_num: %s'%(dataset_name,radiuslst,k_neighborlst,pca,mCNN,maximum,step,class_num))
## set output dir for feature array
outdir_k = '/public/home/sry/mCNN/datasets_array/%s/k_neighbor'%dataset_name
outdir_r = '/public/home/sry/mCNN/datasets_array/%s/radius'%dataset_name
outdir_k_append = '/public/home/sry/mCNN/datasets_array/%s/append_k_neighbor'%dataset_name
outdir_r_append = '/public/home/sry/mCNN/datasets_array/%s/append_radius'%dataset_name
## mCSM array directory
mCSMpath = '/public/home/sry/mCNN/datasets_array/%s/mCSM/%s'%(dataset_name,center)
mCSMpathlst = [mCSMpath + '/' + x for x in os.listdir(mCSMpath)]
mCSMdirlst  = [x + '/' + [y for y in os.listdir(x) if y[-3:]=='npz'][0] for x in mCSMpathlst]
## csv_feature directory
csv_path = '/public/home/sry/mCNN/datasets/%s/csv_feature%s_%s'%(dataset_name,dataset_name,center)
csvdirlst = [csv_path+'/'+x+'/'+x+'.csv' for x in os.listdir(csv_path)]
## The concerned features
keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor', 's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C', 's_Helix',
        's_Strand', 's_Coil', 'sa', 'rsa', 'asa', 'phi', 'psi', 'ph', 'temperature', 'C', 'O', 'N', 'Other',
        'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
        'hydrophobic_bak', 'polar', 'C_mass', 'O_mass', 'N_mass', 'S_mass', 'dC', 'dH', 'dO', 'dN', 'dOther',
        'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',
        'dhydrophobic_bak', 'dpolar', 'dEntropy', 'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G',
        'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M', 'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
        'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
        'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-']

col_num = len(keys)

center_coordlst = []
k_lst  = [] # atom number lst of each mutation
# dfarrlst = []
arrlst = []
ddglst = []
ylst   = []

for csvdir in csvdirlst:
    npydir = csv_path.replace('feature','pdb')+'/MT_'+'_'.join(csvdir.split('/')[-2].split('_')[1:-1])+'/center_coord.npy'
    center_coordlst.append(np.load(npydir))
    df = read_csv(csvdir)
    ddg = df.loc[:, 'ddg'].values[0]
    if ddg>=0:
        ylst.append(1)
    else:
        ylst.append(0)
    arrlst.append(df.loc[:, keys].values)
    # dfarrlst.append(df.loc[:, keys])
    ddglst.append(ddg)
    k_lst.append(len(df))
k_max, k_min, k_avg, std = max(k_lst), min(k_lst), np.mean(k_lst), np.std(k_lst)
# print('k_max: %s, k_min: %s, k_avg: %s, std: %s'%(k_max, k_min, k_avg, std))
# print('directory of the maximum atoms of this dataset:', csvdirlst[k_lst.index(max(k_lst))])
# print('directory of the minimum atoms of this dataset:', csvdirlst[k_lst.index(min(k_lst))])
del df

# generate mCNN features for each k_neighbor in k_neighbor list.
for k_neighbor in k_neighborlst:
    assert max(k_neighborlst) <= k_min ## The max number of neighbors should not greater than k_min !!!
    k_arrlst = []
    for i in range(len(arrlst)):
        center_coord = center_coordlst[i]
        arr = arrlst[i]
        indices = arr[:,0].argsort()
        k_indices = sorted(indices[0:k_neighbor])
        k_arr = arr[k_indices]
        if pca:
            k_arr[:,1:4] = transform(k_arr[:,1:4],center_coord)
        k_arrlst.append(k_arr)
    x = np.array(k_arrlst).reshape(-1,k_neighbor,col_num)
    ddg = np.array(ddglst).reshape(-1,1)
    y = np.array(ylst).reshape(-1,1)
    assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]
    filename = '%s_center_%s_PCA_%s_neighbor_%s'%(dataset_name,center,pca,k_neighbor)
    save_data_array(x,y,ddg,filename,outdir_k)

for radii in radiuslst:
    r_arrlst = []
    for i in range(len(arrlst)):
        center_coord = center_coordlst[i]
        arr = arrlst[i]
        indices = arr[:, 0] <= radii
        r_arr = arr[indices]
        if pca:
            r_arr[:, 1:4] = transform(r_arr[:, 1:4], center_coord)
        r_arrlst.append(r_arr)
    max_atom_num = max(list(map(lambda x:x.shape[0], r_arrlst)))
    for i in range(len(r_arrlst)):
        r_arr = r_arrlst[i]
        gap = max_atom_num - r_arr.shape[0]
        assert gap >= 0
        if gap > 0:
            gap_array = np.zeros((gap, col_num))
            r_arrlst[i] = np.vstack((r_arr, gap_array))
    x = np.array(r_arrlst).reshape(-1, max_atom_num, col_num)
    ddg = np.array(ddglst).reshape(-1, 1)
    y = np.array(ylst).reshape(-1, 1)
    assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]
    filename = '%s_center_%s_PCA_%s_radius_%s' % (dataset_name, center, pca, radii)
    save_data_array(x, y, ddg, filename, outdir_r)
print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))