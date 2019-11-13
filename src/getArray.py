#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, argparse
import numpy as np
from sklearn.decomposition import PCA
from processing import read_csv,str2bool, save_data_array

def transform(coord_array_before,center_coord):
    assert len(coord_array_before) >= 3  # row number.
    pca_model = PCA(n_components=3)
    pca_model.fit(coord_array_before)
    coord_array_after = pca_model.transform(coord_array_before)
    center_coord_after = pca_model.transform(center_coord.reshape(-1, 3))
    coord_array_after = coord_array_after - center_coord_after
    return coord_array_after

print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))

radiuslst=[]
k_neighborlst=[]
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',       type=str,  help='dataset_name')
parser.add_argument('-r', '--radius',     nargs='*', type=float, help='All the radius, separated with space')
parser.add_argument('-k', '--k_neighbor', nargs='*', type=int, help='All the k_neighbors, separated with space')
parser.add_argument('-C', '--center',     type=str,  default='CA', choices=['CA','geometric'], help='The MT site center type')
parser.add_argument('-T', '--PCA',        type=str,  default='False', help='If consider PCA transform, default = False.')
parser.add_argument('--mCNN', type=str,choices=['only','append','merge'],default='only',help='only is mCNN; append is append mCSM to mCNN; merge is two part of features, i.e. mCNN and mCSM, default is only')
parser.add_argument('--max',  type=float, help='max range of wildtype environment')
parser.add_argument('--step', type=float, help='cut off step')
parser.add_argument('--class_num', type=int, choices=[2,8], help='atom classification scheme')
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
print('dataset_name: %s'
      '\nradiuelst: %r'
      '\nk_neighborlst: %r'
      '\npca: %s'
      '\nmCNN: %s'
      '\nmax: %s'
      '\nstep: %s'
      '\natom_class_num: %s'%(dataset_name,radiuslst,k_neighborlst,pca,mCNN,maximum,step,class_num))


outdir_k = '/public/home/sry/mCNN/datasets_array/%s/k_neighbor'%dataset_name
outdir_r = '/public/home/sry/mCNN/datasets_array/%s/radius'%dataset_name
csv_path = '/public/home/sry/mCNN/datasets/%s/csv_feature%s_%s'%(dataset_name,dataset_name,center)


keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor', 's_G', 's_H', 's_b', 's_C', 's_T', 's_B', 's_E', 'sa', 'rsa',
        'asa', 'phi', 'psi', 'ph', 'temperature', 'C', 'O', 'N', 'Other', 'hydrophobic', 'positive', 'negative', 'neutral',
        'acceptor', 'donor', 'aromatic', 'sulphur', 'C_mass', 'O_mass', 'N_mass', 'S_mass', 'dC', 'dH', 'dO', 'dN', 'dOther',
        'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur', 'dEntropy',
        'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M', 'WT_F',
        'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-', 'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E',
        'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-']
col_num = len(keys)

center_coord = np.array([0,0,0])
k_lst = []# atom number lst of each mutation
arrlst = []
ddglst = []
ylst = []
r_arrlst = []

csvdirlst = [csv_path+'/'+x+'/'+x+'.csv' for x in os.listdir(csv_path)]

for csvdir in csvdirlst:
    npydir = csvdir[:csvdir.rfind('_')] + '/center_coord.npy'
    npydir = npydir.replace('feature','pdb')
    center_coord = np.load(npydir)
    df = read_csv(csvdir)
    ddg = df.loc[:, 'ddg'].values[0]
    if ddg>=0:
        ylst.append(1)
    else:
        ylst.append(0)
    arr = df.loc[:, keys].values
    arrlst.append(arr)
    ddglst.append(ddg)
    k_lst.append(len(df))
k_max, k_min = max(k_lst), min(k_lst)
del df

for k_neighbor in k_neighborlst:
    assert max(k_neighborlst) <= k_min ## The max number of neighbors should not greater than k_min !!!
    k_arrlst = []
    for arr in arrlst:
        indices = arr[:,0].argsort()
        k_indices = sorted(indices[0:k_neighbor])
        if center == 'CA':
            center_coord = arr[indices[0],1:4]
        elif center == 'geometric':
            res_arr = df.
            center_coord =
        k_arr = arr[k_indices]
        if pca:
            k_arr[:,1:4] = transform(k_arr[:,1:4],center_coord)
        k_arrlst.append(k_arr)
    x = np.array(k_arrlst).reshape(-1,k_neighbor,col_num)
    ddg = np.array(ddglst).reshape(-1,1)
    y = np.array(ylst).reshape(-1,1)
    assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]

    filename = '%s_neighbor_%s'%(dataset_name,k_neighbor)
    save_data_array(x,y,ddg,filename,outdir_k)

# for radii in radiuslst:
#     pass

print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))