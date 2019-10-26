#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, argparse
import numpy as np
from sklearn.decomposition import PCA
from processing import read_csv


# dataset_name, radius, k_neighbor = sys.argv[1:]

print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))
pca = True
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str, help='dataset_name')
parser.add_argument('-r', '--radius', nargs='*', type=float, help='All the radius, separated with space')
parser.add_argument('-k', '--k_neighbor', nargs='*', type=int, help='All the k_neighbors, separated with space')
parser.add_argument('-T', '--PCA',        type=bool, help='If consider PCA transform, default = True.')
args = parser.parse_args()
dataset_name = args.dataset_name
if args.radius:
    radiuslst = args.radius
if args.k_neighbor:
    k_neighborlst = args.k_neighbor
    # print(k_neighborlst)
if args.PCA:
    pca = args.PCA

outdir_k = '/public/home/sry/mCNN/datasets_array/%s/k_neighbor'%dataset_name
outdir_r = '/public/home/sry/mCNN/datasets_array/%s/radius'%dataset_name
csv_path = '/public/home/sry/mCNN/datasets/%s/csv_feature%s'%(dataset_name,dataset_name)

def save_data_array(x,y,ddg_value,filename,outdir):
    np.savez('%s/%s.npz' % (outdir,filename), x=x,y=y,ddg=ddg_value)
    print('The 3D array which stored numerical representation has stored in local hard drive.')

def transform(coord_array_before,center_coord):
    assert len(coord_array_before) >= 3  # row number.
    pca = PCA(n_components=3)
    pca.fit(coord_array_before)
    coord_array_after = pca.transform(coord_array_before)
    center_coord_after = pca.transform(center_coord.reshape(-1, 3))
    coord_array_after = coord_array_after - center_coord_after
    return coord_array_after

if dataset_name == 'S2648':
    col_num = 59
    keys = ['dist', 'x', 'y', 'z', 'b_factor', 'ph', 'temperature', 'C', 'O', 'N', 'Other', 'dC', 'dH', 'dO', 'dN', 'dOther', 'dEntropy',
            'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M', 'WT_F',
            'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-', 'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E',
            'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-']
elif dataset_name == 'S1925':
    col_num = 60
    keys = ['dist','x','y','z','b_factor','rsa','ph','temperature','C','O','N','Other','dC','dH','dO','dN','dOther','dEntropy',
            'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M', 'WT_F',
            'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-', 'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E',
            'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-']

k_lst = []# atom number lst of each mutation

arrlst = []
ddglst = []
ylst = []

r_arrlst = []
csvdirlst = [csv_path+'/'+x+'/'+x+'.csv' for x in os.listdir(csv_path)]

for csvdir in csvdirlst:
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
    assert max(k_neighborlst) <= k_min  ## The max number of neighbors should not greater than k_min !!!
    k_arrlst = []
    for arr in arrlst:
        indices = arr[:,0].argsort()
        k_indices = sorted(indices[0:k_neighbor])
        center_coord = arr[indices[0],1:4]
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