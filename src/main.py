#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from processing import read_csv
'''
the spatial feature (mCSM, mCNN arrray feature, RSA by stride) are based on refined structure and mutant structure constructed by rosetta
'''

dataset_name = sys.argv[1]
#-----------------------------------------------------------------------------------------------------------------------
## drop duplicates of the original mutant csv and rewrite.
print('\n***drop duplicates of the original mutant csv...')
mt_csv_dir = '../dataset/%s/%s.csv'%(dataset_name,dataset_name)
df = read_csv(csvdir=mt_csv_dir)
len_1 = len(df)
df.drop_duplicates(subset=['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT'],keep='first',inplace=True)
len_2 = len(df)
if len_1 > len_2:
    print('---%s data in the original file.'%len_1)
    print('---%s data remained after dropping duplicates.'%len_2)
    os.system('mv %s ../dataset/%s/%s_old.csv' % (mt_csv_dir, dataset_name, dataset_name))
    df.to_csv(mt_csv_dir,index=False)
else:
    print('---no duplicates were found.')
#-----------------------------------------------------------------------------------------------------------------------
## make parent paths
pathdict = {'log_dir': '../dataset/%s/log'             %dataset_name,
            'err_dir': '../dataset/%s/err'             %dataset_name,
            'bak_dir': '../dataset/%s/bak'             %dataset_name,
            'pdb_dir': '../dataset/%s/pdb'             %dataset_name,

            'rosetta_dir'   : '../dataset/%s/feature/rosetta' %dataset_name,
            'msa_dir'       : '../dataset/%s/feature/msa'     %dataset_name,
            'stride_dir'    : '../dataset/%s/feature/stride'  %dataset_name,

            'mCNN_dir'      : '../dataset/%s/feature/mCNN'   %dataset_name,
            'mCSM_dir'      : '../dataset/%s/feature/mCSM'    %dataset_name,

            'train_data_dir': '../dataset/%s/train_data'      %dataset_name}

for path_name in pathdict:
    if not os.path.exists(pathdict[path_name]):
        os.makedirs(pathdict[path_name])
#-----------------------------------------------------------------------------------------------------------------------
## clear old files
print('\n***Cleaning old files...')
os.system('rm -rf %s/*' % pathdict['log_dir'])
os.system('rm -rf %s/*' % pathdict['err_dir'])
os.system('rm -rf %s/*' % pathdict['msa_dir'])
os.system('rm -rf %s/*' % pathdict['rosetta_dir'])
os.system('rm -rf %s/*' % pathdict['stride_dir'])

print('---cleaning done!')
#-----------------------------------------------------------------------------------------------------------------------
#########################################
## calculate features
#########################################
## generate mdl 0
print('\n***Calculating mdl0...')
os.system('./Rosetta/CalRosetta.py %s first'%dataset_name)

## msa by unrefined structure_mdl0
print('\n***Calculating msa feature on mdl_0...')
# os.system('./MSA/CalMSA.py %s'%dataset_name)
os.system('nohup ./MSA/CalMSA.py %s > %s/cal_msa_features.log 2>&1 &'%(dataset_name,pathdict['log_dir']))

## rosetta
print('\n***Calculating rosetta feature...')
os.system('./Rosetta/CalRosetta.py %s second'%dataset_name)

## stride
print('\n***Calculating stride feature...')
os.system('./Stride/CalSA.py %s'%dataset_name) #stride based on refined and mutant structures

##