#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
'''
the spatial feature (mCSM, mCNN arrray feature, RSA by stride) are based on refined structure and mutant structure constructed by rosetta
'''

## get mutant csv

dataset_name = sys.argv[1]
## make parent paths
pathdict = {'log_dir': '../dataset/%s/log'             %dataset_name,
            'err_dir': '../dataset/%s/err'             %dataset_name,
            'bak_dir': '../dataset/%s/bak'             %dataset_name,
            'pdb_dir': '../dataset/%s/pdb'             %dataset_name,

            'msa_dir'       : '../dataset/%s/feature/msa'     %dataset_name,
            'stride_dir'    : '../dataset/%s/feature/stride'  %dataset_name,
            'mCSM_dir'      : '../dataset/%s/feature/mCSM'    %dataset_name,
            'spatial_dir'   : '../dataset/%s/feature/spatial' %dataset_name,
            'rosetta_dir'   : '../dataset/%s/feature/rosetta' %dataset_name,
            'merge_dir'     : '../dataset/%s/feature/merge'   %dataset_name,
            'train_data_dir': '../dataset/%s/train_data'      %dataset_name}

for path_name in pathdict:
    if not os.path.exists(pathdict[path_name]):
        os.makedirs(pathdict[path_name])

########################################
## clear old files
########################################
print('\n***Cleaning old files...')
os.system('rm -rf %s/*'%pathdict['err_dir'])
os.system('rm -rf %s/*'%pathdict['rosetta_dir'])
os.system('rm -rf %s/*'%pathdict['stride_dir'])
os.system('rm -rf %s/*'%pathdict['msa_dir'])

#########################################
## calculate features
#########################################

## msa by unrefined structure

## rosetta
print('\n***Calculating rosetta feature...')
os.system('./Rosetta/CalRosetta.py %s'%dataset_name)

## stride
print('\n***Calculating stride feature...')
os.system('./Stride/CalSA.py %s'%dataset_name) #stride based on refined and mutant structures

## msa by refined structure
# os.system('./MSA/CalMSA.py %s'%dataset_name)

##