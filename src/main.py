#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time
from processing import read_csv,check_pid

'''
the spatial feature (mCSM, mCNN arrray feature, RSA by stride) are based on refined structure and mutant structure which constructed by pyrosetta.
'''
if len(sys.argv) == 1:
    print('Usage: ./main.py [dataset_name]')
    sys.exit(0)

dataset_name = sys.argv[1]
#-----------------------------------------------------------------------------------------------------------------------
## drop duplicates of the original mutant csv and rewrite it, the original one namely %s_old.csv.
## the primary key is: ['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT'].
print('\n***drop duplicates of the original mutant csv (dataset %s)...'%dataset_name)
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
## make parent paths for each dataset (or each commiting mutation set).
pathdict = {'log_dir': '../dataset/%s/log'             %dataset_name,
            'err_dir': '../dataset/%s/err'             %dataset_name,
            'bak_dir': '../dataset/%s/bak'             %dataset_name,
            'pdb_dir': '../dataset/%s/pdb'             %dataset_name,
            'npz_dir': '../dataset/%s/npz'             %dataset_name,

            'rosetta_dir'    : '../dataset/%s/feature/rosetta' %dataset_name,
            'msa_dir'        : '../dataset/%s/feature/msa'     %dataset_name,
            'stride_dir'     : '../dataset/%s/feature/stride'  %dataset_name,
            # 'mCSM_dir'       : '../dataset/%s/feature/mCSM'    %dataset_name,
            'mCNN_dir'       : '../dataset/%s/feature/mCNN'    %dataset_name}

for path_name in pathdict:
    if not os.path.exists(pathdict[path_name]):
        os.makedirs(pathdict[path_name])
#-----------------------------------------------------------------------------------------------------------------------
##########################################
#          clear old files               #
# !!! USE COMMAND "rm -rf" WITH CARE !!! #
##########################################
# print('\n***Cleaning old files...')
# os.system('rm -rf %s/*' % pathdict['log_dir'])
# os.system('rm -rf %s/*' % pathdict['bak_dir'])
# os.system('rm -rf %s/*' % pathdict['err_dir'])
# os.system('rm -rf %s/*' % pathdict['npz_dir'])
# os.system('rm -rf /public/home/sry/mCNN/dataset/%s/feature/*'%dataset_name)

# os.system('rm -rf %s/*' % pathdict['msa_dir'])
# os.system('rm -rf %s/*' % pathdict['rosetta_dir'])
# os.system('rm -rf %s/*' % pathdict['stride_dir'])
# os.system('rm -rf %s/*' % pathdict['mCNN_dir'])
# os.system('rm -rf %s/*' % pathdict['mCSM_dir'])
# print('---cleaning done!')
#-----------------------------------------------------------------------------------------------------------------------
## calculating features
run_code = 0

## generate mdl 0
if run_code == 0:
    print('\n***Calculating mdl0...')
    run_code += os.system('./Rosetta/CalRosetta.py %s first'%dataset_name)

## msa by unrefined structure_mdl0
if run_code == 0:
    print('\n***Calculating msa feature based on mdl_0...')
    strf_time = time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())
    print('running as nohup')
    nohup_pid = os.popen('nohup ./MSA/CalMSA.py %s > %s/msa.%s.log 2>&1 &echo $!'%(dataset_name,pathdict['log_dir'],strf_time)).readlines()[0].strip()
    run_code += 0

## rosetta
if run_code == 0:
    print('\n***Calculating rosetta feature...')
    run_code += os.system('./Rosetta/CalRosetta.py %s second'%dataset_name)

## stride
if run_code == 0:
    print('\n***Calculating stride feature...')
    run_code += os.system('./Stride/CalSA.py %s'%dataset_name) #stride based on refined and mutant structures

# ## mCSM
# if run_code == 0:
    # print('\n***Calculating mCSM feature...')
    # run_code += os.system('./Spatial/run_coord.py %s --flag first -k 5 --center CA geometric -T False'%dataset_name)#@@++
    # # run_code += os.system('./Spatial/run_mCSM.py %s'%(dataset_name))
    # run_code += os.system('nohup ./Spatial/run_mCSM.py %s > %s/run_mCSM.log 2>&1 &'%(dataset_name,pathdict['log_dir']))

## mCNN
if run_code == 0:
    print('\n***Calculating mCNN feature...')
    check_pid(nohup_pid)
    run_code += os.system('./Spatial/run_coord.py %s --flag all -k 120 110 50 40 60 80 100 --center CA -T False'%dataset_name)
## mCNN on two clusters
# if run_code == 0:
    # print('\n***Calculating mCNN feature...')
    # from processing import shell
    # homedir = shell('echo $HOME')
    # if homedir == '/home/sry':
    #     print('---On server ibm')
    #     run_code += os.system('./Spatial/run_coord.py %s --flag all -k 130 140 150 160 170 180 190 200 --center CA geometric -T False' % dataset_name)
    # elif homedir == '/public/home/sry':
    #     print('---On server hp')
    #     run_code += os.system('./Spatial/run_coord.py %s --flag all -k 30 40 50 60 70 80 90 100 110 120 --center CA geometric -T False' % dataset_name)