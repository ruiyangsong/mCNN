#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
根据rosetta生成的pdb_ref 和 pdb_mut计算mCNN特征并保存至csv，
其中不考虑rosetta mut 失败的条目（根据rosetta结果反向去mt_csv 中查找ddg温度等特征指标）
'''

'File name format: MT_pdb_wtaa_chain_position_mtaa_serial'

import os, time, argparse
import pandas as pd
from mCNN.processing import str2bool, shell, read_csv

#-----------------------------------------------------------------------------------------------------------------------
## parse argument
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',   type=str, help='dataset name')
parser.add_argument('-C', '--center', type=str, choices=['CA','geometric'], default='CA', help='The MT site center type, default is CA.')
parser.add_argument('-P', '--PCA',    type=str, default='False', help='if consider PCA, default is False')
args = parser.parse_args()
dataset_name = args.dataset_name
center       = args.center
pca          = args.PCA
#-----------------------------------------------------------------------------------------------------------------------
## Set required directory variables
homedir            = shell('echo $HOME')
app                = '%s/mCNN/src/Spatial/mCNN.py'%homedir
mt_csv_dir         = '%s/mCNN/dataset/%s/%s.csv'%(homedir, dataset_name, dataset_name)
ref_pdb_dir        = '%s/mCNN/dataset/%s/feature/rosetta/ref_output'%(homedir,dataset_name)
mut_pdb_dir        = '%s/mCNN/dataset/%s/feature/rosetta/mut_output'%(homedir,dataset_name)

ref_mCNN_outdir = '%s/mCNN/dataset/%s/feature/mCNN/wild/center_%s_pca_%s'%(homedir,dataset_name,center,pca)
mut_mCNN_outdir = '%s/mCNN/dataset/%s/feature/mCNN/mutant/center_%s_pca_%s'%(homedir,dataset_name,center,pca)
if not os.path.exists(ref_mCNN_outdir):
    os.makedirs(ref_mCNN_outdir)
if not os.path.exists(mut_mCNN_outdir):
    os.makedirs(mut_mCNN_outdir)
#-----------------------------------------------------------------------------------------------------------------------
for rosetta_mut_tag in os.listdir(mut_pdb_dir):
    pdbid, wtaa, chain, pos, pos_new, mtaa = rosetta_mut_tag.split()






f     = open(MTCSVDIR,'r')
MT_df = pd.read_csv(f)
f.close()
row_lst   = [MT_df.iloc[i,:] for i in range(len(MT_df))]
value_lst = list(map(lambda df: df.values, row_lst))
for values in value_lst:
    if dataset_name == 'S2648':
        key, pdb, wtaa, mtchain, mtposition, mtaa, ph, temperature, ddg = values #S2648
    elif dataset_name == 'S1925':
        key, pdb, wtaa, mtchain, mtposition, mtaa, ph, temperature, ddg, rsa = values #S1925
    pdb         = pdb[0:4]
    sadir       = '/public/home/sry/mCNN/datasets/%s/%s%s/%s.%s'%(dataset_name,saAPP,dataset_name,pdb,saAPP)
    filename    = '%s_%s_%s_%s_%s_%s_%04d' % (dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, key+1)
    coordcsvdir = '%s/MT_%s_%s_%s_%s_%s/MT_%s_%s_%s_%s_%s.csv'%(COORDCSVPATH,pdb,wtaa,mtchain,mtposition,mtaa,pdb,wtaa,mtchain,mtposition,mtaa)
    print('========== The File name is: %s ==========' % filename)
    tag    = filename
    outdir = '%s/%s' % (OUTPATH, tag)
    if not os.path.exists('%s/qsublog'  % outdir):
        os.system('mkdir -p %s/qsublog' % outdir)

    walltime   = 'walltime=24:00:00'
    errfile    = '%s/qsublog/err' % outdir
    outfile    = '%s/qsublog/out' % outdir
    run_Append = '%s/run_Append_%s.sh' % (outdir, filename)

    g = open(run_Append, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines('dataset_name=\"%s\"\n' % dataset_name)
    g.writelines('echo $dataset_name\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('%s %s %s -o %s -f rsa thermo onehot pharm hp mass deltar pharm_deltar hp_deltar msa ddg -S %s -t %s %s -d %s\n'
                 % (app, coordcsvdir, filename, outdir, sadir, ph, temperature, ddg))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_Append)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_Append))
    time.sleep(0.1)













parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str,help='dataset name')
parser.add_argument('center', type=str, choices=['CA','geometric'], default='CA', help='The MT site center type')
args = parser.parse_args()
dataset_name = args.dataset_name
center = args.center














MT_csvdir = '/public/home/sry/mCNN/datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
outpath = '/public/home/sry/mCNN/datasets/%s/csv_pdb%s_%s'%(dataset_name,dataset_name,center)
pdbpath = '/public/home/sry/mCNN/datasets/%s/pdb%s'%(dataset_name,dataset_name)
app = '/public/home/sry/mCNN/src/CalNeighbor.py'

f = open(MT_csvdir, 'r')
MT_df = pd.read_csv(f)
f.close()
MT_df = MT_df.loc[:,['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT']]
MT_df.drop_duplicates(keep='first',inplace=True)
len_df = len(MT_df)
print('********** drop_duplicates on %s, now datanum is: %s **********'%(dataset_name,len_df))
for i in range(len_df):
    line = MT_df.iloc[i, :]
    pdbid, wtaa, chain, position, mtaa = line.PDB[:4], line.WILD_TYPE, line.CHAIN, line.POSITION, line.MUTANT
    pdbdir = '%s/%s.pdb'%(pdbpath,pdbid)
    filename = 'MT_%s_%s_%s_%s_%s'%(pdbid,wtaa,chain,position,mtaa)
    print('========== The File name is: %s =========='%filename)
    # os.system('/public/home/sry/mCNN/src/CalNeighbor.py %s %s %s -o %s -n %s'%(pdbdir,chain,position,outdir,filename))
    tag = filename
    outdir = '%s/%s' % (outpath, tag)
    if not os.path.exists('%s/qsublog'%outdir):
        os.system('mkdir -p %s/qsublog' %outdir)
    walltime = 'walltime=24:00:00'
    errfile = '%s/qsublog/err'%outdir
    outfile = '%s/qsublog/out'%outdir
    run_CalNeighbor = '%s/run_CalNeighbor_%s.sh' % (outdir,filename)

    g = open(run_CalNeighbor, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines('dataset_name=\"%s\"\n' % dataset_name)
    g.writelines('echo $dataset_name\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('%s %s %s %s %s -o %s -C %s\n' % (app, pdbdir,chain,position,filename,outdir,center))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_CalNeighbor)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_CalNeighbor))
    time.sleep(0.1)

#----------------------- for row pdb data -----------------------#

MT_csvdir = '/public/home/sry/mCNN/datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
outpath = '/public/home/sry/mCNN/datasets/%s/csv_pdb%s_%s'%(dataset_name,dataset_name,center)
pdbpath = '/public/home/sry/mCNN/datasets/%s/pdb%s'%(dataset_name,dataset_name)
app = '/public/home/sry/mCNN/src/CalNeighbor.py'

f = open(MT_csvdir, 'r')
MT_df = pd.read_csv(f)
f.close()
MT_df = MT_df.loc[:,['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT']]
MT_df.drop_duplicates(keep='first',inplace=True)
len_df = len(MT_df)
print('********** drop_duplicates on %s, now datanum is: %s **********'%(dataset_name,len_df))
for i in range(len_df):
    line = MT_df.iloc[i, :]
    pdbid, wtaa, chain, position, mtaa = line.PDB[:4], line.WILD_TYPE, line.CHAIN, line.POSITION, line.MUTANT
    pdbdir = '%s/%s.pdb'%(pdbpath,pdbid)
    filename = 'MT_%s_%s_%s_%s_%s'%(pdbid,wtaa,chain,position,mtaa)
    print('========== The File name is: %s =========='%filename)
    # os.system('/public/home/sry/mCNN/src/CalNeighbor.py %s %s %s -o %s -n %s'%(pdbdir,chain,position,outdir,filename))
    tag = filename
    outdir = '%s/%s' % (outpath, tag)
    if not os.path.exists('%s/qsublog'%outdir):
        os.system('mkdir -p %s/qsublog' %outdir)
    walltime = 'walltime=24:00:00'
    errfile = '%s/qsublog/err'%outdir
    outfile = '%s/qsublog/out'%outdir
    run_CalNeighbor = '%s/run_CalNeighbor_%s.sh' % (outdir,filename)

    g = open(run_CalNeighbor, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines('dataset_name=\"%s\"\n' % dataset_name)
    g.writelines('echo $dataset_name\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('%s %s %s %s %s -o %s -C %s\n' % (app, pdbdir,chain,position,filename,outdir,center))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_CalNeighbor)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_CalNeighbor))
    time.sleep(0.1)