#!/usr/bin/env python
# -*- coding: utf-8 -*-

'File name format: MT_pdb_wtaa_chain_position_mtaa_serial'

import os, sys, time
import pandas as pd

dataset_name = sys.argv[1]
# dataset_name = 'S1925'
# MT_csvdir = '/home/ruiyang/HardDrive/Archive/projects/mCNN/ieee_access/supplymentary_data/datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
# outpath = '/home/ruiyang/HardDrive/Archive/projects/mCNN/ieee_access/supplymentary_data/datasets/%s/csv_pdb%s'%(dataset_name,dataset_name)
# pdbpath = '/home/ruiyang/HardDrive/Archive/projects/mCNN/ieee_access/supplymentary_data/datasets/%s/pdb%s'%(dataset_name,dataset_name)
# app = '/home/ruiyang/HardDrive/Archive/projects/mCNN/ieee_access/supplymentary_data/src/CalNeighbor.py'

MT_csvdir = '/public/home/sry/mCNN/datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
outpath = '/public/home/sry/mCNN/datasets/%s/csv_pdb%s'%(dataset_name,dataset_name)
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
    g.writelines('%s %s %s %s -o %s -n %s\n' % (app, pdbdir,chain,position,outdir,filename))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_CalNeighbor)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_CalNeighbor))
    time.sleep(1)