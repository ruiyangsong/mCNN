#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time
import pandas as pd
dataset_name, saAPP = sys.argv[1:]
if saAPP != 'stride':
    print('-'*10+'please use program stride!!!')
# dataset_name = 'S2648'
# MTCSVDIR    = '../../datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
# COORDCSVPATH = '../../datasets/%s/csv_pdb%s'%(dataset_name,dataset_name)
# OUTPATH      = '../../datasets/%s/csv_feature%s'%(dataset_name,dataset_name)

app          = '/public/home/sry/mCNN/src/AppendFeature.py'
MTCSVDIR     = '/public/home/sry/mCNN/datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
COORDCSVPATH = '/public/home/sry/mCNN/datasets/%s/csv_pdb%s'%(dataset_name,dataset_name)
OUTPATH      = '/public/home/sry/mCNN/datasets/%s/csv_feature%s'%(dataset_name,dataset_name)
f = open(MTCSVDIR,'r')
MT_df = pd.read_csv(f)
f.close()
row_lst = [MT_df.iloc[i,:] for i in range(len(MT_df))]
value_lst = list(map(lambda df: df.values, row_lst))
for values in value_lst:
    if dataset_name == 'S2648':
        key, pdb, wtaa, mtchain, mtposition, mtaa, ph, temperature, ddg = values #S2648
    elif dataset_name == 'S1925':
        key, pdb, wtaa, mtchain, mtposition, mtaa, ph, temperature, ddg, rsa = values #S1925
    pdb = pdb[0:4]
    sadir = '/public/home/sry/mCNN/datasets/%s/%s%s/%s.%s'%(dataset_name,saAPP,dataset_name,pdb,saAPP)
    filename = '%s_%s_%s_%s_%s_%s_%04d' % (dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, key+1)
    coordcsvdir = '%s/MT_%s_%s_%s_%s_%s/MT_%s_%s_%s_%s_%s.csv'%(COORDCSVPATH,pdb,wtaa,mtchain,mtposition,mtaa,pdb,wtaa,mtchain,mtposition,mtaa)
    print('========== The File name is: %s ==========' % filename)
    tag = filename
    outdir = '%s/%s' % (OUTPATH, tag)
    if not os.path.exists('%s/qsublog' % outdir):
        os.system('mkdir -p %s/qsublog' % outdir)
    walltime = 'walltime=24:00:00'
    errfile = '%s/qsublog/err' % outdir
    outfile = '%s/qsublog/out' % outdir
    run_Append = '%s/run_Append_%s.sh' % (outdir, filename)

    g = open(run_Append, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines('dataset_name=\"%s\"\n' % dataset_name)
    g.writelines('echo $dataset_name\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    # './AppendFeature.py %s %s -o %s -f rsa thermo onehot deltar msa ddg -r %s -t %s %s -d %s\n'%(COORDCSVPATH,filename,outdir,rsa,ph,temperature,ddg)
    g.writelines('%s %s %s -o %s -f rsa thermo onehot mass deltar msa ddg -S %s -t %s %s -d %s\n' % (app, coordcsvdir, filename, outdir, sadir, ph, temperature, ddg))
    # if dataset_name == 'S1925':
    #     g.writelines('%s %s %s -o %s -f rsa thermo onehot deltar msa ddg -r %s -t %s %s -d %s\n'%(app,coordcsvdir,filename,outdir,rsa,ph,temperature,ddg))
    # elif dataset_name == 'S2648':
    #     g.writelines('%s %s %s -o %s -f thermo onehot deltar msa ddg -t %s %s -d %s\n' % (app,coordcsvdir, filename, outdir, ph, temperature, ddg))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_Append)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_Append))
    time.sleep(1)