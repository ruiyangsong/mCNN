#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, argparse
from processing import read_csv, aa_123dict,shell

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str,help='dataset name')
args = parser.parse_args()
dataset_name = args.dataset_name
HOMEdir   = shell('echo $HOME')
app       = '%s/mCNN/TrRosetta/ref.py'  %HOMEdir
pdbpath   = '%s/mCNN/datasets/%s/pdb%s' %(HOMEdir,dataset_name,dataset_name)
outpath   = '%s/mCNN/datasets/%s/ref%s' %(HOMEdir,dataset_name,dataset_name)
mt_csvdir = '%s/mCNN/datasets/%s/%s_new_map.csv' %(HOMEdir,dataset_name,dataset_name)

df = read_csv(mt_csvdir)
for i in range(len(df)):
    key, PDB, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT = df.iloc[i, :7]
    if dataset_name == 'S1925':
        pdbid = PDB
    elif dataset_name == 'S2648':
        pdbid  = PDB[:-5]
    mt_aa  = aa_123dict[MUTANT]
    pdbdir = '%s/%s_mdl0.pdb'%(pdbpath,pdbid)
    tag = '%s_%s_%s_%s_%s_%s'%(pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT)
    qsubdir = '%s/%s/qsub_log' %(outpath,tag)
    if not os.path.exists(qsubdir):
        os.system('mkdir -p %s' %qsubdir)
    walltime = 'walltime=240:00:00'
    errfile = '%s/err'%qsubdir
    outfile = '%s/out'%qsubdir
    run_prog = '%s/%s/run_prog.sh' % (outpath,tag)
    os.system('cp %s %s/%s/%.pdb'%(pdbdir,outpath,tag,pdbid))

    g = open(run_prog, 'w+')
    g.writelines('#!/usr/bin/env bash\n')
    g.writelines('dataset_name=\"%s\"\n' % dataset_name)
    g.writelines('echo $dataset_name\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('cd %s/%s\n'%(outpath,tag))
    g.writelines('%s %s %s %s\n' % (app, pdbid, POSITION_NEW, mt_aa))
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_prog)
    os.system('%s/bin/getQ.pl'% HOMEdir)
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_prog))
    time.sleep(0.1)