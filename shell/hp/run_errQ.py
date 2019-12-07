#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, argparse
from processing import read_csv, aa_123dict,shell

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str,help='dataset name')
args = parser.parse_args()
dataset_name = args.dataset_name
HOMEdir   = shell('echo $HOME')
app       = '%s/mCNN/TrRosetta/ref.py'%HOMEdir
pdbpath   = '%s/mCNN/datasets/%s/pdb%s'%(HOMEdir,dataset_name,dataset_name)
outpath   = '%s/mCNN/datasets/%s/ref%s'%(HOMEdir,dataset_name,dataset_name)

tag_errdir = '%s/errlst_%s'%(HOMEdir,dataset_name)

with open(tag_errdir) as f:
    lines = f.readlines()
for line in lines:
    pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT = line.strip().split('_')

    mt_aa  = aa_123dict[MUTANT]
    pdbdir = '%s/%s.pdb'%(pdbpath,pdbid)
    tag = '%s_%s_%s_%s_%s_%s'%(pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT)
    qsubdir = '%s/%s/qsub_log' %(outpath,tag)
    if not os.path.exists(qsubdir):
        os.system('mkdir -p %s' %qsubdir)
    walltime = 'walltime=240:00:00'
    errfile = '%s/err'%qsubdir
    outfile = '%s/out'%qsubdir
    run_prog = '%s/%s/run_prog.sh' % (outpath,tag)

    os.system('%s/bin/getQ.pl'%HOMEdir)
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_prog))
    time.sleep(0.1)