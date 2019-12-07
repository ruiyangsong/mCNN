#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from mCNN.processing import read_csv, aa_123dict, shell

def qsub_ref(homedir,app,pdbpath,outpath,mt_csvdir):
    df = read_csv(mt_csvdir)
    df_pdbid = df.loc[:,'PDB']
    df_pdbid.drop_duplicates(keep='first', inplace=True)
    pdbid_lst = df_pdbid.values
    for pdbid in pdbid_lst:
        pdbdir = '%s/%s.pdb'%(pdbpath,pdbid)
        qsubid = 'rosetta_ref_%s'%pdbid
        tag    = pdbid
        qsubdir = '%s/%s/qsub_log' %(outpath,tag)
        if not os.path.exists(qsubdir):
            os.system('mkdir -p %s' %qsubdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err'%qsubdir
        outfile = '%s/out'%qsubdir
        run_prog = '%s/%s/run_prog.sh' % (outpath,tag)
        os.system('cp %s %s/%s'%(pdbdir,outpath,tag))

        g = open(run_prog, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('cd %s/%s\n'%(outpath,tag))
        g.writelines('%s %s\n' % (app, pdbid))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('%s/bin/getQ.pl'% homedir)
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.1)

def qsub_mut(homedir,app,pdbpath,outpath,mt_csvdir):
    df = read_csv(mt_csvdir)
    for i in range(len(df)):
        key, pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT = df.iloc[i, :7]
        mt_aa  = aa_123dict[MUTANT]
        pdbdir = '%s/%s/%s_ref.pdb'%(pdbpath,pdbid,pdbid)
        tag = '%s_%s_%s_%s_%s_%s'%(pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT)
        qsubid = 'rosetta_mut_%s'%tag
        qsubdir = '%s/%s/qsub_log' %(outpath,tag)
        if not os.path.exists(qsubdir):
            os.system('mkdir -p %s' %qsubdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err'%qsubdir
        outfile = '%s/out'%qsubdir
        run_prog = '%s/%s/run_prog.sh' % (outpath,tag)
        os.system('cp %s %s/%s'%(pdbdir,outpath,tag))

        g = open(run_prog, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('cd %s/%s\n'%(outpath,tag))
        g.writelines('%s %s_ref %s %s\n' % (app, pdbid, POSITION_NEW, mt_aa))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('%s/bin/getQ.pl'% homedir)
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.1)

if __name__ == '__main__':
    dataset_name = sys.argv[1]

    HOMEdir = shell('echo $HOME')
    app = '%s/mCNN/src/Rosetta/ref.py' % HOMEdir
    pdbpath = '%s/mCNN/dataset/%s/feature/rosetta/global_output/pdb_mdl0' % (HOMEdir, dataset_name)
    outpath = '%s/mCNN/dataset/%s/feature/rosetta/local_output' % (HOMEdir, dataset_name)
    mt_csvdir = '%s/mCNN/dataset/%s/feature/rosetta/global_output/%s.csv' % (HOMEdir,dataset_name,dataset_name)

    if not os.path.exists(outpath):
        os.system('mkdir -p %s' % outpath)
    qsub_ref(HOMEdir,dataset_name, app, pdbpath, outpath, mt_csvdir)