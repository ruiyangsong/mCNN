#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time
# dataset_name = sys.argv[1]
dataset_name = 'S2648'
user = 'sry'
app = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'
libdir = '/public/library/uniprot20/uniprot20'
datadir = '/public/home/sry/mCNN/datasets/%s/seq%s'%(dataset_name,dataset_name)
outdir = '/public/home/sry/mCNN/msa/psiblast%s'%dataset_name

file_lst= os.listdir(datadir)
for file in file_lst:
    filename = file[:-6] # without extensions.
    tag = 'psiblast_%s'%(filename)
    tmpdir = '/tmp/sry/%s'%tag

    os.system('mkdir -p %s/qsublog'%tmpdir)
    os.system('cp %s/file %s'%(datadir,tmpdir))
    os.chdir(tmpdir)
    walltime = 'walltime = 24:00:00'
    errfile = './qsublog/err'
    outfile = './qsublog/out'
    run_psiblast = 'run_psiblast_%s.sh'%filename

    g = open(run_psiblast, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines('dataset_nam = %s\n'%dataset_name)
    g.writelines('outdir = %s\n'%outdir)
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('%s -query %s -db %s -out blast.out -num_iterations 3\n'%(app,file,libdir))
    g.writelines("echo 'end at:' `date`\n")
    g.writelines('mv -r . ${outdir}')
    g.close()

    os.system('chmod 755 %s' % run_psiblast)
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_psiblast))
    print('%s submitted.'%file)

    time.sleep(1)